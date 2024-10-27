from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import re
import requests
import os
from docx import Document

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def clean_text(text):
    """
    Функция для удаления "мусора" из текста, такого как пустые строки, таблицы,
    лишние пробелы, номера пунктов и лишние символы.
    """
    # Удаление лишних символов табуляции, переносов строк и последовательностей пробелов
    cleaned_text = re.sub(r'\n+', '\n', text)  # Удаление повторяющихся переносов строк
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Замена множественных пробелов на один пробел

    # Удаление строк, содержащих "Таблица" и другие нерелевантные технические данные
    cleaned_text = re.sub(r'(Таблица\s*\d+\.\d+|Приложение\s*\d+)', '', cleaned_text)
    
    # Удаление номеров пунктов и заголовков с номерами (например, 2.3.1, 4 ХАРАКТЕРИСТИКИ ИЗАВ)
    cleaned_text = re.sub(r'(\d+\.\d+(\.\d+)?|^\d+[\sА-Яа-яЁё ]+)', '', cleaned_text)

    # Удаление оставшихся технических фрагментов
    cleaned_text = re.sub(r'[^\w\s,.()ёЁА-Яа-я-]', '', cleaned_text)  # Удаление спецсимволов, кроме пунктуации
    
    # Удаление лишних пробелов, возникающих после чистки
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    
    return cleaned_text

def load_and_split_documents(documents_folder):
    """
    Загрузка документов из локальной директории и разбиение их на фрагменты.
    """
    # Загрузка документов с указанием кодировки utf-8
    loader = DirectoryLoader(
        documents_folder, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    for i in range(len(documents)):
        documents[i].page_content = clean_text(documents[i].page_content)

    # Разбиение документов на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(texts, embedding_model_name):
    """
    Создание эмбеддингов и сохранение их в векторном хранилище ChromaDB.
    """
    # Создание эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Создание векторного хранилища ChromaDB
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    return vectorstore

def call_api_tuned(system_prompt: str, prompt: str):
    """
    Обращается к дообученной на уникальном экологическом QnA-датасете модели и возвращает текстовый результат.
    """   
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion" 
    
    api_key = os.environ.get('YANDEX_API_KEY')
    folder_id = os.environ.get('YANDEX_FOLDER_ID')
    
    if not api_key or not folder_id:
        raise Exception("YANDEX_API_KEY and YANDEX_FOLDER_ID environment variables must be set.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}",  # апи ключ
        "x-folder-id": folder_id  # id каталога
    }
    
    payload = {
        "modelUri": "ds://bt18aohnkrp4njfmrf13",  # модель
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['result']['alternatives'][0]['message']['text']
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def convert_docx_to_txt(folder_path, folder_path_to):
    # Проверяем, существует ли папка
    if not os.path.isdir(folder_path):
        print(f"Папка {folder_path} не найдена.")
        return

    # Проходим по всем файлам в указанной папке
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):  # Проверяем, что это .docx файл
            docx_path = os.path.join(folder_path, filename)
            txt_path = os.path.join(folder_path_to, filename.replace('.docx', '.txt'))

            # Чтение содержимого .docx файла
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            # Запись содержимого в .txt файл
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

            print(f"Файл {filename} успешно преобразован в {txt_path}")

vectorstore = None

@app.on_event("startup")
async def startup_event():
    global vectorstore
    # Конвертация файлов docx в txt
    folder_path = '../documents/put your docx here!'
    folder_path_to = '../documents/processed files'
    convert_docx_to_txt(folder_path, folder_path_to)
    
    # Загрузка и разбиение документов
    documents_folder = '/kaggle/input/ecology-hack'  # Путь к папке с документами
    texts = load_and_split_documents(documents_folder)
    
    # Создание векторного хранилища
    vectorstore = create_vectorstore(texts, "DeepPavlov/rubert-base-cased-sentence")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    global vectorstore
    query = request.query
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    # Формирование контекста из документов
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Контекст: {context}
Вопрос: {query}
Оцени, подходит ли данный контекст для ответа на вопрос! Если контекст является достаточным для ответа, используй его. Если нет, то не пиши в ответе об этом, а сразу отвечай на вопрос! В любом случае, предоставь максимально подробный и точный ответ на вопрос, используя всю свою экспертную информацию.
"""
    system_prompt = "Ты — эксперт в области экологии, защиты окружающей среды и экологического права. Твоя задача — помогать пользователям находить ответы на вопросы об экологии, экологических нормах, правилах и стандартах!"

    try:
        chat_response = call_api_tuned(system_prompt, prompt)
        return {"response": chat_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
