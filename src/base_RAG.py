from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import requests

def load_and_split_documents(documents_folder):
    """
    Загрузка документов из локальной директории и разбиение их на фрагменты.
    """
    # Загрузка документов
    loader = DirectoryLoader(documents_folder, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Разбиение документов на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(texts):
    """
    Создание эмбеддингов и сохранение их в векторном хранилище ChromaDB.
    """
    # Использование 'DeepPavlov/rubert-base-cased' в качестве модели эмбеддингов
    embedding_model_name = 'DeepPavlov/rubert-base-cased'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Создание векторного хранилища ChromaDB
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    return vectorstore

def call_api(prompt: str) -> str:
    url = "YOUR_API_ENDPOINT"  # Замените на фактический URL API
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_API_KEY"  # Если требуется аутентификация
    }
    payload = {
        "modelUri": "gpt://<идентификатор_каталога>/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": "Найди ошибки в тексте и исправь их"
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
        # Извлечение сгенерированного текста из ответа
        generated_text = result.get('text', '')
        return generated_text
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def run_query(retriever, query):
    """
    Запуск поиска и генерация ответа с использованием API.
    """
    # Получение релевантных документов
    docs = retriever.get_relevant_documents(query)
    
    # Формирование контекста из документов
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Создание полного запроса с учетом контекста
    prompt = f"Контекст:\n{context}\n\nВопрос:\n{query}"
    
    # Вызов API с предоставленным промптом
    response = call_api(prompt)
    
    # Возврат результата и исходных документов
    result = {
        "result": response,
        "source_documents": docs
    }
    return result

# Основное выполнение
if __name__ == "__main__":
    # Параметры
    documents_folder = "./documents"  # Путь к папке с документами
    query = "В чем смысл жизни?"  # Пользовательский запрос
    
    # Загрузка и разбиение документов
    texts = load_and_split_documents(documents_folder)
    
    # Создание векторного хранилища
    vectorstore = create_vectorstore(texts)
    
    # Настройка ретривера
    retriever = vectorstore.as_retriever()
    
    # Запуск запроса
    result = run_query(retriever, query)
    
    # Вывод ответа и исходных документов
    print("Ответ:")
    print(result["result"])
    print("\nИсходные документы:")
    for doc in result["source_documents"]:
        print(doc.metadata["source"])
