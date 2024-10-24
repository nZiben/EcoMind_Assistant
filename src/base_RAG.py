from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

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

def create_vectorstore(texts, embedding_model_name, opensearch_url, index_name):
    """
    Создание эмбеддингов и сохранение их в векторном хранилище OpenSearch.
    """
    # Создание эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Создание векторного хранилища OpenSearch
    vectorstore = OpenSearchVectorSearch.from_documents(
        documents=texts,
        embedding=embeddings,
        opensearch_url=opensearch_url,
        index_name=index_name
    )
    return vectorstore

def set_up_llm(model_name):
    """
    Настройка локальной языковой модели (LLM).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Создание пайплайна генерации текста с использованием LLM
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Обёртка пайплайна в HuggingFacePipeline для совместимости с LangChain
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

def build_qa_chain(llm, retriever):
    """
    Построение цепочки RetrievalQA с использованием LLM и ретривера.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def run_query(qa_chain, query):
    """
    Запуск цепочки QA с пользовательским запросом и возврат результата.
    """
    result = qa_chain({"query": query})
    return result

# Основное выполнение
if __name__ == "__main__":
    # Параметры
    documents_folder = "./documents"  # Путь к папке с документами
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Имя модели эмбеддингов
    model_name = "gpt2-medium"  # Имя локальной языковой модели
    opensearch_url = "http://localhost:9200"  # URL вашего OpenSearch экземпляра
    index_name = "langchain_index"  # Имя индекса в OpenSearch
    query = "В чем смысл жизни?"  # Пользовательский запрос
    
    # Загрузка и разбиение документов
    texts = load_and_split_documents(documents_folder)
    
    # Создание векторного хранилища
    vectorstore = create_vectorstore(texts, embedding_model_name, opensearch_url, index_name)
    
    # Настройка ретривера
    retriever = vectorstore.as_retriever()
    
    # Настройка LLM
    llm = set_up_llm(model_name)
    
    # Построение цепочки QA
    qa_chain = build_qa_chain(llm, retriever)
    
    # Запуск запроса
    result = run_query(qa_chain, query)
    
    # Вывод ответа и исходных документов
    print("Ответ:")
    print(result["result"])
    print("\nИсходные документы:")
    for doc in result["source_documents"]:
        print(doc.metadata["source"])
