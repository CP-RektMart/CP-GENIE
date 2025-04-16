from langchain.vectorstores.base import VectorStoreRetriever
from data.store import load_qdrant
from api.core.config import get_vector_store
from langchain.schema import Document

def get_retriever(k: int = 10) -> VectorStoreRetriever:
    return get_vector_store().as_retriever(search_type="similarity", k=k)

def retrieve_docs(query: str, k: int = 10) -> list[Document]:
    retriever = get_retriever(k)
    return retriever.invoke(query)
