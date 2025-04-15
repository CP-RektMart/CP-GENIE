from langchain.vectorstores.base import VectorStoreRetriever
from src.data.store import load_qdrant
from langchain.schema import Document

def get_retriever(k: int = 10) -> VectorStoreRetriever:
    qdrant = load_qdrant()
    return qdrant.as_retriever(search_type="similarity", k=k)


def retrieve_docs(query: str, k: int = 10) -> list[Document]:
    retriever = get_retriever(k)
    return retriever.get_relevant_documents(query)
