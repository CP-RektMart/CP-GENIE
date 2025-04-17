from langchain.vectorstores.base import VectorStoreRetriever
from data.store import load_qdrant
from langchain.schema import Document

def get_retriever(k: int = 10) -> VectorStoreRetriever:
    return load_qdrant().as_retriever(
        search_kwargs={"k": k},
        search_type="similarity",
    )

def retrieve_docs(query: str, k: int = 10) -> list[Document]:
    retriever = get_retriever(k)
    return retriever.invoke(query)
