from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from qdrant_client import QdrantClient
from cp_genie.core.config import Settings

settings = Settings()
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_embedding_model)
def initialize_vectorstore():
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})
