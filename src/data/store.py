import os
import dotenv
from langchain.schema import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from data.scrape import scrape_docs
from model.embedding.embedder import get_dense_embedder, get_sparse_embedder, chunk_docs

dotenv.load_dotenv()

COLLECTION_NAME = "cp-genie"

def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def create_qdrant_vectorstore(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_dense_embedder(),
        sparse_embedding=get_sparse_embedder(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

#  when setup and load qdrant to store data
def setup_qdrant(split_docs: list[Document]) -> QdrantVectorStore:
    client = create_qdrant_client()
    client.recreate_collection(
        collection_name="cp-genie",
        vectors_config={
            "dense": VectorParams(size=os.getenv("EMBEDDING_SIZE"), 
                                  distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )
    qdrant = create_qdrant_vectorstore(client)
    qdrant.add_documents(documents=split_docs)
    return qdrant

# when need to use
def load_qdrant() -> QdrantVectorStore:
    client = create_qdrant_client()
    return create_qdrant_vectorstore(client)

# Utility function to run the full pipeline of scraping, chunking, and embedding
def scrape_and_store() -> None:
    raw_docs = scrape_docs()
    split_docs = chunk_docs(raw_docs)
    setup_qdrant(split_docs)
