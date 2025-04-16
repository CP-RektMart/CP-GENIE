import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
import dotenv

dotenv.load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def chunk_docs(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return text_splitter.split_documents(docs)


def get_dense_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL")
    )

def get_sparse_embedder() -> FastEmbedSparse:
    return FastEmbedSparse(
        model_name=os.getenv("SPARSE_EMBEDDING_MODEL")
    )