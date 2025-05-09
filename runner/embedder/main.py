import os
import sqlalchemy
import re
import emoji
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from models import Base, Metadata, Content, Embedded
# --- Load env variables ---
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Metadata, Content
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

load_dotenv(override=True)

MODEL_KWARGS = {'trust_remote_code': True}
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"
CROSSENCODER_MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

QDRANT_URL ="https://ff9ebd5e-31f8-4e7a-a866-ef23b9b9cd5e.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BATCH_SIZE = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Set your Qdrant endpoint and API key
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

if not client.collection_exists(collection_name="cp-genie"):
  client.recreate_collection(
      collection_name="cp-genie", # change this baed on your collection
      vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
      sparse_vectors_config={
          "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
      },
  )
# --- Database connection ---

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
# --- Create tables if they don't exist
Base.metadata.create_all(engine)

def retrieve_text_content():
    """Retrieves text content from the specified table in Neon."""
    try:
        existing_content = {
          c.id: c.content
          for c in session.query(Content.id, Content.content).all()
        }

        return existing_content

    except sqlalchemy.exc.SQLAlchemyError as e:
        session.rollback()
        print(f"An error occurred: {e}")
        return {}

def retrieve_embedded_content():
    """Retrieves embedded id that already embedded from the specified table in Neon."""
    try:
      results = session.query(Embedded.content_id, Embedded.last_updated).all()
      embedded_content_id = {
        r.content_id : r.last_updated
        for r in results
      }

      return embedded_content_id

    except sqlalchemy.exc.SQLAlchemyError as e:
        session.rollback()
        print(f"An error occurred: {e}")

        return {}

def clean_text(text: str) -> str:
    # Remove unwanted characters and whitespace
    cleaned_text = emoji.replace_emoji(text, '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def document_transformer(list_text: list) -> list[Document]:
    docs = [Document(page_content=text) for text in list_text]

    return docs

def document_creator(text_content_list: dict, old_embedded_content_id: dict) -> tuple[list, dict]:
    docs = []
    new_id_docs = {}

    for id, text in tqdm(text_content_list.items(), desc="Retrieving the new content", position=0, leave=True):
        if id in old_embedded_content_id.keys():
            print(f"[SKIPPED] Content {id} already in QDrant")
            continue

        cleaned = clean_text(text)
        docs.append(cleaned)
        new_id_docs[id] = cleaned

    return docs, new_id_docs

# --- retrieve data from database ---
text_content_list = retrieve_text_content()
old_embedded_content_id = retrieve_embedded_content()

# --- model initialization ---

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,model_kwargs=MODEL_KWARGS)
model = HuggingFaceCrossEncoder(model_name=CROSSENCODER_MODEL_NAME)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)

docs, new_id_docs = document_creator(text_content_list, old_embedded_content_id)
qdocs = document_transformer(docs)

# --- Splitter Documents ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # number of characters
    chunk_overlap=100,    # to retain context between chunks
    separators=["\n\n", "\n", ".", " "],
)
split_docs = text_splitter.split_documents(qdocs)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="cp-genie",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

try:
  qdrant.add_documents(documents=split_docs)
  count = 0
  for id, text in new_id_docs.items():
      record = Embedded(
          content_id= id,
          cleaned_content=text,
          last_updated=datetime.now()
      )

      session.merge(record)

      count += 1
      if count % BATCH_SIZE == 0:
          session.commit()
          print(f"[OK] Sent {count} data to DB")
          count = 0

  session.commit()
  print(f"[OK] Sent {count} data to DB")
except Exception as e:
  print(f"[ERROR] Error occured while sending to QDrant: {e}")
  session.rollback()
finally:
  session.close()
