from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.qdrant import Qdrant
from typing import Optional

llm_instance: Optional[ChatGoogleGenerativeAI] = None
qdrant_instance: Optional[Qdrant] = None
