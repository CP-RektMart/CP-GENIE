from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    sitemap_url: str = "https://www.cp.eng.chula.ac.th/sitemap.xmll"
    groq_api_key: str = ""
    google_api_key: str = ""

    # langsmith_api_key: str = ""

    qdrant_url_naive: str = "http://localhost:6333"
    qdrant_api_key_naive: str = ""

    qdrant_url_contextual: str = "http://localhost:6333"
    qdrant_api_key_contextual: str = ""

    # langsmith_tracing: bool = True
    # langchain_tracing_v2: bool = True

    llm_model: str = "gemini-2.5-flash-preview-04-17"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256

    embedding_model: str = "BAAI/bge-m3"
    embedding_size: int = 1024
    sparse_embedding_model: str = "Qdrant/bm25"

    reranker_model: str = ""

    google_application_credentials: str = ""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        extra="ignore",
        case_sensitive=False,
    )
