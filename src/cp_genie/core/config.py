from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    sitemap_url: str
    groq_api_key: str
    google_api_key: str
    
    langsmith_api_key: str
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str
    qdrant_collection_name: str = "cp_genie"
    redis_url: str = "redis://localhost:6379"
    
    langsmith_tracing: bool = True
    langchain_tracing_v2: bool = True
    
    llm_model: str = "gemini-2.0-flash-001"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256
    
    embedding_model: str = "sentence-transformers/LaBSE"
    embedding_size: int = 768
    sparse_embedding_model: str = "Qdrant/bm25"
    
    reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )
