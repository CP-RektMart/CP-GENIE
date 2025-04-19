from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cp_genie.api.v1.router import router as v1_router
from cp_genie.core.config import Settings
from cp_genie.core.logging import configure_logging
from cp_genie.infrastructure.llm import get_llm
# from cp_genie.infrastructure.redis import redis
from cp_genie.infrastructure.qdrant import client
from cp_genie.infrastructure.qdrant import initialize_vectorstore


settings = Settings()
configure_logging()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    # redis.ping()
    app.state.retriever = initialize_vectorstore()
    app.state.llm = get_llm()
    logger.info("external services ready")
    yield
    client.close()
    # redis.close()


app = FastAPI(
    title="CP-Genie",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["_internal"])
async def health():
    return {"status": "ok"}