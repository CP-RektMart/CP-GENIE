from fastapi import FastAPI
from contextlib import asynccontextmanager
from data.store import load_qdrant
from model.generator.generate import get_llm
from api.router import normal, agentic
from api.utils.error_handlers import add_exception_handlers
from typing import AsyncGenerator
from fastapi.routing import APIRouter

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.llm = get_llm()
    app.state.qdrant = load_qdrant()
    yield

app = FastAPI(lifespan=lifespan)
v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(normal.router, prefix="/normal")
v1_router.include_router(agentic.router, prefix="/agentic")

app.include_router(v1_router)
add_exception_handlers(app)
