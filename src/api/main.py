from fastapi import FastAPI
from contextlib import asynccontextmanager
from data.store import load_qdrant
from model.generator.generate import get_llm
from api.router import chat
from api.utils.error_handlers import add_exception_handlers

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = get_llm()
    app.state.qdrant = load_qdrant()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router, prefix="/ask")
add_exception_handlers(app)