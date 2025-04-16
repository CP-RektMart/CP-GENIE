from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    query: str
    k: int = 10

class MessageHistoryItem(BaseModel):
    type: str
    content: str

class ChatResponse(BaseModel):
    answer: str
    history: List[MessageHistoryItem]