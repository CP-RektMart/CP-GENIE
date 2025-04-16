from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from api.schema.chat import ChatRequest, ChatResponse
from model.memory.chat_memory import get_chat, get_session_history

router = APIRouter()

@router.post("/{session_id}", response_model=ChatResponse)
def ask_question(req: Request, session_id: str, chatrequest: ChatRequest):
    llm = req.app.state.llm
    qdrant = req.app.state.qdrant
    chain = get_chat(session_id, chatrequest.k, llm, qdrant)
    result = chain.invoke(
        {"input": chatrequest.query},
        config={"configurable": {"session_id": session_id}}
    )
    history = get_session_history(session_id)
    messages = [
        {"type": type(m).__name__, "content": m.content} 
        for m in history.messages
    ]
    return ChatResponse(answer=result.content, history=messages)