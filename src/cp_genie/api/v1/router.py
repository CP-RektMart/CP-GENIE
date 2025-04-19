from fastapi import APIRouter, Depends
from fastapi import Request

from cp_genie.domain.rag.normal import NormalRAG
from cp_genie.api.v1.schema import ChatRequest, ChatResponse
from cp_genie.infrastructure.chat_memory import get_by_session_id

router = APIRouter(tags=["chat"])

RAG_CLASSES = {
    "normal":   NormalRAG,
    # "agentic":  AgenticRAG,
}

def get_retriever(req: Request) -> str:
    return req.app.state.retriever

def get_llm(req: Request) -> str:
    return req.app.state.llm

@router.post("/{rag_type}/{session_id}")
async def chat(
    rag_type: str,
    session_id: str,
    chatrequest: ChatRequest,
    retriever: str = Depends(get_retriever),
    llm: str = Depends(get_llm),
) -> ChatResponse:
    
    if rag_type not in RAG_CLASSES:
        raise ValueError(f"Invalid rag_type: {rag_type}")
    if not session_id:
        raise ValueError("session_id is required")
    if not chatrequest.query:
        raise ValueError("query is required")
    
    rag_class = RAG_CLASSES[rag_type]
    memory = get_by_session_id(session_id)
    chain = rag_class(llm, retriever, memory)
    result = chain.invoke({"input": chatrequest.query})
    history = memory.get_messages()
    
    return ChatResponse(
        answer=result["output"],
        history=[{"type": msg.type, "content": msg.content} for msg in history]
    )
