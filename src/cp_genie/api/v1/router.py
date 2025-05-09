from fastapi import APIRouter, Depends
from fastapi import Request

from cp_genie.domain.rag.normal import NormalRAG
from cp_genie.domain.rag.agentic import AgenticRAG

# from cp_genie.domain.rag.contextual import ContextualRAG
from cp_genie.api.v1.schema import ChatRequest, ChatResponse, MessageHistoryItem
from cp_genie.infrastructure.chat_memory import get_by_session_id

router = APIRouter(tags=["chat"])

RAG_CLASSES_NAIVE = {
    "normal": NormalRAG,
    "agentic": AgenticRAG,
}

# RAG_CLASSES_CONTEXTUAL = {
#     "normal": ContextualNaiveRAG,
#     "agentic": ContextualAgenticRAG,
# }


def get_retriever_naive_reranked(req: Request) -> str:
    return req.app.state.retriever_naive_reranked


def get_retriever_fac_reranked(req: Request) -> str:
    return req.app.state.retriever_fac_reranked


def get_retriever_oth_reranked(req: Request) -> str:
    return req.app.state.retriever_oth_reranked


def get_llm(req: Request) -> str:
    return req.app.state.llm


@router.post("/naive/{rag_type}/{session_id}")
async def chat(
    rag_type: str,
    session_id: str,
    chatrequest: ChatRequest,
    retriever: str = Depends(get_retriever_naive_reranked),
    llm: str = Depends(get_llm),
) -> ChatResponse:

    if rag_type not in RAG_CLASSES_NAIVE:
        raise ValueError(f"Invalid rag_type: {rag_type}")
    if not session_id:
        raise ValueError("session_id is required")
    if not chatrequest.query:
        raise ValueError("query is required")

    rag_class = RAG_CLASSES_NAIVE[rag_type]
    memory = get_by_session_id(session_id)
    chain = rag_class(llm, retriever, memory)
    result = chain.invoke(chatrequest.query)
    history = memory.get_messages()

    history_items: list[MessageHistoryItem] = []
    for msg in history:
        history_items.append(
            MessageHistoryItem(
                type=msg.type,
                content=str(msg.content),
            )
        )

    return ChatResponse(
        answer=result["messages"][-1].content,
        history=history_items,
    )
