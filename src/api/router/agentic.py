from fastapi import APIRouter, Request
from api.schema.chat import ChatRequest, ChatResponse, MessageHistoryItem
from src.model.agentic.tool_graph import build_agentic_graph

router = APIRouter()

SESSION_MESSAGES: dict[str, list[dict]] = {}

@router.post("/{session_id}", response_model=ChatResponse)
def ask_agentic(session_id: str, request: ChatRequest, req: Request):
    llm = req.app.state.llm
    qdrant = req.app.state.qdrant
    graph = build_agentic_graph(llm, qdrant)
    
    if session_id not in SESSION_MESSAGES:
        SESSION_MESSAGES[session_id] = []

    input_message = {"role": "user", "content": request.query}
    config = {"configurable": {"thread_id": session_id}}

    final_messages = []
    for step in graph.stream({"messages": [input_message]}, stream_mode="values", config=config):
        final_messages.extend(step["messages"])

    answer = final_messages[-1].content if final_messages else ""
    history = [
        MessageHistoryItem(type=m.type, content=m.content)
        for m in final_messages if hasattr(m, "content")
    ]
    return ChatResponse(answer=answer, history=history)
