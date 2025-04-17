from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from model.generator.generate import build_prompt
from model.retriever.retrieve import retrieve_docs
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from typing import Dict

vector_store = None
llm = None

@tool(response_format="content_and_artifact")
def retrieve(query: str) -> tuple[str, list[Document]]:
    """Retrieve documents from the vector store based on the query."""
    retrieved_docs = retrieve_docs(query, k=10)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: Dict) -> Dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools_node = ToolNode([retrieve])

def generate(state: Dict) -> Dict:
    """Generate a response based on the retrieved documents."""
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    tool_text = "\n\n".join(m.content for m in tool_messages)

    sys_msg = SystemMessage(
        content=(
            "You are an assistant for question-answering tasks. Use the following "
            "retrieved context to answer the question. If unsure, say 'I don't know'.\n\n"
            + tool_text
        )
    )

    usable_messages = [
        m for m in state["messages"]
        if isinstance(m, BaseMessage) and hasattr(m, "content") and m.content
    ]

    prompt = [sys_msg] + usable_messages

    response = llm.invoke(prompt)  # ✅ Now this will never raise “content not specified”
    return {"messages": [response]}

memory = MemorySaver()

def build_agentic_graph(loaded_llm, loaded_vector_store) -> StateGraph:
    global llm, vector_store
    llm = loaded_llm
    vector_store = loaded_vector_store

    builder = StateGraph(dict)
    builder.add_node("query_or_respond", query_or_respond)
    builder.add_node("tools", tools_node)
    builder.add_node("generate", generate)

    builder.set_entry_point("query_or_respond")
    builder.add_conditional_edges("query_or_respond", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=memory)