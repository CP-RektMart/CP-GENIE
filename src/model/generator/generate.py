from langchain.schema import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import get_buffer_string
from langchain_google_genai import ChatGoogleGenerativeAI
import os

SYSTEM_PROMPT = """
You are a helpful assistant. You will be provided with a context and a question.
Your task is to answer the question based on the context.
If the answer is not in the context, say "I don't know".
"""

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS"))
    )
    
def build_prompt(history: BaseChatMessageHistory, docs: list[Document], query: str) -> str:
    chat_str = get_buffer_string(history.messages)
    context_str = "\n".join(doc.page_content for doc in docs)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Chat History:\n{chat_str}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Q: {query}\nA:"
    )
