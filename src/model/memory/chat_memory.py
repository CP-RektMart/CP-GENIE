from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain.vectorstores.base import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from src.model.generator.generate import build_prompt

CHAT_HISTORY_STORE: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in CHAT_HISTORY_STORE:
        CHAT_HISTORY_STORE[session_id] = ChatMessageHistory()
    return CHAT_HISTORY_STORE[session_id]

def get_chat(
    session_id: str, 
    k: int, 
    llm: ChatGoogleGenerativeAI, 
    store: VectorStore
) -> RunnableWithMessageHistory:
    retriever = store.as_retriever(search_type="similarity", k=k)

    def prepare_prompt(input_dict: dict) -> str:
        question = input_dict["input"]
        docs = retriever.invoke(question)
        memory = get_session_history(session_id)
        return build_prompt(memory, docs, question)

    return RunnableWithMessageHistory(
        RunnableLambda(prepare_prompt) | llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
