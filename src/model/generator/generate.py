from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv
from langchain.schema import Document

dotenv.load_dotenv()

SYSTEM_PROMPT = """
    You are a helpful assistant. You will be provided with a context and a question.
    Your task is to answer the question based on the context.
    You should not provide any additional information or context.
    If the answer is not in the context, say "I don't know".
    If the question is not clear, ask for clarification.
    You should be concise and to the point.
    Do not repeat the question in your answer.
    Do not include any disclaimers or unnecessary information.
"""

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS"))
    )


def get_context(docs: list[Document]) -> str:
    context = ""
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content
        context += f"Title: {title}\nContent: {content}\n\n"
    return context


def generate_answer(llm: ChatGoogleGenerativeAI, query: str, context: str) -> str:
    sys_prompt = SYSTEM_PROMPT
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.predict(prompt)