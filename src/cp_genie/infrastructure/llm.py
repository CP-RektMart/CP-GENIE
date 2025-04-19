from langchain_google_genai import ChatGoogleGenerativeAI
from cp_genie.core.config import Settings

setting = Settings()

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=setting.llm_model,
        temperature=setting.llm_temperature,
        max_tokens=setting.llm_max_tokens,
        streaming=True
    )
