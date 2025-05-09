from cp_genie.core.config import Settings
from langchain_google_vertexai import ChatVertexAI

setting = Settings()


def get_llm() -> ChatVertexAI:
    return ChatVertexAI(
        model=setting.llm_model,
        temperature=setting.llm_temperature,
        max_tokens=setting.llm_max_tokens,
    )
