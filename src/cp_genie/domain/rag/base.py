from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]
