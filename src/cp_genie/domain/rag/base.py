from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]
    query: str


class BaseRAG(ABC):
    def __init__(self, llm, retriever, memory):
        if not all([llm, retriever, memory]):
            raise ValueError("llm, retriever, and memory must be provided.")
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.chain = self._build_graph()
        if not isinstance(self.chain, Runnable):
            raise TypeError(
                "self.chain must be a runnable instance after _build_graph."
            )

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        pass

    def invoke(self, query: str) -> State:
        if not query:
            raise ValueError("Input query cannot be empty.")

        self.memory.add_user_message(HumanMessage(content=query))
        initial_state: State = {
            "messages": self.memory.get_messages(),
            "context": [],
            "query": query,
        }

        final_state = self.chain.invoke(initial_state)
        return final_state
