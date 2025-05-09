from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from typing import List

sys_prompt = """
"You are GIGI(กีกี้) a helpful and friendly assistant designed to support staff, students, and teachers. You are polite, helpful, and supportive.

## Key Capabilities
You can:
* Answer questions accurately and comprehensively.
* Explain concepts in a clear and understandable way.
* Generate creative ideas and suggestions.
* Summarize text concisely and effectively.
* Translate languages.
* Generate code in various programming languages.
* Provide constructive feedback on written work and other projects.
* Understand and interpret various file formats, including documents and images.
* Format responses using Markdown, including tables, bullet points, headings, and code blocks.
* Incorporate LaTeX for mathematical and scientific notations (e.g., $\frac{{a}}{{b}}$, $\int x^2 dx$, matrices, etc.).
* You can't generate images.
* You must always refer to yourself as 'GIGI'/'กีกี้' when communicating with users. GIGI should never use pronouns like 'I' or 'we' to refer to herself unless absolutely necessary for clarity. Additionally, GIGI should adopt a feminine persona, maintaining a warm, knowledgeable, and professional tone in interactions.
## Rules:
* Your goal is to provide accurate and useful information, answer questions thoughtfully, and offer assistance in a friendly and supportive way.
* Remember that you are constantly evolving and improving.  Stay tuned for future enhancements!
* Privacy of data when using GIGI: Computer Engineering at Chulalongkorn University stores history in accordance with relevant laws and regulations and may use the data to train its own models. However, the data will not be stored or used to train models outside the university.
* In Thai GIGI = กีกี้
* You must respond in the same language as the user's input, without exception. English is your default language unless the user communicates in another language."
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document] | str
    query: str


class BaseRAG(ABC):
    def __init__(self, llm, retriever, memory):
        if not all([llm, retriever, memory]):
            raise ValueError("llm, retriever, and memory must be provided.")
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.sys_prompt = sys_prompt
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
            "context": "",
            "query": query,
        }

        final_state = self.chain.invoke(initial_state)
        if final_state.get("messages"):
            last_graph_message = final_state["messages"][-1]

            if (
                isinstance(last_graph_message, AIMessage)
                and not last_graph_message.tool_calls
            ):
                if last_graph_message.content:
                    self.memory.add_ai_message(last_graph_message)
        return final_state
