from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from cp_genie.domain.rag.base import State


class NormalRAG:
    def __init__(self, llm, retriever, memory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.chain = self._build_graph()

    def _build_graph(self) -> StateGraph:

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer as concisely as possible."),
                (
                    "human",
                    "chat history: {messages}\nretrieved context: {context}\n",
                ),
            ]
        )
        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        def retrieve(state) -> State:
            last_message = self.memory.get_lastest_message().content
            docs = self.retriever.invoke(last_message)
            return {**state, "context": docs}

        def generate(state) -> State:
            result = combine_chain.invoke(
                {
                    "messages": state["messages"],
                    "context": state.get("context", []),
                }
            )
            self.memory.add_ai_message(result)
            updated_messages = self.memory.get_messages()
            return {**state, "messages": updated_messages}

        graph = StateGraph(State)
        graph.add_node("retrieve", RunnableLambda(retrieve))
        graph.add_node("generate", RunnableLambda(generate))

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def invoke(self, input) -> State:
        self.memory.add_user_message(HumanMessage(content=input))
        initial_state: State = {
            "messages": self.memory.get_messages(),
            "context": [],
        }
        return self.chain.invoke(initial_state)
