from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from cp_genie.domain.rag.base import State

class NormalRAG:
    def __init__(self, llm, retriever, memory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.chain = self._build_graph()

    def _build_graph(self) -> StateGraph:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer as concisely as possible."),
            ("human", "chat history: {messages}\ndocuments: {context}\nquestion: {question}"),
        ])
        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        def retrieve(state) -> State:
            docs = self.retriever.invoke(state["question"])
            return {**state, "context": docs}

        def generate(state) -> State:
            result = combine_chain.invoke({
                "messages": state["messages"],
                "context": state["context"],
                "question": state["question"],
            })
            self.memory.add_user_message(state["question"])
            self.memory.add_ai_message(result)
            return {**state, "output": result}

        graph = StateGraph(State)
        graph.add_node("retrieve", RunnableLambda(retrieve))
        graph.add_node("generate", RunnableLambda(generate))

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def invoke(self, input: dict) -> State:
        state: State = {
            "messages": self.memory.messages,
            "question": input["input"],
            "context": [],
            "output": ""
        }
        return self.chain.invoke(state)