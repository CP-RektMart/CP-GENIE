from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from cp_genie.domain.rag.base import State, BaseRAG


class NormalRAG(BaseRAG):
    def _build_graph(self) -> StateGraph:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer the user's query based on the provided context and chat history. Answer concisely.",
                ),
                (
                    "human",
                    "Retrieved context:\n\n{context}\n\n---\
                    \n\nChat History:\n{messages}\n\n---\
                    \n\nQuery: {query}\nPlease answer the query based on the context and history.",
                ),
            ]
        )
        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        def retrieve(state: State) -> dict:
            query = self.memory.get_lastest_message().content
            docs = self.retriever.invoke(query)
            return {"context": docs}

        def generate(state: State) -> dict:
            query = self.memory.get_lastest_message().content
            result = combine_chain.invoke(
                {
                    "messages": self.memory.get_history_message(),
                    "context": state.get("context", []),
                    "query": query,
                }
            )
            self.memory.add_ai_message(result)
            return {"messages": [result]}

        graph = StateGraph(State)
        graph.add_node("retrieve", RunnableLambda(retrieve))
        graph.add_node("generate", RunnableLambda(generate))

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()
