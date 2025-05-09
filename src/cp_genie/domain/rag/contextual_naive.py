from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from cp_genie.domain.rag.base import State, ContextualRAG


class ContextualNaiveRAG(ContextualRAG):
    def _build_graph(self) -> StateGraph:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.sys_prompt
                    + "\n\nAdditional Instructions: When responding to users, focus on providing accurate information from the retrieved context. Present information in a clear, structured manner. If the context doesn't contain relevant information, rely on your general knowledge while clearly indicating this distinction.",
                ),
                (
                    "human",
                    """# Retrieved Context
                    
{context}

# Chat History
{messages}

# Current Query
{query}

Please answer the query based on the retrieved context and conversation history. Be precise and informative. If the retrieved context doesn't contain information directly relevant to the query, clearly state this and provide your best response based on general knowledge.""",
                ),
            ]
        )
        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        def retrieve(state: State) -> dict:
            query = self.memory.get_lastest_message().content
            results_fac = self.retriever_fac.invoke(query)
            result_oth = self.retriever_oth.invoke(query)
            results = results_fac + result_oth
            return {"context": results}

        def generate(state: State) -> dict:
            query = self.memory.get_lastest_message().content
            result = combine_chain.invoke(
                {
                    "messages": self.memory.get_history_message(),
                    "context": state.get("context", ""),
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
