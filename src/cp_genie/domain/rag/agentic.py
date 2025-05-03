from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, cast
from langchain_core.documents import Document
from cp_genie.domain.rag.base import State, BaseRAG


class AgenticRAG(BaseRAG):
    def _build_graph(self) -> StateGraph:

        @tool
        def retrieve(query: str) -> List[Document]:
            """
            Retrieves information related to the input query
            from a vector database containing information
            on Computer Engineering at Chulalongkorn University.
            Use this tool ONLY when the user asks a question that requires specific
            knowledge about the university or department. Do not use for general conversation.
            """
            docs = self.retriever.invoke(query)
            return [d.page_content for d in docs]

        tools = [retrieve]
        tool_node = ToolNode(tools)
        llm_with_tools = self.llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.sys_prompt,
                ),
                (
                    "human",
                    "Retrieved context:\n\n{context}\n\n---\
                    \n\nChat History:\n{messages}\n\n---\
                    \n\nQuery: {query}\n",
                ),
            ]
        )

        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        def agent(state: State) -> dict:
            current_messages = state["messages"]
            history = [
                msg for msg in current_messages if not isinstance(msg, SystemMessage)
            ]
            messages_for_llm = [SystemMessage(content=self.sys_prompt)] + history
            response = llm_with_tools.invoke(messages_for_llm)
            return {"messages": [response]}

        def generate(state: State) -> dict:
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                raise ValueError(
                    "Last message is not a ToolMessage. Generation node expects tool output."
                )

            raw = last_message.content
            retrieved_texts: List[str] = cast(List[str], raw)
            if not isinstance(retrieved_texts, list) or not all(
                isinstance(t, str) for t in retrieved_texts
            ):
                retrieved_texts = [str(retrieved_texts)]

            query = ""
            for msg in reversed(messages[:-1]):
                if isinstance(msg, HumanMessage):
                    query = str(msg.content)
                    break
            if not query:
                # should not be here
                query = "Answer that I don't know the answer."

            generation = combine_chain.invoke(
                {
                    "messages": messages,
                    "context": retrieved_texts,
                    "query": query,
                }
            )

            # print("\n--------Retrieved context----------\n")
            # for i, doc in enumerate(retrieved_docs, start=1):
            #     print(f"--- Document {i} ---")
            #     print("Source:", doc.metadata.get("source", "N/A"))
            #     print("Title:", doc.metadata.get("title", "N/A"))
            #     print("Content preview:")
            #     print(doc.page_content)
            #     print()
            self.memory.add_ai_message(generation)
            return {"messages": [generation]}

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.add_node("generate", generate)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )

        graph.add_edge("tools", "generate")
        graph.add_edge("generate", END)

        return graph.compile()  # type: ignore[return-value]
