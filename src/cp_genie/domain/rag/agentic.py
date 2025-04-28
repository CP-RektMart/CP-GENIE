from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from cp_genie.domain.rag.base import State
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List
from langchain_core.documents import Document


class AgenticRAG:
    def __init__(self, llm, retriever, memory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.chain = self._build_graph()

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
            return docs

        tools = [retrieve]
        tool_node = ToolNode(tools)
        llm_with_tools = self.llm.bind_tools(tools)

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

        def query_or_respond(state: State) -> dict:
            print("--- Agent Node: Query or Respond ---")
            messages = state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def generate(state: State) -> dict:
            print("--- Generate Node ---")
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                raise ValueError(
                    "Last message is not a ToolMessage. Generation node expects tool output."
                )

            retrieved_docs = last_message.content
            if not isinstance(retrieved_docs, list) or not all(
                isinstance(doc, Document) for doc in retrieved_docs
            ):
                if isinstance(retrieved_docs, str):
                    retrieved_docs = [Document(page_content=retrieved_docs)]
                else:
                    retrieved_docs = [Document(page_content=str(retrieved_docs))]

            generation = combine_chain.invoke(
                {
                    "messages": messages,
                    "context": retrieved_docs,
                }
            )

            self.memory.add_ai_message(generation)
            return {"messages": [generation]}

        graph = StateGraph(State)
        graph.add_node("agent", query_or_respond)
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

        return graph.compile()

    def invoke(self, input) -> State:
        self.memory.add_user_message(HumanMessage(content=input))
        initial_state: State = {
            "messages": self.memory.get_messages(),
            "context": [],
        }
        return self.chain.invoke(initial_state)
