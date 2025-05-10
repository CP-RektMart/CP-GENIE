from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, cast
from langchain_core.documents import Document
from cp_genie.domain.rag.base import State, ContextualRAG


class ContextualAgenticRAG(ContextualRAG):
    def _build_graph(self) -> StateGraph:

        @tool
        def retrieve_fac(query: str) -> List[Document]:
            """
            ใช้สำหรับค้นหาข้อมูลพื้นฐานเกี่ยวกับอาจารย์ในภาควิชาวิศวกรรมคอมพิวเตอร์ หรือ จุฬาลงกรณ์มหาวิทยาลัย
            เช่น ชื่อ-นามสกุล ตำแหน่งทางวิชาการ วุฒิการศึกษา ภาควิชา หรือประวัติการสอน
            """
            retrieved_docs = self.retriever_fac.invoke(query)
            serialized = "\n\n".join(
                (f"Context: {doc.page_content}") for doc in retrieved_docs
            )
            return serialized

        @tool
        def retrieve_oth(query: str) -> List[Document]:
            """
            ใช้สำหรับค้นหาข้อมูลอื่น ๆ ที่เกี่ยวข้องกับภาควิชาวิศวกรรมคอมพิวเตอร์ จุฬาลงกรณ์มหาวิทยาลัย
            เช่น งานวิจัยของอาจารย์ บทความวิชาการ ข่าวกิจกรรม งานบริการวิชาการ หรือรางวัลต่าง ๆ
            """

            retrieved_docs = self.retriever_oth.invoke(query)
            serialized = "\n\n".join(
                (f"Context: {doc.page_content}") for doc in retrieved_docs
            )
            return serialized

        tools = [retrieve_fac, retrieve_oth]
        tool_node = ToolNode(tools)
        llm_with_tools = self.llm.bind_tools(tools)

        agent_system_prompt = (
            self.sys_prompt
            + """
## Tool Usage Guidelines
When interacting with users, you have access to specialized tools to help provide accurate information:

1. The retrieve tool lets you search for specific information about Chulalongkorn University and its Computer Engineering department.
2. Usetone in all interactions.
"""
        )

        generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.sys_prompt
                    + """\n\nAdditional Instructions: 
When responding to the user after retrieving information, you should:
1. Focus on answering the specific question using the retrieved information
2. Structure your response in a clear, organized manner
3. Acknowledge the source of the information (from university resources)
4. If the retrieved information is incomplete or doesn't fully address the query, acknowledge this
5. Stay true to GIGI's warm, knowledgeable personality
""",
                ),
                (
                    "human",
                    """# Retrieved Information

{context}

# Conversation History
{messages}

# User's Query
{query}

Please respond to the user's query based on the retrieved information and conversation history. Maintain GIGI's persona and communication style throughout your response.""",
                ),
            ]
        )

        combine_chain = create_stuff_documents_chain(self.llm, generation_prompt)

        def agent(state: State) -> dict:
            current_messages = state["messages"]
            history = [
                msg for msg in current_messages if not isinstance(msg, SystemMessage)
            ]
            messages_for_llm = [SystemMessage(content=agent_system_prompt)] + history
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
            docs = [Document(page_content=raw)]
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
                    "context": docs,
                    "query": query,
                }
            )

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
