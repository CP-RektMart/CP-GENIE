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

            Use this tool when:
            1. The user asks about specific university details, programs, or department information
            2. The question requires factual knowledge about Chulalongkorn University
            3. You need to verify information before responding
            4. The user is asking about policies, procedures, or resources related to the university

            Don't use this tool for:
            1. General knowledge questions unrelated to the university
            2. Simple greetings or conversation
            3. Questions about topics outside of the university context
            4. Personal opinions or subjective assessments
            """
            return self.retriever.invoke(query)

        tools = [retrieve]
        tool_node = ToolNode(tools)
        llm_with_tools = self.llm.bind_tools(tools)

        agent_system_prompt = (
            self.sys_prompt
            + """
## Tool Usage Guidelines
When interacting with users, you have access to specialized tools to help provide accurate information:

1. The retrieve tool lets you search for specific information about Chulalongkorn University and its Computer Engineering department.
2. Use this tool strategically - only when the user's question requires specific university knowledge.
3. When using the retrieve tool, formulate a clear, specific query focusing on the key information need.
4. After retrieving information, integrate it smoothly into your response, maintaining your helpful personality.
5. If the retrieved information doesn't fully answer the question, acknowledge this and provide the best response you can.

Remember to maintain GIGI's warm, knowledgeable tone in all interactions.
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
            retrieved_docs = [Document(page_content=str(raw))]

            query = ""
            for msg in reversed(messages[:-1]):
                if isinstance(msg, HumanMessage):
                    query = str(msg.content)
                    break
            if not query:
                # should not be here
                query = "Answer that I don't know the answer."

            print("\n--------Retrieved context----------\n")
            print(retrieved_docs)
            print("\n------------------\n")
            generation = combine_chain.invoke(
                {
                    "messages": messages,
                    "context": retrieved_docs,
                    "query": query,
                }
            )

            print("\n--------Retrieved context----------\n")
            for i, doc in enumerate(retrieved_docs, start=1):
                print(f"--- Document {i} ---")
                print("Source:", doc.metadata.get("source", "N/A"))
                print("Title:", doc.metadata.get("title", "N/A"))
                print("Content preview:")
                print(doc.page_content)
                print()
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
