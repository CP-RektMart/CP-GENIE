from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from cp_genie.domain.rag.base import State
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent


class AgenticRAG:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools  # list of Tool objects
        self.memory = memory
        self.chain = self._build_graph()

    def _build_graph(self) -> StateGraph:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the available tools to retrieve useful context and answer as concisely as possible.",
                ),
                (
                    "human",
                    "chat history: {messages}\nretrieved context: {context}\nquestion: {question}",
                ),
            ]
        )
        combine_chain = create_stuff_documents_chain(self.llm, prompt)

        # Set up an agent to choose tools
        agent = create_tool_calling_agent(self.llm, self.tools)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        def retrieve(state) -> State:
            tool_inputs = {"input": state["question"]}
            tool_result = agent_executor.invoke(tool_inputs)
            return {**state, "context": tool_result.get("output", [])}

        def generate(state) -> State:
            result = combine_chain.invoke(
                {
                    "messages": state["messages"],
                    "context": state["context"],
                    "question": state["question"],
                }
            )
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
            "output": "",
        }
        return self.chain.invoke(state)
