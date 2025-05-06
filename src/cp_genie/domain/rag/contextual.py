from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from cp_genie.domain.rag.base import State, BaseRAG


class ContextualRAG(BaseRAG):
    def _build_graph(self) -> StateGraph:
        # Analyze query prompt
        query_analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query analyzer that helps improve information retrieval. 
                Your task is to:
                1. Analyze the user's query in the context of the conversation history
                2. Extract key information needs
                3. Identify implicit questions based on conversation context
                4. Generate an improved search query that captures the full context
                
                Output format should be a JSON with:
                - original_query: The original query from the user
                - enhanced_query: An improved query that includes context from conversation
                - context_points: List of key contextual points from the conversation history
                """,
                ),
                ("human", "Chat History:\n{chat_history}\n\nCurrent Query: {query}"),
            ]
        )

        # Evaluates retrieved documents for relevance
        context_analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a context analyzer that evaluates retrieved documents.
                Your task is to:
                1. Analyze each retrieved document for relevance to the current query
                2. Score documents on a scale of 0-10 for relevance, where 10 is highly relevant
                3. Suggest if additional retrieval is needed
                4. Identify specific missing information that should be searched for
                
                Output format should be a JSON with:
                - document_scores: List of relevance scores for each document
                - needs_additional_retrieval: Boolean indicating if more information is needed
                - missing_context: Description of information that should be searched for
                """,
                ),
                (
                    "human",
                    "Query: {query}\n\nEnhanced Query: {enhanced_query}\n\nRetrieved Documents:\n{documents}",
                ),
            ]
        )

        # Final Generation Prompt
        generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.sys_prompt
                    + "\n\nAdditional Instructions: When responding to the user, make sure to specifically address their query while incorporating relevant information from the provided context. Maintain the conversation flow naturally.",
                ),
                (
                    "human",
                    """Retrieved context:
                    
{context}

---

Chat History:
{messages}

---

Query: {query}

Enhanced understanding of query: {enhanced_query}

Please answer the query based on the context and history, addressing both explicit and implicit questions.""",
                ),
            ]
        )

        combine_chain = create_stuff_documents_chain(self.llm, generation_prompt)

        def analyze_query(state: State) -> dict:
            """Analyze the query in context of conversation history to create an enhanced query"""
            messages = state["messages"]
            query = state["query"]

            # Extract chat history for context
            chat_history = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    chat_history += f"User: {msg.content}\n"
                elif not isinstance(msg, SystemMessage):
                    chat_history += f"Assistant: {msg.content}\n"

            # If this is the first query or very little history, skip enhancement
            if chat_history.count("User:") <= 1:
                return {
                    "enhanced_query": query,
                    "original_query": query,
                    "context_points": [],
                }

            # Analyze query in context of conversation
            analysis_result = self.llm.invoke(
                query_analysis_prompt.format(chat_history=chat_history, query=query)
            )

            try:
                # Parse the JSON response - in a real implementation, use proper parsing
                # This is simplified for clarity
                result_text = analysis_result.content

                if "enhanced_query" in result_text:
                    # Extract enhanced query using simple string parsing
                    # In production, use proper JSON parsing
                    import json
                    import re

                    # Try to extract JSON from the text
                    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                    if json_match:
                        try:
                            result_json = json.loads(json_match.group(0))
                            enhanced_query = result_json.get("enhanced_query", query)
                            context_points = result_json.get("context_points", [])
                            return {
                                "enhanced_query": enhanced_query,
                                "original_query": query,
                                "context_points": context_points,
                            }
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                print(f"Error parsing query analysis result: {e}")

            # Fallback to original query if parsing fails
            return {
                "enhanced_query": query,
                "original_query": query,
                "context_points": [],
            }

        def retrieve(state: State) -> dict:
            """Retrieve documents based on enhanced query"""
            enhanced_query = state.get("enhanced_query", state["query"])

            # Primary retrieval
            docs = self.retriever.invoke(enhanced_query)

            # If we have context points, do additional targeted retrievals
            context_points = state.get("context_points", [])
            additional_docs = []

            if context_points and len(context_points) > 0:
                # Limit to top 3 context points to avoid too many retrievals
                for point in context_points[:3]:
                    if isinstance(point, str):
                        supplementary_docs = self.retriever.invoke(point)
                        additional_docs.extend(supplementary_docs)

            # Combine and deduplicate documents
            all_docs = docs + additional_docs
            # Simple deduplication by content
            unique_docs = []
            seen_contents = set()
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)

            return {"context": unique_docs[:10]}  # Limit to top 10 docs

        def analyze_context(state: State) -> dict:
            """Analyze retrieved documents for relevance and determine if additional retrieval is needed"""
            query = state["query"]
            enhanced_query = state.get("enhanced_query", query)
            docs = state.get("context", [])

            if not docs:
                return {"needs_additional_retrieval": False}

            # Format documents for analysis
            docs_text = "\n\n".join(
                [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

            analysis_result = self.llm.invoke(
                context_analysis_prompt.format(
                    query=query, enhanced_query=enhanced_query, documents=docs_text
                )
            )

            try:
                # Parse the JSON response - in a real implementation, use proper parsing
                result_text = analysis_result.content
                import json
                import re

                # Try to extract JSON from the text
                json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                if json_match:
                    try:
                        result_json = json.loads(json_match.group(0))
                        needs_additional = result_json.get(
                            "needs_additional_retrieval", False
                        )
                        missing_context = result_json.get("missing_context", "")
                        doc_scores = result_json.get("document_scores", [])

                        # Filter documents by relevance score if scores are provided
                        if doc_scores and len(doc_scores) == len(docs):
                            relevant_docs = []
                            for i, score in enumerate(doc_scores):
                                if (
                                    score >= 3
                                ):  # Only keep documents with score 3 or higher
                                    relevant_docs.append(docs[i])

                            if relevant_docs:
                                state["context"] = relevant_docs

                        return {
                            "needs_additional_retrieval": needs_additional,
                            "missing_context": missing_context,
                        }
                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                print(f"Error parsing context analysis result: {e}")

            return {"needs_additional_retrieval": False}

        def additional_retrieve(state: State) -> dict:
            """Perform additional retrieval for missing information"""
            missing_context = state.get("missing_context", "")

            if not missing_context:
                return {}

            # Retrieve additional documents
            additional_docs = self.retriever.invoke(missing_context)

            # Add new docs to existing context
            current_docs = state.get("context", [])

            # Simple deduplication
            seen_contents = {doc.page_content for doc in current_docs}
            unique_new_docs = []
            for doc in additional_docs:
                if doc.page_content not in seen_contents:
                    unique_new_docs.append(doc)
                    seen_contents.add(doc.page_content)

            combined_docs = current_docs + unique_new_docs

            # Limit to top docs to prevent context overflow
            return {"context": combined_docs[:12]}

        def generate(state: State) -> dict:
            """Generate response using the retrieved context"""
            query = state["query"]
            enhanced_query = state.get("enhanced_query", query)
            context = state.get("context", [])

            if not context:
                # If no relevant documents were found, fall back to the model's knowledge
                result = self.llm.invoke(
                    f"Chat History:\n{state['messages']}\n\nQuery: {query}\n\nPlease answer based on your knowledge."
                )
                self.memory.add_ai_message(result)
                return {"messages": [result]}

            result = combine_chain.invoke(
                {
                    "messages": state["messages"],
                    "context": context,
                    "query": query,
                    "enhanced_query": enhanced_query,
                }
            )

            self.memory.add_ai_message(result)
            return {"messages": [result]}

        def should_retrieve_additional(state: State) -> str:
            """Decide whether to do additional retrieval"""
            needs_additional = state.get("needs_additional_retrieval", False)
            return "additional_retrieve" if needs_additional else "generate"

        # Build the graph
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("analyze_query", RunnableLambda(analyze_query))
        graph.add_node("retrieve", RunnableLambda(retrieve))
        graph.add_node("analyze_context", RunnableLambda(analyze_context))
        graph.add_node("additional_retrieve", RunnableLambda(additional_retrieve))
        graph.add_node("generate", RunnableLambda(generate))

        # Set entry point
        graph.set_entry_point("analyze_query")

        # Add edges
        graph.add_edge("analyze_query", "retrieve")
        graph.add_edge("retrieve", "analyze_context")

        # Conditional edge based on context analysis
        graph.add_conditional_edges(
            "analyze_context",
            should_retrieve_additional,
            {"additional_retrieve": "additional_retrieve", "generate": "generate"},
        )

        graph.add_edge("additional_retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()  # type: ignore[return-value]
