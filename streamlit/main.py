import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
# from src.cp_genie.domain.rag.contextual import setup_rag  

# Page configuration
st.set_page_config(page_title="RAG Chat System", page_icon="🤖")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("RAG Chat System")

# Set up the RAG chain

# Display chat message history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input for RAG
if prompt := st.chat_input("Ask something about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     try:
    #         # Get response from RAG
    #         response = chain.invoke({"question": prompt})
    #         ai_response = response["answer"]
            
    #         # Display the response
    #         message_placeholder.markdown(ai_response)
            
    #         # Add AI response to chat history
    #         st.session_state.messages.append(AIMessage(content=ai_response))
    #     except Exception as e:
    #         message_placeholder.error(f"Error: {str(e)}")