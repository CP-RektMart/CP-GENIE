import streamlit as st
import requests

st.set_page_config(
    page_title="CP Genie",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the title of the app
st.markdown(
    """
    <div style="text-align: center;">
        <h1>CP <span style="color:pink;">Genie</span></h1>
        <p>CP Genie is an AI-powered assistant inspired by ChulaGenie based on Gemini-2.5-flash</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []


session_id = st.number_input("Enter session ID:", min_value=1, step=1, value=1)
# Define a global variable for session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = 1


def ask_rag(prompt):
    model = "normal"
    base_url = f"http://localhost:8000/api/v1/{model}/{session_id}"

    # Define the payload
    payload = {"query": prompt}
    # no headers
    headers = {"Content-Type": "application/json"}
    # Send the POST request
    with st.spinner("กำลังคิด วิเคราะห์ แยกแยะ..."):
        response = requests.post(base_url, json=payload, headers=headers)
    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        answer = data.get("answer", "Sorry, I couldn't find an answer.")
        # Append the assistant's response to the session state messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.insert(0, {"role": "assistant", "content": answer})
    else:
        st.error(f"Error: {response.status_code} - {response.text}")


def run():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("อยากถามกีกี้ว่า...")
    if prompt:
        # Append the human's message to the session state messages
        st.session_state.messages.insert(0, {"role": "human", "content": prompt})
        ask_rag(prompt)

    # Display the chat history
    for message in st.session_state.messages:
        if message["role"] == "human":
            st.chat_message("human").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])


if __name__ == "__main__":
    run()
