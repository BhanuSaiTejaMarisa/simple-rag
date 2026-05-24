import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Simple RAG")
st.caption("Ask questions about Transformers, LLMs, and Cognitive Psychology")

# session state holds chat history between reruns
if "messages" not in st.session_state:
    st.session_state.messages = []      # display messages (role + content)
if "api_history" not in st.session_state:
    st.session_state.api_history = []   # serialized history sent to API

# display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# chat input
query = st.chat_input("Ask something...")

if query:
    # show user message immediately
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/ask", json={
                "query": query,
                "history": st.session_state.api_history
            })
            data = response.json()

        st.write(data["answer"])

    # update state
    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
    })
    st.session_state.api_history = data["history"]
