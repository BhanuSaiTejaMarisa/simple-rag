import streamlit as st
from agent import ask
from langchain_core.messages import HumanMessage, AIMessage

st.title("Simple RAG")
st.caption("Ask questions about Transformers, LLMs, and Cognitive Psychology")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(query=query, history=st.session_state.history)
        st.write(result["answer"])
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
    st.session_state.history = result["history"]
