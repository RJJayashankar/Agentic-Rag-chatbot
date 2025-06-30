import streamlit as st
from agent import graph  # Compiled LangGraph agent

st.set_page_config(page_title="Cybersecurity RAG Agent", page_icon=":shield:", layout="centered")

st.title("🛡️ Cybersecurity RAG Agent")
st.write("Ask any cybersecurity-related question based on the loaded PDF knowledge base.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Use a form to control input + clearing
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question:", key="user_text")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    try:
        result = graph.invoke({"query": user_input})
        response = result.get("response", "No response generated.")
    except Exception as e:
        response = f"Error: {e}"

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Agent", response))

# ✅ Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**🛡️ Agent:** {message}")
