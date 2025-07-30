import streamlit as st
from core_chatbot import ask_bot

st.title("🧠 Medical Chatbot (Gemini)")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask a medical question:")

if user_input:
    if any(k in user_input.lower() for k in ["symptom", "treatment", "pain", "medicine", "doctor", "disease"]):
        answer = ask_bot(user_input)
    else:
        answer = "⚠️ I only respond to medical-related questions."

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

for speaker, msg in reversed(st.session_state.chat):
    st.markdown(f"**{speaker}:** {msg}")
