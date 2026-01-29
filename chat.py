import os
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

from app import run_agent

load_dotenv()
# retrieval behavior
TOP_K = 5
NOT_FOUND_DISTANCE = 0.35  # larger distance => less similar

COMPANY_NAME = os.getenv("COMPANY_NAME", "Hyundai")
COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "ipinfo@hyundai.com")
COMPANY_PHONE = os.getenv("COMPANY_PHONE", "+000 00 000 00 00")

st.set_page_config(page_title="Customer Support RAG + Tickets", layout="centered")

st.title("Customer Support Chat")
st.caption(f"{COMPANY_NAME} • {COMPANY_EMAIL} • {COMPANY_PHONE}")

if "chat" not in st.session_state:
    st.session_state.chat = []

# render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask a question (or say: create a ticket...)")

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    openai_client = OpenAI()

    system_prompt = f"""
You are a helpful customer support assistant for {COMPANY_NAME}.
Company contact info:
- Email: {COMPANY_EMAIL}
- Phone: {COMPANY_PHONE}

Rules:
1) Use the tool search_docs to answer questions from documents. Provide citations as: filename (page N).
2) If best match distance > {NOT_FOUND_DISTANCE} OR you cannot find the answer, say you couldn't find it and suggest creating a support ticket.
3) If the user asks to create a ticket, use create_ticket tool.
4) Keep answers short and clear.
"""

    # build messages with history
    messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.chat:
        messages.append({"role": m["role"], "content": m["content"]})

    # run agent
    answer = run_agent(openai_client, messages)

    # show assistant answer
    st.session_state.chat.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
