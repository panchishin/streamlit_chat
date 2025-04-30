from langchain.schema import ChatMessage
import streamlit as st
import helper

st.set_page_config(page_title="Chatbot", page_icon="🤖")

def ask_helper(question:str, history:list[ChatMessage], container):
    full_response = ""
    for chunk in helper.process_question(question=question, history=history):
        full_response += chunk
        container.markdown(full_response)
    return full_response


with st.sidebar:
    "# Sidebar"
    if st.button("Clear Chat"):
        st.session_state.messages = [ChatMessage(role="assistant", content="Ask me anything")]
    if st.button("Redraw Chat"):
        messages = list(x for x in st.session_state.messages)
        st.session_state.messages = []
        for msg in messages:
            st.session_state.messages.append(msg)
    if st.button("Load"):
        for x in helper.HandlerLoad().do(question="/load", history=st.session_state.messages):
            pass
        messages = list(x for x in st.session_state.messages)
        st.session_state.messages = []
        for msg in messages:
            st.session_state.messages.append(msg)

    if st.button("Save"):
        for x in helper.HandlerSave().do(question="/save", history=st.session_state.messages):
            pass

if "messages" not in st.session_state:
    st.session_state.messages = [ChatMessage(role="assistant", content="Ask me anything")]

for msg in st.session_state.messages:
    if msg.role != "system":
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input(placeholder="Ask anything"):

    if 'stop' in prompt:
        st.info("You said stop!")
        st.stop()

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        response = ask_helper(question=prompt, history=st.session_state.messages, container=container)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))