from langchain.schema import ChatMessage
import streamlit as st
import requests
import json

DEFAULT_SYSTEM = """Reply to the user without pleasantries.
Keep verbosity to a medium-low.
If the users last message isn't a question then reply with 'acknowledged'.
At the end of your answer end without bidding for engagment with phrases like 'How can I help you today?'."""

st.set_page_config(page_title="Chatbot", page_icon="🤖")

def docker_ask(system:str, history, container, temperature:float=0.01):

    messages = [{
        "role": "system",
        "content": system
    }]
    for item in history:
        messages.append({
            "role": item.role,
            "content": item.content
        })

    payload = {
        "model": "ai/qwen2.5:latest",
        "messages": messages,
        "options" : {"temperature":temperature},
        "stream": True,
    }

    response = requests.post("http://localhost:12434/engines/llama.cpp/v1/chat/completions", data=json.dumps(payload), stream=True)

    result = ""
    try:
        for chunk in response.iter_content(chunk_size=None):
            decoded_chunk = chunk.decode('utf-8')
            if "[DONE]\n\n" in decoded_chunk:
                break
            decoded_chunk = decoded_chunk.replace('data: ', '')
            decoded_chunk = json.loads(decoded_chunk)

            if 'choices' in decoded_chunk:
                decoded_chunk = decoded_chunk['choices'][0]['delta'].get('content', '')
            result += decoded_chunk
            container.markdown(result)

    except Exception as e:
        print(e)

    return result


with st.sidebar:
    "# Sidebar"
    if st.button("Clear Chat"):
        del st.session_state.messages

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="Ask me anything")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input(placeholder="Ask anything"):

    if 'stop' in prompt:
        st.info("You said stop!")
        st.stop()

    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        response = docker_ask(system=DEFAULT_SYSTEM, history=st.session_state.messages, container=container)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))