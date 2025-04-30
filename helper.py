from dataclasses import dataclass
import os
import json
import requests
import re
import duckduckgo_search as ddg
import requests
from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright

SERVICE = "DOCKER" # or "OLLAMA"

# Get the home directory of the current user
home_directory = os.path.expanduser("~")

@dataclass
class WebSearchResult:
    title: str
    href: str
    body: str

def web_search(search: str) -> list[WebSearchResult]:
    return [ WebSearchResult(**x) for x in ddg.DDGS().text(search, max_results=5)]

class ChatHistory:
    def __init__(self) -> None:
        self.clear()
    def append(self, data: dict[str, str]):
        self.history.append(data)
    def pop(self):
        if len(self.history) >= 2:
            self.history.pop()
            self.history.pop()
    def clear(self):
        self.history: list[dict[str, str]] = []

def ollama_ask(system: str, question: str, history: ChatHistory, temperature: float = 0.01, record: bool = True):
    user_question = {
        "role": "user",
        "content": question,
    }

    if record:
        history.append(user_question)

    messages = [{
        "role": "system",
        "content": system
    }]

    messages.extend(history.history)
    messages.append(user_question)

    payload = {
        "model": "qwen2.5:7b",
        "messages": messages,
        "options" : {"temperature":temperature},
        "stream": True,
    }

    response = requests.post("http://localhost:11434/api/chat", data=json.dumps(payload), stream=True)

    result = []
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                decoded_chunk = json.loads(chunk.decode('utf-8'))
                decoded_chunk = decoded_chunk.get("message",{}).get("content","")
                yield decoded_chunk
                if record:
                    result.append(decoded_chunk)
    except:
        pass

    if record:
        history.append({
            "role": "assistant",
            "content": "".join(result)
        })

def docker_ask(system: str, question: str, history: ChatHistory, temperature: float = 0.01, record: bool = True):
    user_question = {
        "role": "user",
        "content": question,
    }

    if record:
        history.append(user_question)

    messages = [{
        "role": "system",
        "content": system
    }]

    messages.extend(history.history)
    messages.append(user_question)

    payload = {
        "model": "ai/qwen2.5:latest",
        "messages": messages,
        "options" : {"temperature":temperature},
        "stream": True,
    }

    response = requests.post("http://localhost:12434/engines/llama.cpp/v1/chat/completions", data=json.dumps(payload), stream=True)

    result = []
    try:
        for chunk in response.iter_content(chunk_size=None):
            decoded_chunk = chunk.decode('utf-8')
            if "[DONE]\n\n" in decoded_chunk:
                break
            decoded_chunk = decoded_chunk.replace('data: ', '')
            decoded_chunk = json.loads(decoded_chunk)

            if 'choices' in decoded_chunk:
                decoded_chunk = decoded_chunk['choices'][0]['delta'].get('content', '')
            yield decoded_chunk
            if record:
                result.append(decoded_chunk)
    except Exception as e:
        print(e)

    if record:
        history.append({
            "role": "assistant",
            "content": "".join(result)
        })

ask = docker_ask if SERVICE == "DOCKER" else ollama_ask

def load_file(partial_name: str) -> tuple[str, str]:
    """
    Load a file based on a partial name.
    Args:
        partial_name (str): The partial name of the file to load.
    Returns:
        tuple: A tuple containing the matched file name and its contents.
    """
    # Check if a file with the exact name exists
    if os.path.isfile(partial_name):
        with open(partial_name, 'r') as file:
            return partial_name, file.read()

    # If not, search for files containing the partial name
    for filename in os.listdir("."):
        if partial_name.lower() in filename.lower():
            with open(filename, 'r') as file:
                return filename, file.read()

    raise "No match"

async def async_curl(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url=url, timeout=5000, wait_until="domcontentloaded")
        webpage = await page.text_content("body")
        await browser.close()
    soup = BeautifulSoup(webpage, 'html.parser')
    webpage = soup.get_text()
    return webpage

def curl(url: str) -> str:
    asyncio.run(async_curl(url))

system_prompt = "You are an artificial assistant named 'Helper'. Do what the user asks of you."

handlers = []

class HandlerClear:
    def desc(self):
        return "/clear or /reset - clear chat history"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question in ['/clear', '/reset'] else 0.0
    def do(self, question:str, history:ChatHistory):
        history.clear()
        yield "history reset"

class HandlerSystem:
    def desc(self):
        return "/system - change the system prompt"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question.startswith('/system') else 0.0
    def do(self, question:str, history:ChatHistory):
        global system_prompt
        new_prompt = question.replace("/system","").strip()
        if len(new_prompt) == 0:
            yield f"The current system prompt is: {system_prompt}"
        else:
            yield f"Changed system prompt to: {new_prompt}"

class HandlerFile:
    def desc(self):
        return "/file - load a file.  partial file names will match"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question[:5] == '/file' else 0.0
    def do(self, question:str, history:ChatHistory):
        if len(question.split(" ")) == 1:
            yield "Please specify a file name"
            return
        file_name, file_content = load_file(question.split(" ")[1])
        if file_name is not None:
            history.append({"role": "user", "content": f"Here is the content of the file called {file_name}\n{file_content}"})
            yield f"Loaded {file_name} into conversation"
        else:
            yield "Cannot find a file that matches"

class HandlerQuit:
    def desc(self):
        return "/bye or /quit - to quit"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question in ['/bye','/quit','bye','quit'] else 0.0
    def do(self, question:str, history:ChatHistory):
        yield "Bye"

class HandlerHelp:
    def desc(self):
        return "/help - this command"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question in ['help', '/help', '/h', '?', '/?'] else 0.0
    def do(self, question:str, history:ChatHistory):
        global handlers
        desciptions =list([ handler.desc() for handler in handlers if handler.desc() is not None ])
        desciptions = sorted(desciptions)
        yield "\n".join(desciptions)

class HandlerSummarize:
    def desc(self):
        return "/summarize - summarize and replace the entire chat history"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question == '/summarize' else 0.0
    def do(self, question:str, history:ChatHistory):
        summarize_prompt = "Summarize the conversation, but copy important facts from the conversation so that they are retrained."

        message  = "The following is a summary of our previous conversation:\n"
        yield message
        summary = [message]

        for item in ask(system=summarize_prompt, question="Summarize the conversation", history=history):
            yield item
            summary.append(item)

        history.clear()
        history.append({"role": "assistant", "content": "".join(summary)})

class HandlerSave:
    def desc(self):
        return "/save - save the current conversation"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question == '/save' else 0.0 
    def do(self, question:str, history:ChatHistory):
        with open(home_directory +"/code/helper/conversation.json", 'w') as file:
            json.dump(history.history, file, indent=4)
        yield f"Conversation saved to conversation.json"

class HandlerLoad:
    def desc(self):
        return "/load - load the current conversation"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question == '/load' else 0.0 
    def do(self, question:str, history:ChatHistory):
        with open(home_directory + "/code/helper/conversation.json", 'r') as file:
            history.history = json.load(file)
        yield f"Conversation loaded from conversation.json"

class HandlerHistory:
    def desc(self):
        return "/history - show the current conversation.  Use '/history full' to see the entire conversation."
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if question[:8] == '/history' else 0.0 
    def do(self, question:str, history:ChatHistory):
        if 'full' in question:
            for turn in history.history:
                yield f"-------------------\nROLE: {turn['role']}\n{turn['content']}\n\n"
        else:
            yield f"The conversation contains {len(history.history)} turns\n"
            if len(history.history) > 16:
                yield "The last 16 turns are:\n"
            for turn in history.history[-16:]:
                content = re.sub("\n+"," / ",turn['content'])
                if len(content) > 160:
                    content = content[:150] + "... [truncated]"
                yield " > " + turn['role'] + " : " + content + "\n"

class HandlerDefault:
    def desc(self):
        return "[text] - send text to the AI"
    def match(self, question:str, history:ChatHistory) -> float:
        return 0.4
    def do(self, question:str, history:ChatHistory):
        for out in ask(system=system_prompt, question=question, history=history):
            yield out

class HandlerUnknown:
    def desc(self):
        return None
    def match(self, question:str, history:ChatHistory) -> float:
        return 0.9 if question[0] == "/" else 0.0
    def do(self, question:str, history:ChatHistory):
        yield "Unknown command"

class HandlerWebSearch:
    def desc(self):
        return "/search [text] - search the web for information"
    def match(self, question:str, history:ChatHistory) -> float:
        return 1.0 if "/search" in question[:10] else 0.0
    def do(self, question:str, history:ChatHistory):
        history.append({"role": "user", "content": question})

        search = "".join(list(ask(system="Respond to the users question by creating a web search query that will answer the users question.  Respond only with a web search phrase to be sent to Google or DuckDuckGo.  Do no answer the users question",
                       question=question, history=history, record=False)))

        search_message = f"Searching the web using '{search}'...\n\n"

        yield search_message

        search_results:list[WebSearchResult] = web_search(search)
        search_result = "\n".join([f"{result.title} {result.href}\n{result.body}\n" for result in search_results])

        history.append({"role": "assistant", "content": search_message + search_result, "urls": list(result.href for result in search_results)})
        yield search_result

        if (len(search_results) == 0 or "EOF" in search_results[0].title):
            yield "[No results found]"
            history.append({"role": "assistant", "content": "[No results found]"})
            return

        yield "\nAnswering the question...\n\n"

        response = []
        try:
            prompt = f"The user has copied search results from the web.  Respond to the user by summarizing it the information.  Do not invent new information or answer questions."

            for x in ask(system=prompt, question=question, history=ChatHistory(), record=False):
                response.append(x)
                yield x
        except:
            pass

        history.append({"role": "assistant", "content": "".join(response)})

class HandlerCurl():
    def desc(self) -> str:
        return "/curl [url] - Perform a web request to the specified URL and return the response into the chat."
    def match(self, query: str, history: ChatHistory) -> float:
        return 1.0 if "/curl" in query[:5] else 0
    def do(self, query: str, history: ChatHistory):
        history.append({"role": "user", "content": query})
        url = query.replace("/curl ", "").strip()
        yield f"Fetching the URL ({url}) ...\n"
        webpage = curl(url)
        yield "Extracting meaningful information...\n\n"
        response = []
        try:
            for item in ask(system=f"Rewrite the human readable parts of following web page into markdown format.  The last message from the user is a web page copied from {url}. Format the response as markdown.  Do not use pleasantries, just give the user the information",
                            question=webpage, history=history, record=False):
                yield item
                response.append(item)
        except Exception as e:
            pass

        history.append({"role": "assistant", "content": "".join(response)})


handlers = [HandlerHelp(), HandlerSystem(), HandlerClear(), HandlerFile(), HandlerQuit(), HandlerLoad(), HandlerSave(), HandlerHistory(), HandlerSummarize(), HandlerDefault(), HandlerWebSearch(), HandlerUnknown(), HandlerCurl()]

history = ChatHistory()
def main(history) -> None:
    handler = HandlerUnknown()
    while type(handler) != HandlerQuit:
        question = input("Question: ")
        if len(question) == 0:
            continue
        print()

        handler_scores = [(h.match(question, history), h) for h in handlers]
        handler_scores.sort(reverse=True, key=lambda x: x[0])

        handler = HandlerUnknown()
        score = -1.0

        for h in handlers:
            s = h.match(question, history)
            if s > score:
                score, handler = s, h
            if score == 1.0:
                break
        
        for out in handler.do(question, history):
            print(end=out)

        print("\n")


if __name__ == "__main__":
    print()
    try:
        for out in HandlerLoad().do(question="/load", history=history):
            print(end=out)
        print()
        for out in HandlerHistory().do(question="/history", history=history):
            print(end=out)
        print()
        main(history)
    except KeyboardInterrupt as k:
        pass
    except EOFError as e:
        pass

    for out in HandlerSave().do(question="/save", history=history):
        print(end=out)
    print()