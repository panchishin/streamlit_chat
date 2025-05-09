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
from langchain.schema import ChatMessage

SERVICE = "OLLAMA" # "DOCKER" # or "OLLAMA"


# Set up all the necessary directories
home_directory = os.path.expanduser("~")
helper_dir = os.path.join(home_directory, ".helper")
os.makedirs(helper_dir, exist_ok=True)
web_cache_dir = os.path.join(home_directory, "webcache")
os.makedirs(web_cache_dir, exist_ok=True)
conversation_log = os.path.join(helper_dir, "conversation.json")
if not os.path.exists(conversation_log):
    with open(conversation_log, "w") as f:
        f.write("[]")


@dataclass
class WebSearchResult:
    title: str
    href: str
    body: str

def web_search(search: str) -> list[WebSearchResult]:
    return [ WebSearchResult(**x) for x in ddg.DDGS().text(search, max_results=5)]

system_prompt = """You are an artificial assistant named 'Helper'. Do what the user asks of you.
Reply to the user without pleasantries.
Keep verbosity to a medium-low.

Do not ask the user if there is something else you can help with.  Your answers need to be robotic.
If the users last message isn't a question or a request to summarize then reply with 'Acknowledged'.
"""

def prepend_system_message(history:list[ChatMessage]):
    if len(history) == 0 or history[0].role != "system":
        history.insert(0,ChatMessage(role="system", content=system_prompt))

temperature = 0.4
top_k = 20
model = "qwen2.5:7b"
model = "qwen2.5-coder-3b-instruct"
max_tokens = -1
url = 'http://localhost:1234/v1/chat/completions'
# Define headers and data for the request
headers = {
    'Content-Type': 'application/json'
}

def lmstudio_ask(history: list[ChatMessage]):

    prepend_system_message(history)

    data = {
        "model": "qwen2.5-coder-3b-instruct",
        "messages": list({"role": item.role, "content": item.content } for item in history),
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": True
    }

    response = requests.post("http://localhost:1234/v1/chat/completions", headers=headers, json=data, stream=True)
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                
                chunk = chunk.decode('utf-8')[6:]
                try:
                    chunk = json.loads(chunk)
                    chunk = chunk["choices"][0]
                    # print(json.dumps(chunk,indent=4))
                    yield chunk.get("delta",{}).get("content","")
                    if chunk.get("finish_reason",None) is not None:
                        break 
                except Exception as e:
                    yield f" [{e}] "

    except Exception as e:
       print(e)



ask = lmstudio_ask


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
    return soup.get_text()
    
    # # Remove <script> tags
    # for script in soup(["script", "style"]):
    #     script.decompose()  # or use script.extract()
    
    # # Return the cleaned content as a string
    # return str(soup)


def curl(url: str) -> str:
    url_key = re.sub(r"[^a-zA-Z0-9]+", "_", url) + ".url"

    # check in the .webcache directory for a file with the url_key
    cache_file = os.path.join(web_cache_dir, url_key)

    # check if the cache file exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return f.read()
    
    result = asyncio.run(async_curl(url))
    with open(cache_file, "w") as f:
        f.write(result)
    return result

handlers = []

class HandlerClear:
    def desc(self):
        return "/clear or /reset - clear chat history"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question in ['/clear', '/reset'] else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        history.clear()
        yield "history reset"

class HandlerSet:
    def desc(self):
        return "/set [top_k, temp] - query or set the values"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question.startswith('/set') else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        global system_prompt, top_k, temperature
        parts = question.split()
        if len(parts) == 2 and parts[1] == "top_k":
            yield f"top_k is set to {top_k}"
        elif len(parts) == 2 and parts[1] == "temp":
            yield f"temp is set to {temperature}"
        elif len(parts) == 2:
            yield f"Unknown setting {parts[1]}"
        elif len(parts) == 3 and parts[1] == "top_k":
            yield f"top_k was {top_k}. "
            top_k = int(parts[2])
            yield f"Updated to {top_k}"
        elif len(parts) == 3 and parts[1] == "temp":
            yield f"temp was {temperature}. "
            temperature = float(parts[2])
            yield f"Updated to {temperature}"
        else:
            yield f"top_k is set to {top_k}\n"
            yield f"temp is set to {temperature}"

class HandlerSystem:
    def desc(self):
        return "/system - change the system prompt"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question.startswith('/system') else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        global system_prompt
        new_prompt = question.replace("/system","").strip()
        if len(new_prompt) == 0:
            yield f"The current system prompt is: {system_prompt}"
        else:
            system_prompt = new_prompt
            yield f"Changed system prompt to: {new_prompt}"

class HandlerFile:
    def desc(self):
        return "/file - load a file.  partial file names will match"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question[:5] == '/file' else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        if len(question.split(" ")) == 1:
            yield "Please specify a file name"
            return
        file_name, file_content = load_file(question.split(" ")[1])
        if file_name is not None:
            history.append(ChatMessage(role="system", content=f"Contents of file '{file_name}'\n{file_content}"))
            yield f"Loaded {file_name} into conversation"
        else:
            yield f"Cannot find a file that matches '{file_name}'"

class HandlerQuit:
    def desc(self):
        return "/bye or /quit - to quit"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question in ['/bye','/quit','bye','quit'] else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        yield "Bye"

class HandlerHelp:
    def desc(self):
        return "/help - this command"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question in ['help', '/help', '/h', '?', '/?'] else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        global handlers
        desciptions =list([ handler.desc() for handler in handlers if handler.desc() is not None ])
        desciptions = sorted(desciptions)
        yield "\n".join(desciptions)

class HandlerSummarize:
    def desc(self):
        return "/summarize - summarize the entire chat history"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question == '/summarize' else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        summarize_prompt = """Summarize the conversation.
Copy important facts from the conversation so that they are retained.
This summary will be the only thing in the chat history after this command is executed."""

        summarize_history = [ChatMessage(role="system", content=summarize_prompt)] + [item for item in history if item.role != "system"]

        for item in ask(history=summarize_history):
            yield item

class HandlerSave:
    def desc(self):
        return "/save - save the current conversation"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question == '/save' else 0.0 
    def do(self, question:str, history:list[ChatMessage]):
        with open(conversation_log, 'w') as file:
            history = list( {'role':x.role,'content':x.content} for x in history )
            json.dump(history, file, indent=4)
        yield f"Conversation saved to conversation.json"

class HandlerLoad:
    def desc(self):
        return "/load - load the current conversation"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question == '/load' else 0.0 
    def do(self, question:str, history:list[ChatMessage]):
        with open(conversation_log, 'r') as file:
            data = json.load(file)
        history.clear()
        for record in data:
            history.append(ChatMessage(role=record['role'], content=record['content']))
        yield f"Conversation loaded from conversation.json"

class HandlerHistory:
    def desc(self):
        return "/history - show the current conversation.  Use '/history full' to see the entire conversation."
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if question[:8] == '/history' else 0.0 
    def do(self, question:str, history:list[ChatMessage]):
        visible_history = list(x for x in history if x.role != "system")
        if 'full' in question:
            for turn in visible_history:
                yield f"-------------------\nROLE: {turn.role}\n{turn.content}\n\n"
        else:
            yield f"The conversation contains {len(visible_history)} turns\n"
            if len(visible_history) > 16:
                yield "The last 16 turns are:\n"
            for turn in visible_history[-16:]:
                content = re.sub("\n+"," / ",turn.content)
                if len(content) > 160:
                    content = content[:150] + "... [truncated]"
                yield " > " + turn.role + " : " + content + "\n"

class HandlerDefault:
    def desc(self):
        return "[text] - send text to the AI"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 0.4
    def do(self, question:str, history:list[ChatMessage]):
        return ask(history=history)

class HandlerUnknown:
    def desc(self):
        return None
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 0.9 if question[0] == "/" else 0.0
    def do(self, question:str, history:list[ChatMessage]):
        yield "Unknown command"

class HandlerWebSearch:
    def desc(self):
        return "/search [text] - search the web for information"
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if "/search" in question[:10] else 0.0
    def do(self, question:str, history:list[ChatMessage]):

        system="""Respond to the users question by creating a web search query that will answer the users question.
Respond only with a web search phrase to be sent to Google or DuckDuckGo.
Do no answer the users question"""

        question = question.replace("/search ", "")
        search_history = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=question)]
        search = "".join(list(ask(history=search_history)))

        yield f"Searching the web using '{search}'...\n"

        search_results:list[WebSearchResult] = web_search(search)
        search_result = "\n".join([f"Title: {result.title}\nURL: {result.href}\nSummary: {result.body}\n" for result in search_results])

        if (len(search_results) == 0 or "EOF" in search_results[0].title):
            yield "[No results found]"
        else:
            yield "## Results for " + search + "\n" + search_result

class HandlerCurl():
    def desc(self) -> str:
        return "/curl [url] - Perform a web request to the specified URL and return the response into the chat."
    def match(self, question:str, history:list[ChatMessage]) -> float:
        return 1.0 if "/curl" in question[:5] else 0
    def do(self, question:str, history:list[ChatMessage]):

        url = question.replace("/curl ", "").strip()

        url_key = re.sub(r"[^a-zA-Z0-9]+", "_", url) + ".md"

        # check in the .webcache directory for a file with the url_key
        cache_file = os.path.join(web_cache_dir, url_key)

        # check if the cache file exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                yield f.read()
                return

        yield f"Fetching the URL ({url}) ...\n"
        webpage = curl(url)

        # get rid of additional white space
        webpage = re.sub("( *\n+ *)+","\n",webpage, flags=re.DOTALL)
        webpage = re.sub(" +"," ",webpage, flags=re.DOTALL)

        for _ in range(5):
            webpage = re.sub("\{[^\{\}]*\}"," ",webpage, flags=re.DOTALL)
            webpage = re.sub("\([^\(\)]*\)"," ",webpage, flags=re.DOTALL)

        webpage = re.sub("\nfunction +[^ \n]* *\n","\n",webpage, flags=re.DOTALL)

        yield "Extracting meaningful information...\n\n"

        message = f"""
--- Start of webpage
{webpage}
--- End of webpage

Convert the above webpage into markdown.  Be sure to include all facts.  Do not include CSS or Javascript.
It is from the url {url}.
"""

        summary = []

        history = [ChatMessage(role="user", content=message)]
        for item in ask(history=history):
            summary.append(item)
            yield item

        
        with open(cache_file, "w") as f:
            f.write("".join(summary))


handlers = [HandlerSet(), HandlerHelp(), HandlerSystem(), HandlerClear(), HandlerFile(), HandlerQuit(), HandlerLoad(), HandlerSave(), HandlerHistory(), HandlerSummarize(), HandlerDefault(), HandlerWebSearch(), HandlerUnknown(), HandlerCurl()]


def process_question(question: str, history: list[ChatMessage]):
    prepend_system_message(history)
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
    
    history.append(ChatMessage(role="user", content=question))
    return handler.do(question, history)

history = []
def main(history) -> None:
    handler = HandlerUnknown()
    while type(handler) != HandlerQuit:
        question = input("Question: ")
        if len(question) == 0:
            continue
        print()

        result = []
        try:
            for out in process_question(question=question, history=history):
                print(end=out)
                result.append(out)
        except Exception as e:
            result = ["Error found : ",f"{e}"]

        history.append(ChatMessage(role="assistant", content="".join(result)))
        print("\n")

if __name__ == "__main__":
    print()
    # try:
    #     for out in HandlerLoad().do(question="/load", history=history):
    #         print(end=out)
    #     print()
    #     for out in HandlerHistory().do(question="/history", history=history):
    #         print(end=out)
    #     print()
    #     main(history)
    # except KeyboardInterrupt as k:
    #     pass
    # except EOFError as e:
    #     pass
    main(history)

    for out in HandlerSave().do(question="/save", history=history):
        print(end=out)
    print()

# print(end="starting ... ")
# for item in process_question(question="/curl https://www.serendipityalpacas.ca/", history=[]):
#     print(end=item)
