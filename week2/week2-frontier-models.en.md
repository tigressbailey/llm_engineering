# Week 2: Frontier Models

This week takes Week 1's single call and adds model switching, a UI, conversation history, and calls into your own functions. The project is an airline support assistant: look up fares and cities, then add images and audio.

The days build in this order:

1. Day 1: call several frontier models with the same message format, look at prompt cache, and let two models talk to each other.
2. Day 2: build a UI in Gradio and put an input box on the Week 1 brochure.
3. Day 3: `chat(message, history)`, a chatbot that receives the full history.
4. Day 4: tool calling. The model only says "please run this function." Our code runs it.
5. Day 5: the airline assistant adds a database lookup and image generation, laid out with Gradio Blocks.

Code for this week:

- Labs: `day1.ipynb` through `day5.ipynb`
- End-of-week exercise: `week2 EXERCISE.ipynb`
- Helpers: `scraper.py`, `generator_practice.py`, `revealer.py`

## Concepts

Many providers expose an OpenAI-compatible endpoint, so Day 1 still uses the `openai` package and only changes `base_url` and the key. The notebook connects to OpenAI, Anthropic, Gemini, DeepSeek, Groq, and Grok. OpenRouter is a router: one key can reach many models, for example GLM from z.ai. Each provider also has an official SDK. The compatible endpoint exists so you can keep one client.

Training and inference are different times. Training changes weights. Inference leaves the weights fixed and generates an output from an input. This week is entirely inference. Training starts in Week 6.

Prompt caching makes a repeated long prefix cheaper. A hit requires the start of the two requests to match character for character. Put stable instructions, examples, and tool definitions first. Put the user content that changes every call at the end.

- OpenAI: automatic prefix matching. Cached input is about one quarter of the normal input price in their docs.
- Anthropic: you mark which span is cached. Writing the cache costs extra. Later hits are much cheaper.
- Gemini: both implicit and explicit caching.

The shape of conversation history:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "first turn"},
    {"role": "assistant", "content": "the model's reply"},
    {"role": "user", "content": "the next turn"},
]
```

When two models argue, each call keeps that model's own system message at the front, then appends what the other model just said and calls again. A stabler three-way conversation sends one system message and one user message each time. The user message pastes the whole conversation so far and says "you are Alex, say the next line." You do not have to forge another participant's assistant role.

Tools do not mean the model executes code on your machine. The flow is:

1. You describe the function name, arguments, and purpose in JSON, and pass that as `tools`.
2. If the model decides a fare lookup is needed, the response contains the function name and arguments, and it is not yet the final answer.
3. Your code actually calls `get_ticket_price` and sends the return value back as another message.
4. The model writes the sentence a person should read.

One response can request several tools. You can also chain tool, model, tool across several steps. The Day 4 exercise adds a tool that sets a ticket price.

Gradio builds a frontend from the Python description, starts a local server on Starlette, and mounts your Python functions as backend routes. Three UIs:

- `gr.Interface`: one function behind a simple form.
- `gr.ChatInterface`: a standard chat. The callback is `chat(message, history)`.
- `gr.Blocks`: you place text boxes, images, and buttons, then wire events to callbacks. The Day 5 airline assistant uses this.

`share=True` publishes the local UI for a while through Gradio's tunnel. Corporate networks and antivirus software often block it. `auth` adds a username and password. Keep the password in `.env`. Streaming uses a generator: `yield` pieces of text from the function.

The Responses API is a newer OpenAI interface. This course stays on `chat.completions`, because that shape is compatible across providers. The Responses API is tied to OpenAI, and several of its features are paid.

## What each day does

### Day 1: switching models and cache

Keys go in `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `GROK_API_KEY`, `OPENROUTER_API_KEY`. Skip any provider you do not have. Local models are still `localhost:11434/v1`.

Then use LiteLLM to look at caching, and let one cheap model debate another cheap model, each with its own system prompt. Changing the system prompt turns a contrarian into a pessimist or an optimist.

### Day 2: Gradio

Start from a `shout` function, add Markdown output, dark mode, login, and streaming, then replace the callback with `message_gpt`. The system text lives in a global variable. Changing it changes the model's tone. The exercise is to add a company-URL input to the Week 1 Day 5 brochure.

### Day 3: chatbot

`chat(message, history)` receives the current sentence and the prior conversation, builds the message list, calls the model, and returns a string. Later the system prompt says the assistant knows this company, and includes one example answer. That is one-shot prompting again.

### Day 4 and Day 5: airline support

The example tool looks up a fare by city. The fare can come from a small SQLite file, `prices.db`. The improvements are handling several tool calls in one response, and handling tool calls that follow one another. Day 5 adds:

- An image model draws a vacation picture for a city. The notebook mentions DALL-E 3 at about 4 cents per image. The code also uses `gpt-image-1-mini`.
- Audio as the other multimodal path.
- Blocks puts chat, images, and tools on one page.

This is one step of an agent workflow: the model decides whether to query the database, the program runs the query, and the result goes back to the model.

## What each library does

- `openai`: still the default client. Change `base_url` to call Gemini, Groq, DeepSeek, or Ollama.
- Official Anthropic and Google SDKs: another way to make the same call. The course uses compatible endpoints first, then points out that official libraries exist.
- OpenRouter: one key routed to many models.
- `litellm`: a thinner layer. `completion(model="provider/model", messages=...)` unifies providers and is used to inspect the prompt cache.
- `langchain`: Day 1 is only an introduction. It wraps models and messages again. It can do more, and it is heavier. Week 5 is where the course uses it for RAG.
- `gradio`: chat and form UIs. `Interface`, `ChatInterface`, `Blocks`.
- `python-dotenv`: reads `.env`.
- SQLite (`prices.db`): the ordinary database behind the tool, fares by city.
- `requests` and `BeautifulSoup`: the brochure path still fetches web pages.
