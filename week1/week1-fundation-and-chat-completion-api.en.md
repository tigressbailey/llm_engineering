# Week 1: Foundations and the Chat Completions API

This week calls a frontier model from code for the first time. The project is a small web summarizer: give it a URL, fetch the page text, and have the model write a summary. By the end of the week it becomes a company brochure: pick the useful links on a site, then write a brochure for prospective clients, investors, and recruits from those pages.

The days build in this order:

1. Day 1: fetch a page, call OpenAI with a system message and a user message, and produce a summary.
2. Day 2: what Chat Completions is, and how the same OpenAI client calls Gemini and local Ollama.
3. Day 4: inspect tokens with tiktoken, and see that the model has no memory of its own.
4. Day 5: two model calls, first to choose links, then to write the brochure, with streaming.

Code for this week:

- Fetch pages: `scraper.py`
- Summarize with local Ollama: `solution.py`
- Exercise: `week1 EXERCISE.ipynb`
- Labs: `day1.ipynb`, `day2.ipynb`, `day4.ipynb`, `day5.ipynb`

## Concepts

A frontier model is a closed model at the current top of capability. This week names OpenAI GPT, Anthropic Claude, Google Gemini, and xAI Grok. Models you can run locally or treat as open include Meta Llama, Mistral Mixtral, Alibaba Qwen, Google Gemma, Microsoft Phi, DeepSeek, and OpenAI GPT-OSS. A small local model has no API fee and the data stays on the machine. Its capability is usually below a frontier model.

Chat Completions is the usual API for a chat model. The name means: here is a conversation so far, predict what should come next. OpenAI introduced it. Later, almost every provider offered the same message shape.

A call has at least two roles:

- system: the task and the tone, for example "you are a snarky assistant, write a short summary in markdown."
- user: the material for this turn, for example the page text.

The API receives a list:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
```

A longer conversation appends `assistant` (what the model said last turn) and a new `user` message. The point of Day 4: every HTTP call is stateless. The model does not remember that you told it your name last time. The appearance of memory comes from the engineer resending the whole conversation so far on every call. The model predicts the next token on that longer sequence. The ChatGPT product works the same way. Each turn pays for the history as well, because the model really does read that context again.

An endpoint is an HTTP address. The `openai` Python package is a thin wrapper around that HTTP call. It does not contain model weights. Change `base_url` and `api_key`, and the same client talks to another compatible endpoint.

- Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`
- Local Ollama: `http://localhost:11434/v1`. The `api_key` can be any string, for example `ollama`.

One-shot prompting puts one complete example in the prompt so the model copies the format. Day 5 uses it to ask which links are worth following. Stricter Structured Outputs wait until Week 8.

Streaming sends tokens back as they are generated, so the UI looks like a typewriter. The brochure step uses this.

Limits of fetching pages: `requests` plus BeautifulSoup only see the HTML the server returns directly. Pages rendered by JavaScript, including many React sites, come back empty. CloudFront-style protection can return 403. Day 1 truncates the body at 2,000 characters so one call does not carry too many tokens.

## What each day does

### Day 1: page summary

1. `load_dotenv` reads `OPENAI_API_KEY` from `.env`.
2. `fetch_website_contents` sends a GET, drops script, style, img, and input tags, keeps the title, and truncates to 2,000 characters.
3. `messages_for` packs the system prompt and "summarize this page" plus the body into a message list.
4. `openai.chat.completions.create(model=..., messages=...)` reads `response.choices[0].message.content`.

In a product this is summarization. News, earnings, and resumes use the same split: body in the user message, instructions in the system message.

### Day 2: one client, different endpoints

Call OpenAI once more and confirm the Chat Completions shape. Then point `OpenAI(base_url=..., api_key=...)` at Gemini, or at local Ollama. The homework replaces the Day 1 OpenAI summary with Ollama's `llama3.2`. On a small machine use `llama3.2:1b`. `llama3.3` and Llama 4 are too large for a laptop. `solution.py` is one way to write that homework.

Install Ollama first. Opening `http://localhost:11434/` should show `Ollama is running`. If it does not, run `ollama serve` in a terminal, then `ollama pull llama3.2`.

### Day 4: tokens and the illusion of memory

tiktoken encodes a string into token ids, then decodes them one by one, so you can see the model consumes word pieces, which may be smaller than whole words. Then make two independent calls on purpose. The first tells the model a name. The second only asks "what is my name?" and it cannot answer. Put the first turn's user and assistant messages into the second call's `messages`, and it can.

### Day 5: brochure

The input is a company name and a homepage URL.

1. `fetch_website_links` collects the links on the page.
2. The first model call (the notebook uses gpt-5-nano) decides which links are about the company, customers, or jobs, and turns a relative path such as `/about` into a full URL. The prompt includes one JSON example. That is the one-shot.
3. Fetch the chosen pages, put them into a second call, and write a markdown brochure. Changing the system prompt switches the tone from formal to humorous.
4. Stream the brochure back piece by piece.

These two calls are an early form of a later agent: one model's output becomes the next model's input.

## What each library does

- `openai`: the Python client for Chat Completions. `OpenAI()` talks to OpenAI by default. Pass `base_url` to talk to a compatible endpoint.
- `load_dotenv` from `python-dotenv`: reads secrets from `.env`. After editing `.env`, save it and run `load_dotenv(override=True)` again.
- `requests`: HTTP GET for the page.
- `bs4.BeautifulSoup`: parses HTML and extracts the title, body, and links.
- `tiktoken`: splits text with a model's vocabulary so you can see length. It does not generate text.
- Ollama: an OpenAI-compatible HTTP server on the machine, default port 11434, with open weights such as Llama behind it.
