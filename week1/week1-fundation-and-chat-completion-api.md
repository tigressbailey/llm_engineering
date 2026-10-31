# Week 1 基础与 Chat Completions API 笔记

这周第一次用代码调用前沿模型。项目是一个会总结网页的小工具：给一个 URL，抓下正文，再让模型写成摘要。周末把它升级成公司宣传册：先挑出网站上有用的链接，再根据多页内容生成一份给客户、投资人和求职者看的 brochure。

课程按天递进：

1. Day 1：抓网页，用 system 和 user 两条消息调用 OpenAI，生成摘要。
2. Day 2：Chat Completions 是什么，以及怎么用同一个 OpenAI 客户端去调 Gemini 和本机 Ollama。
3. Day 4：用 tiktoken 看 token，并弄清模型没有记忆。
4. Day 5：两次调用模型，先选链接，再写宣传册，并用流式输出。

对应代码：

- 抓网页：`scraper.py`
- 用本地 Ollama 做摘要：`solution.py`
- 练习：`week1 EXERCISE.ipynb`
- 正课：`day1.ipynb`、`day2.ipynb`、`day4.ipynb`、`day5.ipynb`

## 概念

Frontier model 是当前能力最强的那一档闭源模型。这周点名的有 OpenAI 的 GPT、Anthropic 的 Claude、Google 的 Gemini、xAI 的 Grok。开源或可本地跑的有 Meta 的 Llama、Mistral 的 Mixtral、阿里的 Qwen、Google 的 Gemma、微软的 Phi、DeepSeek、OpenAI 的 GPT-OSS。本地小模型不收 API 费，数据不出本机，能力通常弱于前沿模型。

Chat Completions 是调用聊天模型最常用的接口。名字的意思是：这里有一段已经发生的对话，请预测接下来该出现什么。它由 OpenAI 先做出来，后来几乎所有厂商都提供了同一种消息格式。

一次调用至少有两种角色：

- system：规定任务和语气，例如「你是一个刻薄的助手，用 markdown 写短摘要」。
- user：这一轮要处理的具体内容，例如网页正文。

API 收到的是一个列表：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
```

多轮对话再追加 `assistant`（模型上一轮说的话）和新的 `user`。Day 4 的重点是：每一次 HTTP 调用都是无状态的。模型不会记得上次你说过名字。所谓记忆，是工程师每次把至今为止的整段对话再发一遍。模型只是在这条更长的序列上预测下一个 token。ChatGPT 产品也是这个做法。代价是每一轮都要为历史内容付 token 费，因为模型确实要重新看过这些上下文。

Endpoint 是一个 HTTP 地址。`openai` 这个 Python 包只是对 HTTP 请求的薄封装，里面没有模型权重。换 `base_url` 和 `api_key`，同一个客户端就能打到别的兼容端点。

- Gemini：`https://generativelanguage.googleapis.com/v1beta/openai/`
- 本机 Ollama：`http://localhost:11434/v1`，`api_key` 可以随便填，例如 `ollama`

One-shot prompting 是在提示里给一个完整例子，让模型照着格式回答。Day 5 用它要求模型返回哪些链接值得跟进。更严格的 Structured Outputs 留到 Week 8。

流式输出（streaming）让 token 一边生成一边回来，界面上像打字机。宣传册那一步用了这个。

抓网页的限制：`requests` 加 BeautifulSoup 只能拿到服务器直接返回的 HTML。用 JavaScript 渲染的页面（很多 React 站）正文是空的。CloudFront 一类防护可能直接 403。Day 1 把正文截到 2000 字符，避免一次塞太多 token。

## 各天在做什么

### Day 1：网页摘要

1. `load_dotenv` 读 `.env` 里的 `OPENAI_API_KEY`。
2. `fetch_website_contents` 发 GET，去掉 script、style、img、input，拼上标题，截断到 2000 字符。
3. `messages_for` 把 system prompt 和「请总结这段网页」加正文装成消息列表。
4. `openai.chat.completions.create(model=..., messages=...)` 取 `response.choices[0].message.content`。

商业上这就是摘要：新闻、财报、简历都可以换成同一套「正文进 user，要求进 system」。

### Day 2：同一个客户端，不同端点

先再调一次 OpenAI，确认 Chat Completions 的形状。然后把 `OpenAI(base_url=..., api_key=...)` 指到 Gemini，或指到本机 Ollama。作业是把 Day 1 的摘要从 OpenAI 改成 Ollama 上的 `llama3.2`。机器小就用 `llama3.2:1b`。`llama3.3` 和 Llama 4 对笔记本来说太大。`solution.py` 就是这个作业的一种写法。

Ollama 要先安装，浏览器打开 `http://localhost:11434/` 应看到 `Ollama is running`。没启动就在终端跑 `ollama serve`，再 `ollama pull llama3.2`。

### Day 4：token 和假记忆

用 tiktoken 把一段文字编成 token id，再逐个 decode，看模型实际吃进去的是词片段而不是整词。接着故意做两轮独立调用：第一轮告诉它名字，第二轮只问「我叫什么」，它答不上来。把第一轮的 user 和 assistant 放进第二次的 `messages`，它就能答出来。

### Day 5：宣传册

输入是公司名和主页 URL。

1. `fetch_website_links` 取出页面上的链接。
2. 第一次调用模型（notebook 里用 gpt-5-nano）：判断哪些链接和公司介绍、客户、招聘有关，并把 `/about` 这种相对路径补成完整 URL。提示里给一个 JSON 例子，这就是 one-shot。
3. 把选中的页面再抓下来，拼进第二次调用，写出 markdown 宣传册。改 system prompt 就能把语气从正式换成幽默。
4. 用流式接口把宣传册逐段打出来。

这两次调用已经是后来 agent 的雏形：一个模型的输出成为下一个模型的输入。

## 库各自干什么

- `openai`：Chat Completions 的 Python 客户端。`OpenAI()` 默认打 OpenAI。传入 `base_url` 就打兼容端点。
- `python-dotenv` 的 `load_dotenv`：从 `.env` 读密钥。改完 `.env` 要保存，并重新 `load_dotenv(override=True)`。
- `requests`：发 HTTP GET 抓网页。
- `bs4.BeautifulSoup`：解析 HTML，取标题、正文和链接。
- `tiktoken`：按某个模型的词表把文字切成 token，用来看长度，不负责生成。
- Ollama：在本机提供一个 OpenAI 兼容的 HTTP 服务，默认端口 11434，后面跑的是 Llama 这类开源权重。
