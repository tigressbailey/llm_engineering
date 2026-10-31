# Week 2 前沿模型笔记

这周把 Week 1 的一次调用，扩成能换模型、能做界面、能记住对话、能调用你自己的函数。项目落在一个航空公司客服：查票价、查城市，后面再加上图片和语音。

课程按天递进：

1. Day 1：用同一套消息格式调用多家前沿模型，看 prompt cache，并让两个模型对着聊天。
2. Day 2：用 Gradio 做界面，把 Week 1 的宣传册套上输入框。
3. Day 3：`chat(message, history)`，用完整历史做出聊天机器人。
4. Day 4：工具调用。模型只表示「请你运行这个函数」，真正的代码由我们执行。
5. Day 5：航空客服加上数据库查询、图片生成，并用 Gradio Blocks 拼界面。

对应代码：

- 正课：`day1.ipynb` 到 `day5.ipynb`
- 周末练习：`week2 EXERCISE.ipynb`
- 辅助：`scraper.py`、`generator_practice.py`、`revealer.py`

## 概念

多家厂商都提供 OpenAI 兼容端点，所以 Day 1 仍然用 `openai` 这个包，只换 `base_url` 和 key。notebook 里接过 OpenAI、Anthropic、Gemini、DeepSeek、Groq、Grok。OpenRouter 是一个路由：一把 key 后面可以转到很多模型，例如 z.ai 的 GLM。各家也有自己的官方 SDK，兼容端点只是为了少写一套客户端。

推理和训练是两段时间。训练时改权重。推理时权重不动，只根据输入生成输出。这周全部是推理。后面 Week 6 才训练。

Prompt caching 让重复的长前缀更便宜。能命中的条件是：两次请求的开头逐字相同。所以固定的说明、例子、工具定义放在前面，每次变化的用户内容放在后面。

- OpenAI：自动按前缀匹配。文档里缓存输入大约是普通输入价格的四分之一。
- Anthropic：要显式标记哪一段被缓存。写入缓存要多付一截，之后命中便宜很多。
- Gemini：隐式和显式两种都有。

对话历史的形状：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "第一句"},
    {"role": "assistant", "content": "模型的回复"},
    {"role": "user", "content": "下一句"},
]
```

两个模型对骂时，每一方只把自己的 system 放在开头，然后把对方刚说的话追加进同一个 `messages` 列表再调用。三方对话更稳的写法是每次只发一条 system 和一条 user，user 里贴上至今为止的全部发言，并写明「你现在是 Alex，请说下一句」。这样不必伪造别人的 assistant 角色。

工具（tools）不是模型在你机器上直接执行代码。流程是：

1. 你用一份 JSON 描述函数名、参数和用途，放进 `tools`。
2. 模型如果觉得该查票价，就在响应里给出函数名和参数，而不是最终答案。
3. 你的代码真正调用 `get_ticket_price`，把返回值再作为一条消息发回去。
4. 模型根据这个结果写出给人看的句子。

一次响应里可以要求多个工具。也可以工具、再调用、再工具，串成好几步。Day 4 的练习是再加一个「设置票价」的工具。

Gradio 根据 Python 描述生成前端，用 Starlette 起一个本地服务，并把你的 Python 函数挂成后端路由。三种界面：

- `gr.Interface`：一个函数对一个简单表单。
- `gr.ChatInterface`：标准聊天。回调是 `chat(message, history)`。
- `gr.Blocks`：自己摆文本框、图片、按钮，再把事件接到回调上。Day 5 的航空助手用这个。

`share=True` 通过 Gradio 的隧道把本地界面临时公开。公司网络或杀毒软件常会拦住。`auth` 可以加用户名密码，密码应放在 `.env`。流式响应用生成器：函数里 `yield` 一段段文本。

Responses API 是 OpenAI 较新的接口。这门课明确继续用 `chat.completions`，因为后者各家兼容，Responses API 绑在 OpenAI 上，而且不少能力要额外付费。

## 各天在做什么

### Day 1：换模型和缓存

密钥写进 `.env`：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`、`DEEPSEEK_API_KEY`、`GROQ_API_KEY`、`GROK_API_KEY`、`OPENROUTER_API_KEY`。没有的可以只看不跑。本地模型仍是 `localhost:11434/v1`。

接着用 LiteLLM 看缓存，再让一个便宜模型和另一个便宜模型按各自的 system prompt 辩论。改 system prompt 就能把角色从抬杠换成悲观或乐观。

### Day 2：Gradio

从一个 `shout` 函数开始，加上 Markdown 输出、深色模式、登录、流式输出，最后把回调换成 `message_gpt`。system 文案存在全局变量里，改它就改了模型语气。作业是给 Week 1 Day 5 的宣传册加一个输入公司 URL 的界面。

### Day 3：聊天机器人

`chat(message, history)` 收到当前这句话和之前的对话，拼成消息列表，调用模型，返回字符串。后面把 system prompt 写成「你了解这家公司」，并给一个示例回答，这又是 one-shot。

### Day 4 和 Day 5：航空客服

工具例子是按城市查票价，票价可以来自一个小的 SQLite（`prices.db`）。改进点是一次处理多个 tool call，以及连续多轮 tool call。Day 5 加上：

- 用图像模型按城市画一张度假图。notebook 里提到 DALL-E 3，大约一张 4 美分，代码里也出现过 `gpt-image-1-mini`。
- 语音或音频作为多模态的另一路。
- Blocks 把聊天、图片和工具串在一个页面上。

这是 agent 工作流的一步：模型决定要不要查数据库，程序执行，再把结果交回模型。

## 库各自干什么

- `openai`：仍然是默认客户端。换 `base_url` 调 Gemini、Groq、DeepSeek、Ollama。
- Anthropic、Google 官方 SDK：同一件事的另一套写法。课程先走兼容端点，再点出官方库的存在。
- OpenRouter：一个 key 路由到多家模型。
- `litellm`：更薄的一层，`completion(model="provider/model", messages=...)` 统一各家，并用来看 prompt cache。
- `langchain`：Day 1 只是第一次见面。它把模型和消息再包一层，功能多，也更重。Week 5 才正式用它做 RAG。
- `gradio`：聊天和表单界面。`Interface`、`ChatInterface`、`Blocks`。
- `python-dotenv`：读 `.env`。
- SQLite（`prices.db`）：工具函数背后的普通数据库，按城市查票价。
- `requests`、`BeautifulSoup`：宣传册那条线继续抓网页。
