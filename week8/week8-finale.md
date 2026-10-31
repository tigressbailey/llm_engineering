# Week 8 收尾：部署、RAG 和砍价 Agent 笔记

这周把前七周收成一个会自己找便宜货的系统。扫描优惠 RSS，估计真实价格，折扣够大就推送到手机，并用 Gradio 看日志和结果表。

估计价格的不是一个模型，而是三个专家的加权：

- Specialist：Week 7 微调过的开源模型，部署在 Modal 上。
- Frontier：GPT 加上 Chroma 里相似商品的价格，也就是 RAG。
- Neural network：Week 6 那个 PyTorch 网络的权重，只做推理。

课程按天递进：

1. Day 1：在 Modal 上部署微调模型，并包成 SpecialistAgent。
2. Day 2：把训练集做成向量库，加上 FrontierAgent 和 EnsembleAgent。
3. Day 3：ScannerAgent 读 RSS，MessagingAgent 用 Pushover 推送。
4. Day 4：PlanningAgent 把上面串起来。另一版让模型自己决定何时调用这三个工具。
5. Day 5：Gradio 界面，`price_is_right.py`。

对应代码：

- 远程服务：`hello.py`、`llama.py`、`pricer_ephemeral.py`、`pricer_service.py`、`pricer_service2.py`
- 编排：`deal_agent_framework.py`
- 界面：`price_is_right.py`、`log_utils.py`
- Agent：`agents/` 下的 `specialist_agent.py`、`frontier_agent.py`、`neural_network_agent.py`、`ensemble_agent.py`、`scanner_agent.py`、`messaging_agent.py`、`planning_agent.py`、`autonomous_planning_agent.py`
- 已发现的优惠记在 `memory.json`
- 向量库目录是 `products_vectorstore`

## 概念

Modal 把一个 Python 函数放到他们的 GPU 容器里跑。本地调用 `pricer.price.remote(description)` 时，机器上不需要加载大模型。第一次构建镜像可能要大约 10 分钟。之后冷启动大约 30 秒，容器还醒着大约 2 秒。`MIN_CONTAINERS = 1` 可以一直留一个容器，延迟稳定在 2 秒左右，但会持续扣额度。也可以把 `scaledown_window` 调到 1200 秒，让它热 20 分钟。

两种部署：

- 临时应用：notebook 一关，容器就没了。用来试 `hello` 和第一次加载 Llama。
- `uv run modal deploy`：变成可长期调用的已部署应用。`SpecialistAgent` 用 `modal.Cls.from_name("pricer-service", "Pricer")` 连上它。

Hugging Face token 在 Modal 里是一条名为 `huggingface-secret` 的 secret，键是 `HF_TOKEN`。代码里如果写的是 `hf-secret`，要和控制台上第一列的名字对齐。区域可以指定，例如 `@app.function(region="eu")`，会略贵一点。

三个估价器看的是同一段改写后的描述：

1. Specialist 把描述送到 Modal，返回微调模型的价格。
2. Frontier 用 `all-MiniLM-L6-v2` 把描述编成 384 维向量，在 Chroma 里取 5 个最像的训练商品和它们的真实价格，放进 prompt，再让 `gpt-5.1` 只回答一个数。这是推理期的 RAG，没有再训练。
3. Neural network 加载 Week 6 的 `deep_neural_network.pth`，本地前向一次。权重在 https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK 这个目录里。

`EnsembleAgent` 的组合是固定权重，不是再训一个回归：

```text
0.8 * frontier + 0.1 * specialist + 0.1 * neural_network
```

前沿模型加 RAG 占大头，是因为 Week 6 里零样本前沿模型已经很强，相似商品的真实价格再补上具体价位。微调和神经网络提供另外两个偏差不同的估计。

扫描到的是 `Deal`：描述、标价、链接。估价减去标价得到 `discount`。`Opportunity` 把这三样放在一起。`PlanningAgent` 只保留折扣超过 50 美元的最好一条，并调用通知。

记忆是一份 URL 列表，写在 `memory.json`。下一轮扫描把已经通知过的链接交给 scanner，避免重复推送。`reset_memory` 只留文件里的前两条，用来演示。

有两个规划器：

- `PlanningAgent`：代码写死顺序。扫描，对前 5 条估价，按折扣排序，超过阈值就通知。
- `AutonomousPlanningAgent`：把 `scan_the_internet_for_bargains`、`estimate_true_value`、`notify_user_of_deal` 登记成工具。模型自己决定调用哪一个、调用几次。这是 Week 2 工具调用的完整版。

## 各天在做什么

### Day 1：Modal 和 Specialist

1. 在 modal.com 建 token，本机执行 `uv run modal token set --token-id ... --token-secret ...`。Windows 上如果虚拟环境读不到用户目录里的 `.modal.toml`，把文件放到实验目录，或把 `MODAL_TOKEN_ID` 和 `MODAL_TOKEN_SECRET` 写进 `.env`。
2. `hello.py` 确认函数能在云上跑，包括指定欧洲区域的版本。
3. 在 Modal 控制台建 secret：名字 `huggingface-secret`，键 `HF_TOKEN`。
4. 部署 pricer 服务。预处理器默认可以用 Llama。想换 Groq，在 `.env` 里设 `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b`。
5. `SpecialistAgent.price` 只是一层远程调用。

### Day 2：向量库和集成

用 sentence-transformers 给训练集编码。全量大约 80 万条，作者在 GPU 上大约 30 分钟，做不完就用 lite。Chroma 集合叫 `products`，metadata 里带价格。t-SNE 加 Plotly 再画一次，看类目是否分开，和 Week 5 的可视化是同一手法。

`FrontierAgent.find_similars` 取 5 条，`make_context` 把「相似商品 + 价格」贴进用户消息。`EnsembleAgent` 先用 `Preprocessor` 把 RSS 里的杂乱文字收成 Week 6 那种四行描述，再问三个专家。

### Day 3：扫描和通知

`ScannerAgent` 拉优惠 RSS，用一个较小的 GPT 把每条收成商品描述和价格，并避开 `memory` 里已有的 URL。输出用 Pydantic 模型 `DealSelection` 约束成 JSON。

Pushover：在 https://pushover.net/ 注册，建一个应用，得到 user key 和 API token，写入 `.env` 的 `PUSHOVER_USER` 和 `PUSHOVER_TOKEN`，手机装上客户端。`MessagingAgent` 先让模型写成一条短通知，再 POST 到 Pushover。

### Day 4：规划

先用假函数把工具 JSON 走通，再换成真的 scanner、ensemble、messenger。`DealAgentFramework.run` 调用 `planner.plan(memory)`，有结果就追加进 `memory.json`。

固定流程的阈值是 `DEAL_THRESHOLD = 50`。自主规划那一版由模型决定要不要通知，工具实现里仍然做扫描、估价和推送。

### Day 5：界面

`price_is_right.py` 用 Gradio Blocks。一块滚动显示日志，一块表格列出商品、标价、估计价、折扣和链接。日志从 logging 队列里抽，所以 agent 里的 `self.log` 会出现在页面上。运行就是执行这个文件，让循环不断调用 `DealAgentFramework.run`。

作者在另一个仓库 tech2ai 的 segment4 里，把自主规划换成了 OpenAI Agents SDK，并接了 MCP。那是这门课之后的延伸，不是这周要交的代码。

## 库各自干什么

- `modal`：构建镜像、声明 GPU 函数、读取 Hugging Face secret、`deploy` 之后用 `Cls.from_name` 远程调用。
- `huggingface_hub` 和 Modal 的 secret：容器启动时下载 Week 7 的基座和适配器。
- `sentence_transformers`：`all-MiniLM-L6-v2`，本地把商品描述变成 384 维向量。
- `chromadb`：`PersistentClient` 存在 `products_vectorstore`，集合名 `products`。查询返回文档和 metadata 里的价格。
- `openai`：FrontierAgent 的 `gpt-5.1`，以及 scanner、messenger 里的小模型。
- `torch`：加载 `deep_neural_network.pth` 做推理。
- `litellm`：`Preprocessor` 把原始描述改写成统一的五行文本。
- `pydantic`：`Deal`、`DealSelection`、`Opportunity`，用来校验模型吐出的 JSON。
- `feedparser`：Scanner 读取 DealNews 的 RSS。解析在 `agents/deals.py` 的 `ScrapedDeal.fetch`。
- `requests`：MessagingAgent 调用 Pushover 的 HTTP API。
- `gradio`：最终界面。
- `sklearn` 的 t-SNE、`plotly`：只用于查看向量是否按类目聚在一起。
