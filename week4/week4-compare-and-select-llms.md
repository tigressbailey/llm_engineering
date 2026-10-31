# Week 4 比较并选择模型笔记

这周的任务不是再接一个新 API，而是学会怎么为一项业务选模型，并用一个能算时间的实验把选择做实。实验是：让模型把一段 Python 改写成更快的 C++（Day 5 再改写成 Rust），编译运行，比加速倍数。

课程按天递进：

1. Day 1：比功能、比成本、看缩放法则和基准测试的局限。
2. Day 2：五张常用排行榜，以及盲测竞技场。
3. Day 3：AI 工程师的五步，并用前沿闭源模型做 Python 到 C++。
4. Day 4：同一道题改用开源模型（Ollama、Groq、OpenRouter）。
5. Day 5：把目标语言换成 Rust，并区分技术指标和业务指标。

对应代码：

- 正课：`day1.ipynb` 到 `day5.ipynb`
- 让模型知道你这台机器的编译器：`system_info.py`
- Gradio 样式：`styles.py`

## 概念

选模型先分两列看。

能力相关：

- 开源还是闭源。闭源走 API。开源可以本机或 Groq 这类托管推理。
- 聊天、推理，还是两者混合。推理模型会花更多 token 在内部步骤上。
- 发布日期和知识截止日期。截止日期之后的事实它没在训练里见过。
- 参数量和训练 token 数。
- 上下文窗口有多长。

成本相关：

- 许可证能不能商用。
- 推理费用：按 token 的 API 价，或自己 GPU 的电费和租金。
- 训练或微调费用。
- 接到产品里要写多少胶水代码，多久能上线。
- 速率限制、生成速度、首 token 延迟。

Chinchilla 缩放法则（DeepMind，2022）说：固定算力时，参数量和训练 token 要一起涨，单独把模型做大或单独堆数据都会浪费。经验比例大约是每个参数配 20 个训练 token。GPT-3 是 1750 亿参数配相对较少的数据。Chinchilla 用约 700 亿参数和更多数据，结果超过了 GPT-3。这条法则是选「基座模型怎么训练才划算」的经验，不是你调用 API 时要调的旋钮。

课程点了六个用来拉开差距的难基准：

- HLE（Humanity's Last Exam）：各领域专家出的极难题。
- GPQA：研究生水平的科学题，搜索引擎上没有现成答案。
- MMLU-Pro：多学科选择题的加强版。
- AIME：高难度数学竞赛。
- LiveCodeBench：持续更新的编程题，减轻「题目已经进训练集」的问题。
- MuSR：多步软推理，例如谜题和情景。

基准分数不能单独当选型依据。常见问题：

- 训练数据污染：题和答案可能已经在训练集里。
- 各家提示词、温度、评测脚本不一致，分数不好横比。
- 一项考试只覆盖一种能力。
- 有标准答案的题测不出语气、判断和业务是否真的变好。
- 饱和：顶尖模型都接近满分，榜单失去区分度。
- 过拟合榜单：为了这一套题优化，真实任务反而变差。
- 模型有可能识别出自己正在被评测。

所以还要看人类盲测。竞技场（LM Arena，现在常写作 arena.ai）把两个匿名模型的回答并排给真人选，用 Elo 一类的分数排名。它比较的是人的偏好，和自动考试不是同一件事。

排行榜可以当候选清单，不应当成最终决定：

- Artificial Analysis：https://artificialanalysis.ai/ 总览用 Intelligence Index 对上每次任务的成本，编码用 Coding Agent Index 对上每次任务的成本。落在又强又便宜的那一区的模型优先看。评测说明在 https://artificialanalysis.ai/evaluations
- Vellum：https://www.vellum.ai/llm-leaderboard 同一页能看到上下文长度和 API 价格。
- Scale SEAL：https://labs.scale.com/leaderboard
- Hugging Face Open LLM Leaderboard：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard 面向开源模型。
- Live Bench：https://live-bench.ai/ 题目持续更换，用来减轻数据污染。

AI 工程师同时做两件事：搞清楚数据和问题（数据科学），以及把系统做出来（软件工程）。角色可以偏三种：把系统做快（系统优化）、把模型做准（调模型）、在模型上面做产品（应用）。

选型的五步：

1. 弄清业务问题，以及用什么数字判断成功。闲聊返回一个销售额，本身还不是业务问题。
2. 准备：用上面的榜单圈出几个候选，看窗口、价格、许可证、知识截止日期。
3. 选择：做小原型，在你的任务上比，而不是只看榜。
4. 定制：Week 5 的 RAG，Week 6 和 Week 7 的微调，Week 8 的 agent。
5. 产品化：部署、限流、监控、密钥。

评估分两层：

- 模型指标：loss、perplexity、准确率、precision、AUC。好优化，离业务远。
- 业务指标：这周实验里就是同一段计算跑完要多少秒，以及加速倍数。还可以是 ROI、节省的时间、客户是否满意。

## 实验：Python 改写成更快的本地代码

题目是让模型把一段 Python 收成高性能的 C++，在你这台机器上编译运行。跑不跑编译是可选的，也可以把代码贴到 https://www.programiz.com/cpp-programming/online-compiler/ 看个大致差别。`system_info.py` 收集操作系统和编译器，写进提示，让生成的代码符合本机。

Day 3 用较贵的闭源模型：GPT-5、Claude Sonnet 4.5、Gemini 2.5 Pro、Grok 4。要省钱就换成 gpt-5-nano 这一档。Day 4 换开源：本机 Ollama，以及 Groq、OpenRouter 上的托管开源模型。

笔记本里记录的一次 C++ 加速（相对原来的 Python，Ed 的实验）：

- Gemini 2.5 Pro：1440 倍
- Grok 4：1060 倍
- gpt-oss 20B：238 倍
- GPT-5：233 倍
- Claude Sonnet 4.5：184 倍
- Qwen3 Coder 30B：168 倍
- DeepSeek Coder v2：168 倍
- gpt-oss 120B：14 倍
- Qwen 2.5 Coder：编译或运行失败

Day 5 改成生成 Rust 之后，名次变了。多个闭源模型失败，gpt-oss 120B 变成最快的一档（记录的耗时约 0.000304），Grok 4 其次，gpt-oss 20B 再次。同一批模型，换一种目标语言，排名可以翻转。这就是「必须在你的任务上做原型」的原因。

商业上，模型的用法大致三类：在别人的产品外面包一层聊天（语言学习、编程助手），做成垂直产品（法律、招聘、医疗、数据平台），以及做成会调用工具的 agent（编程 agent、通用 agent）。

## 库各自干什么

这周几乎不引入新的模型库，用的还是 Week 2 的客户端。

- `openai` 客户端，以及 Groq、OpenRouter、Ollama 的兼容端点：同一道改写题目打到不同模型。
- `subprocess` 和 `system_info.py`：探测本机编译器，并在你愿意时编译、计时。
- `gradio` 和 `styles.py`：如果把比较过程做成小界面，用它摆样式。
- 排行榜网站本身不是库。它们用来生成候选名单，实验计时用来做最终选择。
