# Week 5 RAG 笔记

这周做的是 Insurellm（虚构保险科技公司）的内部问答助手。员工问产品、员工、合同、公司资料时，答案要准，成本要低。做法是 RAG：先从知识库里找出相关段落，再把这些段落放进 prompt，让生成模型据此回答。

知识库在 `knowledge-base/`，全是 markdown，按文件夹分类型：`products`、`employees`、`contracts`、`company`。

课程按天递进：

1. Day 1：关键词硬匹配，理解 RAG 为什么要把上下文塞进 prompt。
2. Day 2：切块、做向量、存进 Chroma，并画出来检查。
3. Day 3：用向量库检索，再调用聊天模型回答。
4. Day 4：用测试集分别评估「找得准不准」和「答得好不好」。
5. Day 5：丢掉 LangChain，用 LLM 切块、改写问题、重排结果。

对应代码：

- 基础入库：`implementation/ingest.py`
- 基础问答：`implementation/answer.py`
- 聊天界面：`app.py`
- 评估：`evaluation/test.py`、`evaluation/eval.py`、`evaluation/tests.jsonl`
- 进阶入库和问答：`pro_implementation/ingest.py`、`pro_implementation/answer.py`

## 概念

RAG 是 Retrieval Augmented Generation。模型自己的权重里没有这家公司的内部资料。每次提问时，程序先去知识库检索，把相关原文追加到 prompt，模型再生成答案。课程里的做法是把检索到的内容放进 system prompt，用户问题仍放在 user prompt。

没有 RAG 时，模型只能靠训练时见过的公开知识，公司合同和员工档案会答错或瞎编。把整库原文每次都塞进 prompt 也不行：token 贵，上下文有长度上限，无关段落还会干扰回答。RAG 只塞最相关的几段。

两阶段：

1. 入库（ingest），离线做一次，文档有更新时重做。读文档，切成 chunk，每个 chunk 变成向量，连同原文和 metadata 存进向量库。
2. 问答（query），每次用户提问都做。问题也变成向量，在向量库里找最接近的 chunk，拼进 prompt，调用聊天模型。

向量嵌入（embedding）是一段文字变成的一串浮点数。意思接近的文字，向量也接近。检索比的是向量距离，所以「谁创立了公司」可以命中写着 founder 的段落，不要求问题里出现文档里的原词。

生成模型和嵌入模型不是同一种：

- 自回归模型看前文，预测下一个 token，用来生成答案。GPT 这一类。
- 自编码模型通读整段输入，用来理解和编码。BERT、OpenAI Embeddings、Hugging Face 的 sentence-transformers 这一类。向量是它算出来的，不是向量数据库算出来的。

Token 和向量也不是一回事。Token 是切分文字后的小块，在词表里是一个编号，本身不能比较意思。向量是表示含义的数字。文字先切成 token，嵌入模型再把它们收成一个向量。

`tiktoken` 只负责按某个模型的词表数 token，用来估算长度和费用。它不产生 RAG 用的向量。

- `encoding_for_model("gpt-4.1-nano")` 先查模型名对照表，再取出对应分词器。对照表在已安装的 `tiktoken/model.py`，字典是 `MODEL_TO_ENCODING`（全名）和 `MODEL_PREFIX_TO_ENCODING`（前缀）。
- `get_encoding("o200k_base")` 直接拿分词器。`o200k_base` 是分词器名字，不是模型名。`o` 来自 GPT-4o 那一代，`200k` 表示词表大约 20 万，`base` 是标准版。GPT-4o、GPT-4.1、GPT-5 都指向它。更早的 GPT-4 用 `cl100k_base`（大约 10 万）。
- 当前安装的 tiktoken 还不认识 `gpt-6-luna`。未合并的 [tiktoken PR 616](https://github.com/openai/tiktoken/pull/616) 打算把 GPT-6 家族也指到 `o200k_base`。本地计数可以写 `tiktoken.get_encoding("o200k_base")`。不要把 `MODEL` 设成 `o200k_base`，`encoding_for_model` 只认模型名。

Chunk 是检索的单位。整篇文档太长，塞进 prompt 浪费 token，也容易把无关句子带进来。切太碎又会丢掉上下文。常见折中是固定字符数，相邻块重叠一段，避免一句话被切在边界上。Day 2 notebook 用 1000 字符、重叠 200。`implementation/ingest.py` 用 500 字符、重叠 200。重叠意味着同一句话可能出现在两个块里，检索更不容易漏，存储会变多。

Metadata 是附在块上的标签，例如来源路径和 `doc_type`。检索结果可以显示出处，评估和过滤也靠它。

Temperature 控制生成时怎么抽 token。0 表示尽量总选概率最高的 token，适合要事实的问答。它不等于创造力。想让口吻更灵活，改 system prompt。temperature 为 0 也不保证每次输出逐字相同。

## 一条 RAG 怎么串起来

基础版（Day 2 入库 + Day 3 问答）：

1. 用 `DirectoryLoader` 和 `TextLoader` 读每个文件夹里的 `.md`，给每篇文档加上 `doc_type`。
2. 用 `RecursiveCharacterTextSplitter` 切块。它按段落、换行、空格逐级往下切，尽量不把句子从中间截断。
3. 选一个嵌入模型，把每个块变成向量。Day 2 和 Day 3 notebook 用 Hugging Face `all-MiniLM-L6-v2`（384 维，本地、免费）。`implementation/` 改成 OpenAI `text-embedding-3-large`。同一个库的入库和查询必须用同一个嵌入模型，维度和空间才对得上。
4. `Chroma.from_documents(...)` 写入本地目录 `vector_db`。已有库就先删掉 collection 再重建，避免新旧向量混在一起。
5. 提问时 `vectorstore.as_retriever()` 把问题编码成向量，取出最相近的若干块。`implementation/answer.py` 里 `RETRIEVAL_K = 10`。
6. 把这些块的正文拼进 system prompt 的 `{context}`。
7. `ChatOpenAI` 带着历史消息和当前问题生成答案。不知道就说不知道。

`app.py` 用 Gradio 把对话和检索到的原文分两栏显示。

进阶版（Day 5，`pro_implementation/`）在检索前后各加一层 LLM，向量库换成 `preprocessed_db`：

1. 不用 LangChain 读文件，自己遍历 `knowledge-base/**/*.md`。
2. 让 LLM 决定怎么切。每个块有三部分：headline（方便被搜到的短标题）、summary（几句摘要）、original_text（原文，不改写）。存进向量库的正文是这三部分拼在一起。提示里要求覆盖全文，块与块大约重叠 25% 或 50 个词。
3. 用 OpenAI embeddings API 自己算向量，用 Chroma 的 `PersistentClient` 写入。
4. 提问时先把用户问题改写成一条更短、更具体的检索问句（query rewriting）。原问题和改写后的问题各检索一次（`RETRIEVAL_K = 20`），合并去重。
5. 再让 LLM 按相关程度重排全部候选（rerank），只留前 `FINAL_K = 10` 段放进 prompt。
6. 生成答案时带上来源路径，方便核对。

## 各天在讲什么

### Day 1：暴力 RAG

不用向量。把员工姓氏、产品名读进一个字典，问题按空格切开，哪个词正好是字典的 key，就把整篇文档塞进 system prompt。`Who is Lancaster and what is carllm?` 能中，换一种说法就中不了。它用来看清 RAG 的最小闭环：检索、拼 prompt、调用 `openai.chat.completions`、用 Gradio 聊天。语义检索要解决的就是这种「必须撞上原词」的限制。

### Day 2：切块、向量、可视化

先用 `tiktoken` 看整库大约多少 token，判断能不能整库塞进上下文。然后 LangChain 加载、切块、嵌入、写入 Chroma。notebook 里大约 76 个文件、413 个块。

可视化不是检索的一步。取出全部向量，用 t-SNE 压到 2 维或 3 维，Plotly 按 `doc_type` 上色。用来看同类文档是否聚在一起。t-SNE 只保证「原来接近的点在图上仍尽量接近」，坐标轴没有业务含义。

### Day 3：检索加生成

LangChain 里两个常用对象：

- Retriever：包住嵌入模型和向量库，`retriever.invoke(question)` 返回文档列表。
- LLM：包住聊天模型，`llm.invoke(messages)` 返回回答。

单独调用 LLM 而不给检索结果，它并不知道 Insurellm 的内部事实。

### Day 4：评估

课程把评估看得比框架选型更重要。换 chunk 大小、嵌入模型、检索条数，都要用同一套测试看数字，而不是凭感觉。

测试集在 `evaluation/tests.jsonl`，一共 150 题。每题有 question、keywords、reference_answer、category。类别大致是：direct_fact 70，temporal 20，spanning 20，comparative 10，numerical 10，relationship 10，holistic 10。keywords 是检索结果里应该出现的词，用来给检索打分，不是让模型背答案。

检索指标（对每个 keyword 算，再取平均）：

- MRR：正确内容第一次出现的名次的倒数。排第 1 得 1，排第 2 得 1/2，没出现得 0。越高说明第一眼就能看到相关块。
- nDCG：相关块是否排在更前面。相关放在后面会打折。
- keyword coverage：这道题的关键词有百分之多少至少出现在某条检索结果里。接近 recall。

答案指标用另一个 LLM 当裁判，对照 reference answer 打 1 到 5 分：

- accuracy：事实对不对。有事实错误就是 1 分。
- completeness：参考答案里的信息有没有说全。
- relevance：是否只回答所问的问题，有没有扯无关内容。

检索好而答案差，问题在 prompt 或生成模型。答案瞎编而检索里根本没有依据，问题在检索或模型没遵守「不知道就说不知道」。

### Day 5：十条进阶手段

notebook 列了这些方向。本周代码真正做了切块实验、文档预处理、query rewriting、reranking。其余是清单，没有在这周的主代码里实现。

1. Chunking：改块大小和重叠，用 Day 4 的测试集比较。
2. Encoder：换嵌入模型，同样用测试集选。
3. Prompt：写清角色、今天的日期、检索到的上下文、对话历史。
4. Document pre-processing：入库前用 LLM 写标题和摘要，和原文一起编码。
5. Query rewriting：把口语问题改成更适合检索的短问句。
6. Query expansion：一个问题展开成多个检索问句。
7. Re-ranking：向量检索先多取一些，再用 LLM 重排，丢掉不太相关的。
8. Hierarchical RAG：先检索摘要，再下钻到原文。
9. Graph RAG：顺着相关文档之间的关系继续取内容。
10. Agentic RAG：让 agent 自己决定查向量库、查 SQL，或使用别的工具。

RAG 的局限也在 Day 4 里写了：它是在 prompt 里补知识，效果靠试验；切块策略会明显改变结果；检索没命中时，模型会突然说不知道；一个听起来合理的架构，测试分数可能很差。

## 库各自干什么

按这周真正用到的来记。

读文件和配置：

- `pathlib`、`glob`：找 `knowledge-base` 下的文件。Day 1 和 pro 版直接用它们读文本。
- `python-dotenv` 的 `load_dotenv`：从 `.env` 读 `OPENAI_API_KEY` 等密钥。
- `tiktoken`：按模型词表数 token。见上面「概念」一节。
- `numpy`：把 Chroma 里的向量收成矩阵，交给 t-SNE。

LangChain，Day 2 和 Day 3、以及 `implementation/` 在用。它把加载、切块、向量库、聊天模型收成统一接口，上手快。模型 API 本身已经很简单时，这层会显得重，部分封装也会过时。Day 5 为了自己控制切块和检索，就不用它了。

- `DirectoryLoader`、`TextLoader`（`langchain_community.document_loaders`）：按目录读文本，得到 `Document`（正文加 metadata）。
- `RecursiveCharacterTextSplitter`（`langchain_text_splitters`）：按字符数递归切块。同包里的 `MarkdownTextSplitter` 在 `ingest.py` 里被注释掉，按 markdown 结构切是备选。
- `HuggingFaceEmbeddings`（`langchain_huggingface`）：本地跑 sentence-transformers。这周用 `all-MiniLM-L6-v2`。
- `OpenAIEmbeddings`（`langchain_openai`）：调用 OpenAI 的嵌入接口。这周的备选和 `implementation/` 的正式选择是 `text-embedding-3-large`。
- `ChatOpenAI`（`langchain_openai`）：调用聊天模型。基础版模型名是 `gpt-4.1-nano`，temperature 0。
- `Chroma`（`langchain_chroma`）：LangChain 对 Chroma 的封装。`from_documents` 负责嵌入并写入，`as_retriever` 负责查询。
- `SystemMessage`、`HumanMessage`、`convert_to_messages`（`langchain_core`）：把 system、历史、当前问题装成 LangChain 消息。

向量库和可视化：

- Chroma：本地向量数据库，数据在 `vector_db/` 或 `preprocessed_db/`。它存储向量和原文，按距离查询。向量要由嵌入模型事先算好。
- `chromadb.PersistentClient`：不经过 LangChain，直接打开同一个库。pro 版用 `collection.add` 写入，用 `collection.query` 查询。
- `sklearn.manifold.TSNE`：把几百维压成 2 维或 3 维，只为画图。
- `plotly.graph_objects`：画可交互散点图。

生成、结构和界面：

- `openai.OpenAI`：官方 SDK。Day 1 用它的 chat completions。pro 版用它的 `embeddings.create`。
- `litellm.completion`：用一个接口调不同供应商的模型。pro 版入库用 `openai/gpt-4.1-nano` 做切块，回答脚本里换成 `groq/openai/gpt-oss-120b`。结构化输出用 `response_format=某个 Pydantic 模型`。
- `pydantic.BaseModel`：规定 LLM 必须返回的 JSON 形状。切块用 `Chunk` / `Chunks`，重排用 `RankOrder`，评估用 `RetrievalEval` / `AnswerEval` / `TestQuestion`。
- `tenacity.retry`：调用失败时按指数退避重试。pro 版用来对付限流。
- `multiprocessing.Pool`：多个文档并行切块。限流时把 `WORKERS` 降到 1。
- `tqdm`：切块进度条。
- `gradio`：聊天界面。Day 1 和 Day 3 用 `ChatInterface`。`app.py` 用 `Blocks`，一边对话，一边显示检索片段和来源。

课程里提到、这周主代码没有接入的向量库：开源的 Qdrant、FAISS（内存），托管的 Pinecone、Weaviate，以及带向量功能的 Postgres、Mongo、Elasticsearch。嵌入模型还提到 word2vec、BERT、Google `gemini-embedding-001`。

## 基础版和进阶版怎么选

基础版适合先跑通：LangChain 切块，一个嵌入模型，Chroma，检索后直接生成。改 `implementation/ingest.py` 里的 `chunk_size`、`chunk_overlap` 和嵌入模型后要重新入库，再跑 `evaluation/eval.py`。

进阶版把「怎么切」和「怎么排」交给 LLM，检索更准，入库和每次提问都更慢、更贵。重排之前先多取（20 条），重排后再砍到 10 条，是为了让向量搜索的漏网段落还有机会被 LLM 排到前面。

查询用的嵌入模型必须和入库时的相同。换模型就要重建向量库。Hugging Face 的 384 维向量和 OpenAI 的高维向量不能混在同一个 collection 里查询。
