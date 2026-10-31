# Week 3 Hugging Face、开源模型与 Google Colab 笔记

这周从「调用别人的 API」转到「在 GPU 上直接跑开源权重」。正课笔记本大多在 Google Colab 里，本地的 `day2` 到 `day5` 主要是入口和说明。本地能直接看的是库的分工、Colab 的机型，以及 `visualizer.py` 里「一个 token 一个 token 预测」的演示。

课程按天递进：

1. Day 1：Hugging Face 网站和几个核心库，以及 Colab 用来干什么。
2. Day 2：transformers 的高层接口 pipelines。Colab：https://colab.research.google.com/drive/1aMaEw8A56xs0bRM4lu8z7ou18jqyybGm
3. Day 3：tokenizer。一个模型配自己的 tokenizer。Colab：https://colab.research.google.com/drive/1WD6Y2N7ctQi1X9wa6rpkg8UfyA4iSVuz
4. Day 4：transformers 里的模型对象，也就是权重和前向计算。Colab：https://colab.research.google.com/drive/1hhR9Z-yiqjUe7pJjVQw4c74z_V3VchLy
5. Day 5：先看推理时下一个 token 怎么选出来，再做一个会议纪要生成器。Colab：https://colab.research.google.com/drive/1KSMxOCprsl1QRpt_Rq0UqCAyMtPqDQYx

对应代码：

- 本地说明：`day1.ipynb` 到 `day5.ipynb`
- 逐 token 看概率：`visualizer.py`

Colab 上的课用免费或低价的 T4 就能跑。T4 大约 15GB 显存。付费的 A100 大约 40GB。也可以只用 CPU，只是慢。

## 概念

Hugging Face Hub 是模型和数据集的仓库，用来下载和上传。本地或 Colab 里的代码通过库去拉这些文件，而不是把几百 GB 的权重交进 git。

transformers 有两层用法：

- 高层：`pipeline`。写成 `my_pipeline = pipeline("任务名")`，然后 `result = my_pipeline(输入)`。情感分类、摘要、生成这类标准推理用它就够。
- 底层：先拿出这个模型专属的 tokenizer，再拿出模型本身，自己做 tokenize、前向、decode。Day 3 和 Day 4 就是拆开这两步。

Tokenizer 不属于「所有模型共用的一把刀」。每个模型发布时带着自己的词表。用 A 模型的 tokenizer 去喂 B 模型的权重，id 对不上，输出没有意义。Week 1 的 tiktoken 是 OpenAI 模型那一套词表。Hugging Face 模型用的是 `transformers` 里随模型下载的 tokenizer。

推理时模型做的事仍然是预测下一个 token。`visualizer.py` 里的 `TokenPredictor` 把这件事摊开：

1. 用 `chat.completions.create`，`temperature=0`，`seed=42`，让选择尽量固定。
2. `logprobs=True` 和 `top_logprobs=3` 让接口返回这一步概率最高的几个 token，以及它们的对数概率。
3. `stream=True`，每来一个 token 就记下来。
4. 用 networkx 和 matplotlib 把「当前 token，以及当时另外两个候选」画出来。

会议纪要那一节把开源语音模型和文本模型接在一起：上传一段会议音频（笔记本会连 Google Drive），转写，再整理成纪要。音频要自己准备。

后面几周还会用到这周点名的三个训练库，这周只建立名字：

- `peft`：参数高效微调。只训练新加的一小部分权重，不全量改原模型。Week 7 的 QLoRA 属于这一类。
- `trl`：Transformer Reinforcement Learning，用来组织语言模型的训练，包括监督微调和偏好训练。
- `accelerate`：把同一套训练代码分布到 GPU 上，处理设备和混合精度。

## 各天怎么用

### Day 1：账号和运行环境

1. 在 huggingface.co 准备账号。后面 Colab 要把 HF token 存成 notebook secret，拉受限制的模型时要用。
2. 在 colab.research.google.com 打开课程链接。免费档通常够用。
3. 运行时选 GPU 时，优先 T4。

Colab 的用途：在一块带 GPU 的机器上跑 notebook，和别人共享这份 notebook，并接上 Google Drive 这类服务。

### Day 2：pipeline

选一个任务名创建 pipeline，把句子或一段文丢进去，拿结构化结果。Colab 末尾列了全部任务名。这一层适合先确认「这个开源模型能不能做这件事」，再决定要不要下到 tokenizer 和模型。

### Day 3 和 Day 4：tokenizer 与模型

Day 3 对比不同模型的切词结果：同一句话，词表不同，token 个数和切分边界就不同。Day 4 加载模型权重，在 T4 上做一次前向。pipeline 内部就是这两步的打包。

### Day 5：看见采样，再做会议纪要

先跑本地的 `visualizer.py`（它走的是 OpenAI API，用来把 logprobs 显示出来），建立「生成就是反复选下一个 token」的图景。再到 Colab 把音频变成会议纪要。纪要本身仍是：转写文本放进 prompt，模型生成摘要。

## 库各自干什么

- `huggingface_hub`：下载、上传模型和数据集，处理 token 和仓库。
- `datasets`：按 Hub 上的名字加载数据集，后面 Week 6 的亚马逊商品用的就是它。
- `transformers`：模型、tokenizer、`pipeline`。
- `peft`：只训练少量新增参数的微调。
- `trl`：语言模型的训练循环。
- `accelerate`：多 GPU 和混合精度。
- Google Colab：带 GPU 的云端 Jupyter。免费 T4，付费可以上 A100。
- `openai`：`visualizer.py` 用来拉流式 token 和 top logprobs。
- `networkx`、`matplotlib`：把候选 token 画成图。
