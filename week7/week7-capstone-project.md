# Week 7 毕业项目：微调开源模型笔记

这周继续 The Price is Right。Week 6 微调的是 OpenAI 上的 GPT-4.1-nano，权重留在对方那边，之后每次调用仍按他们的 API 计费。这周改为微调一个开源模型：权重下载到 Colab 的 GPU 上，训完的适配器可以自己保存、上传到 Hugging Face。

本地笔记本几乎只有提纲和数据准备。训练和评估的正课在 Colab。

课程按天递进：

1. Day 1：QLoRA 是什么。Colab：https://colab.research.google.com/drive/1oE6jCl8yUxsT1V4N43BD_0Wx5u0A9AQe
2. Day 2：把商品写成训练提示，选定基座模型，把提示数据集推到 Hub。Colab：https://colab.research.google.com/drive/1wO3lNMrMfprlJZF4X9fSsQ8tYC3SRZbh
3. Day 3 和 Day 4：在 Colab 上实际训练。https://colab.research.google.com/drive/1fBTm_jzrFGr88PDOFTQF7JIlQ1JW15BG
4. Day 5：在测试集上评估微调后的开源模型。Colab：https://colab.research.google.com/drive/16e8aY_BlHjzzcR-2dCyDMCPdOQ8XeN1e

对应代码：

- 数据格式仍是 `pricer/items.py`，和 Week 6 同一个 `Item`。
- 打分仍是 `pricer/evaluator.py` 的 `Tester`：平均绝对误差（美元）、MSE、r²。
- 本地天数说明：`day1.ipynb`、`day2.ipynb`、`day3 and 4.ipynb`、`day5.ipynb`

提示数据集：

- https://huggingface.co/datasets/ed-donner/items_prompts_lite
- https://huggingface.co/datasets/ed-donner/items_prompts_full

## 概念

全量微调会更新基座的每一个参数。一个几十亿参数的模型，优化器状态比权重要大好几倍，一张 T4（大约 15GB）放不下。

QLoRA 把两件事叠在一起：

1. 量化：基座权重用 4 bit 存在显存里，前向时再还原成计算用的精度。基座本身在训练中冻结。
2. LoRA：在某些层旁边加一对很小的矩阵。训练只更新这对矩阵。推理时可以把它们合并回原层，也可以分开挂着。

效果是：你得到一个「基座加一小份适配器」。适配器只有几十或几百 MB，上传和下载都便宜。换任务就换适配器，基座可以共用。Week 3 说的 `peft` 就是加载和训练这些适配器的库，`trl` 提供监督微调的训练循环，`accelerate` 和 `bitsandbytes` 负责把 4 bit 模型放上 GPU。

提示和 Week 6 相同。`Item.make_prompt` 把描述包成：

```text
What does this cost to the nearest dollar?

（描述）

Price is $123.00
```

训练集保留完整价格，让模型学习续写。测试用 `test_prompt()`，在 `Price is $` 处截断，模型只生成金额。这样微调和评估的格式一致，解析时只要从字符串里抽出数字。

和 Week 6 的 API 微调相比：

- 开源微调的钱花在 Colab GPU 上，训完之后推理可以自己托管。Week 8 就把这个权重放到 Modal 上。
- API 微调不用管显存和 CUDA，但不能把权重拿下来。
- 两者都用同一套美元误差比较，才能知道开源小模型微调后有没有打过零样本的前沿模型。课程前面的结论是：微调后的开源模型可以超过没微调的前沿模型。

## 各天在做什么

### Day 1

在 Colab 里把 QLoRA 跑通一遍：加载一个因果语言模型，挂上 4 bit 量化和 LoRA，看显存和可训练参数的比例。可训练参数应该远小于总参数。这一天的目的是确认环境，而不是把 80 万条商品训完。

### Day 2

用 Week 6 已经改写好的商品，调用 `make_prompt` 生成带价格的训练文本和不带价格的测试文本。小集和大集分别推到 `items_prompts_lite` 和 `items_prompts_full`。基座选一个能放进 T4 或你租用的 GPU 的指令模型。选太大的基座，量化之后仍然会在加载或激活值上爆显存。

### Day 3 和 Day 4

在 Colab 上用 `SFTTrainer` 一类的训练器跑监督微调。小集用来确认 loss 在下降、生成里能出现 `Price is $`。全量数据训练更久，也更贵。训完把适配器推到自己的 Hugging Face 仓库，记下仓库名，Week 8 的 Modal 服务要拉它。

### Day 5

加载基座加适配器，对测试集调用和 Week 6 相同的 `evaluate`。对比对象是：常数基线、随机森林、Week 6 的神经网络、零样本前沿模型、Week 6 微调过的 GPT-4.1-nano。看的是平均误差多少美元，以及绿、橙、红的比例。

## 库各自干什么

- `transformers`：加载因果语言模型和它自己的 tokenizer。
- `peft`：给模型加上 LoRA，并保存、加载适配器。
- `bitsandbytes`：4 bit 量化，QLoRA 的 Q 来自这里。
- `trl`：监督微调的 trainer，把「提示续写价格」组织成训练步。
- `accelerate`：选择 GPU、混合精度、梯度累积。
- `datasets`：读 `items_prompts_lite` 或 `items_prompts_full`。
- `torch`：训练循环底下的张量和反传。
- Google Colab：提供 T4 或更大的 GPU。本地 notebook 只负责把你指到对应的 Colab。
- `pricer/evaluator.py`：训练之外的业务评分，不看 loss，看美元误差。
