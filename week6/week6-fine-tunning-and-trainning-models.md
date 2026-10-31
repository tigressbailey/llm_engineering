# Week 6 微调与训练模型笔记

这周开始毕业项目 The Price is Right：给一段商品描述，预测它的价格。数据来自亚马逊评论数据的商品元信息。预测出的价格后面会交给 Week 8 的砍价 agent。价格是回归问题，用传统机器学习就能做。这里改用生成模型，是因为对错可以用美元衡量，而且模型已经懂语言里的品牌和规格。

课程按天递进：

1. Day 1：从原始目录里筛出干净的商品，推到 Hugging Face Hub。
2. Day 2：用 LLM 把杂乱描述改写成统一格式。
3. Day 3：定评价指标，先做常数基线、随机森林和 XGBoost。
4. Day 4：用 PyTorch 训一个自己的神经网络，再看没训练过的前沿模型零样本能估多准。
5. Day 5：把 GPT-4.1-nano 微调成只会报价的私有模型。

对应代码在 `pricer/`：

- 原始记录解析：`parser.py`、`loaders.py`
- 一条商品：`items.py`
- 批量改写：`preprocessor.py`、`batch.py`
- 打分：`evaluator.py`
- 神经网络：`deep_neural_network.py`

数据集可以直接用课程作者已经处理好的版本，不必自己付预处理费用：

- 小集：https://huggingface.co/datasets/ed-donner/items_lite
- 全量：https://huggingface.co/datasets/ed-donner/items_full

原始数据在 https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## 概念

训练是用输入和输出去改模型参数。训练之后，模型在没见过的输入上仍给出靠谱输出，这叫泛化。大模型本身就很能泛化，所以零样本（不微调，直接问「这个多少钱」）已经不差。

从零训练一个千亿参数模型要数千万到数亿美元。实际做法是迁移学习：拿一个已经预训练好的模型，再用你的任务数据继续训练一小段，这就是微调。这周微调的是前沿闭源模型的 API。Week 7 改成微调开源权重。

一条样本是 `Item`：标题、类目、价格，加上可选的重量、摘要、完整描述。发给模型的提示长这样：

```text
What does this cost to the nearest dollar?

（改写后的描述）

Price is $123.00
```

训练时价格留在末尾，让模型学习把描述续写成 `Price is $...`。测试时用 `test_prompt()` 把价格切掉，只留到 `Price is $`，看模型接着生成的数字。

评估是业务指标，不是只看 loss。`evaluator.Tester` 对一批商品调用你的预测函数，解析出数字，和真实价格比绝对误差。

- 平均绝对误差，单位是美元。课程里全量随机森林大约 56.40 美元。
- MSE：平方误差的平均，大错会被放大。
- r²：预测对价格变异的解释比例。
- 颜色：误差小于 40 美元，或小于真实价格的 20%，算绿。小于 80 美元或 40% 算橙。其余算红。

默认抽 200 条、5 个线程。预测结果如果是字符串，会去掉 `$` 和逗号再抽第一个数。

## 各天在做什么

### Day 1：清洗

从 Hub 上某个类目的原始 JSON 里读出标题、描述、特征和详情。`scrub` 拼成一段文字，`get_weight` 尽量解析重量。丢掉没有价格、价格离谱或正文太差的记录。清洗对最终误差的影响，常常大于后面调几个超参。做好后用 `Item.push_to_hub` 分成 train、validation、test 推上去。

### Day 2：改写成统一格式

原始描述又长又乱。`Preprocessor` 让 LLM 只输出四行：短标题、类目、品牌、一句话描述、一句话特性，不要零件号。默认模型是 `groq/openai/gpt-oss-20b`。

`Batch` 把商品按 1000 条一组写成 JSONL，交给批处理接口，完成后再把摘要写回 `Item`。小集大约 2 万条训练数据，预处理不到 1 美元。全量大约 80 万条，大约 30 美元。可以整段跳过，明天直接加载作者的 Hub 数据集。`LITE_MODE = True` 走小集。

### Day 3：基线

先要有一个弱模型当对照，否则不知道神经网络或 LLM 是否真的更好。

- 常数基线：永远猜训练集的平均价或中位数。
- 词袋加随机森林：每个词出现的次数是特征。决策树像一层层 if。单棵树容易过拟合。随机森林做 100 棵树，每棵看一部分样本和一部分词，最后取平均。全量 80 万条作者跑了约 15 小时，平均误差约 56.40 美元。
- XGBoost：也是很多棵树，但是一棵接一棵，后一棵去补前一棵的误差（梯度下降）。通常更快，也更不容易过拟合。Mac 上如果 import 失败，先 `brew install libomp`。这一节可以跳过。

### Day 4：神经网络和零样本 LLM

`DeepNeuralNetwork` 用 PyTorch 搭：输入是文本向量，默认 10 层、隐藏宽度 4096、dropout 0.2，中间是残差块。`DeepNeuralNetworkRunner` 负责训练约 5 个 epoch、存 `deep_neural_network.pth`、再对一条描述做推理。这是为了建立直觉，不是这周的最终系统。权重文件 Week 8 还会下载来用。

然后不训练，直接把描述交给前沿模型，让它凭预训练时见过的商品知识报价。作者的结论是：前沿模型零样本已经很强。第二天的微调是在这个基础上再挤误差。

### Day 5：微调 GPT-4.1-nano

1. 把样本写成 JSONL。每行是一次对话：user 是去掉价格的提示，assistant 是 `Price is $数字`。上传到 OpenAI。
2. 在微调接口上开一个任务，基座是 GPT-4.1-nano。OpenAI 建议 50 到 100 条就够试。作者用了 2 万条，花费大约 3.42 美元。自己练习用 100 条即可。
3. 微调完成后会得到一个带后缀的模型名。测试时仍走 Chat Completions，只是 `model` 换成这个名字。`Tester` 用同一套美元误差和零样本、随机森林对比。

微调之后的模型更听话：它更稳定地只吐出一个价格，并且更贴近这批亚马逊商品的价格分布。它不会因此获得训练数据里没有的新事实。

## 库各自干什么

- `datasets`：`load_dataset` 读 Hub 上的原始数据和整理后的 `items_lite`、`items_full`。`DatasetDict.push_to_hub` 把 train、validation、test 推回去。
- `pydantic`：`Item` 的字段校验，以及 `model_dump` / `model_validate` 和 Hub 上的一行记录互转。
- `litellm`：`Preprocessor` 用一个接口调用 Groq 上的模型，并累计 token 和费用。
- OpenAI Batch 和 Files API：`batch.py` 上传 JSONL、提交批任务、取回改写结果。微调那一天再用 Fine-tuning API。
- `torch`：自己写的深度网络、残差块、保存和加载权重。
- `scikit-learn`：随机森林，以及 `mean_squared_error`、`r2_score`。
- `xgboost`：梯度提升树基线。
- `pandas`、`plotly`：误差散点图和误差趋势。
- `tqdm`、`ThreadPoolExecutor`：评估时并行跑 200 条并显示进度。
