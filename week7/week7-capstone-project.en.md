# Week 7 Capstone: Fine-Tune an Open-Source Model

This week continues The Price is Right. Week 6 fine-tuned GPT-4.1-nano on OpenAI. The weights stay on their side, and every later call is still billed on their API. This week fine-tunes an open model: the weights are downloaded onto a Colab GPU, and the trained adapter can be saved and uploaded to Hugging Face.

The local notebooks are mostly an outline and data preparation. The training and evaluation labs are in Colab.

The days build in this order:

1. Day 1: what QLoRA is. Colab: https://colab.research.google.com/drive/1oE6jCl8yUxsT1V4N43BD_0Wx5u0A9AQe
2. Day 2: turn products into training prompts, pick a base model, and push the prompt dataset to the Hub. Colab: https://colab.research.google.com/drive/1wO3lNMrMfprlJZF4X9fSsQ8tYC3SRZbh
3. Day 3 and Day 4: train on Colab. https://colab.research.google.com/drive/1fBTm_jzrFGr88PDOFTQF7JIlQ1JW15BG
4. Day 5: evaluate the fine-tuned open model on the test set. Colab: https://colab.research.google.com/drive/16e8aY_BlHjzzcR-2dCyDMCPdOQ8XeN1e

Code for this week:

- The record format is still `pricer/items.py`, the same `Item` as Week 6.
- Scoring is still `Tester` in `pricer/evaluator.py`: mean absolute error in dollars, MSE, and r².
- Local day notes: `day1.ipynb`, `day2.ipynb`, `day3 and 4.ipynb`, `day5.ipynb`

Prompt datasets:

- https://huggingface.co/datasets/ed-donner/items_prompts_lite
- https://huggingface.co/datasets/ed-donner/items_prompts_full

## Concepts

Full fine-tuning updates every parameter of the base model. For a model with several billion parameters, optimizer state is several times larger than the weights. A T4, about 15GB, cannot hold that.

QLoRA stacks two ideas:

1. Quantization: the base weights sit in GPU memory at 4 bits, and are restored to the compute dtype for the forward pass. The base stays frozen during training.
2. LoRA: a pair of small matrices is attached beside some layers. Training updates only that pair. At inference you can merge them back into the original layer, or keep them separate.

The result is a base model plus a small adapter. The adapter is tens or hundreds of megabytes, so upload and download stay cheap. A new task means a new adapter. The base can be shared. `peft` from Week 3 loads and trains these adapters. `trl` provides the supervised fine-tuning loop. `accelerate` and `bitsandbytes` put the 4-bit model on the GPU.

The prompt matches Week 6. `Item.make_prompt` wraps the description as:

```text
What does this cost to the nearest dollar?

(description)

Price is $123.00
```

The training set keeps the full price so the model learns to continue the text. The test uses `test_prompt()`, which truncates at `Price is $` and lets the model generate only the amount. Fine-tuning and evaluation then share one format, and parsing only has to pull a number out of the string.

Compared with the Week 6 API fine-tune:

- Open fine-tuning spends money on Colab GPU time. After training, inference can be hosted by you. Week 8 puts these weights on Modal.
- API fine-tuning does not require you to manage VRAM or CUDA, and you cannot download the weights.
- Both are compared with the same dollar error, which is how you tell whether the fine-tuned small open model beats a zero-shot frontier model. The course's earlier conclusion is that the fine-tuned open model can beat a frontier model that was not fine-tuned.

## What each day does

### Day 1

Get QLoRA running once in Colab: load a causal language model, attach 4-bit quantization and LoRA, and look at memory and the ratio of trainable parameters. Trainable parameters should be far smaller than the total. The goal of the day is to confirm the environment, not to finish 800,000 products.

### Day 2

Take the products Week 6 already rewrote. `make_prompt` builds the training text with the price and the test text without it. Push the lite and full sets to `items_prompts_lite` and `items_prompts_full`. Pick an instruction base model that fits a T4 or the GPU you rented. A base that is too large still runs out of memory on load or on activations after quantization.

### Day 3 and Day 4

Run supervised fine-tuning on Colab with a trainer in the `SFTTrainer` family. The lite set checks that loss falls and that generations contain `Price is $`. The full set takes longer and costs more. Push the adapter to your own Hugging Face repo and keep the repo name. The Week 8 Modal service will pull it.

### Day 5

Load the base plus the adapter and call the same `evaluate` as Week 6 on the test set. Compare against the constant baseline, the random forest, the Week 6 neural network, a zero-shot frontier model, and the Week 6 fine-tuned GPT-4.1-nano. Read the mean error in dollars, and the share of green, orange, and red.

## What each library does

- `transformers`: loads the causal language model and its tokenizer.
- `peft`: attaches LoRA, and saves and loads the adapter.
- `bitsandbytes`: 4-bit quantization. The Q in QLoRA comes from here.
- `trl`: the supervised fine-tuning trainer, which turns "continue the prompt with a price" into training steps.
- `accelerate`: GPU selection, mixed precision, and gradient accumulation.
- `datasets`: reads `items_prompts_lite` or `items_prompts_full`.
- `torch`: the tensors and backprop under the training loop.
- Google Colab: the T4 or a larger GPU. The local notebooks only point you at the matching Colab.
- `pricer/evaluator.py`: the business score outside training. It reads dollar error, not loss.
