# Week 3: Hugging Face, Open-Source Models, and Google Colab

This week moves from calling someone else's API to running open weights directly on a GPU. Most of the lab notebooks live in Google Colab. Local `day2` through `day5` are mainly entry points and notes. What you can read locally is how the libraries split, which Colab machine to use, and the token-by-token prediction demo in `visualizer.py`.

The days build in this order:

1. Day 1: the Hugging Face website, the core libraries, and what Colab is for.
2. Day 2: the high-level transformers interface, pipelines. Colab: https://colab.research.google.com/drive/1aMaEw8A56xs0bRM4lu8z7ou18jqyybGm
3. Day 3: tokenizers. Each model has its own tokenizer. Colab: https://colab.research.google.com/drive/1WD6Y2N7ctQi1X9wa6rpkg8UfyA4iSVuz
4. Day 4: the model object in transformers, meaning the weights and the forward pass. Colab: https://colab.research.google.com/drive/1hhR9Z-yiqjUe7pJjVQw4c74z_V3VchLy
5. Day 5: watch how the next token is chosen at inference time, then build a meeting-minutes generator. Colab: https://colab.research.google.com/drive/1KSMxOCprsl1QRpt_Rq0UqCAyMtPqDQYx

Code for this week:

- Local notes: `day1.ipynb` through `day5.ipynb`
- Per-token probabilities: `visualizer.py`

The Colab labs run on a free or low-cost T4. A T4 has about 15GB of GPU memory. A paid A100 has about 40GB. CPU-only also works, and it is slow.

## Concepts

The Hugging Face Hub is a store of models and datasets, for download and upload. Code on your machine or in Colab pulls those files through a library. The weights, which can be hundreds of gigabytes, do not go into git.

transformers has two levels:

- High level: `pipeline`. Write `my_pipeline = pipeline("task name")`, then `result = my_pipeline(input)`. Sentiment, summarization, and generation are enough at this level.
- Low level: take the tokenizer that belongs to this model, take the model itself, and do tokenize, forward, and decode yourself. Day 3 and Day 4 split those steps apart.

A tokenizer is tied to one model. Each model is released with its own vocabulary. Feeding model B's weights with model A's tokenizer misaligns the ids, and the output is meaningless. Week 1's tiktoken is the vocabulary family for OpenAI models. A Hugging Face model uses the tokenizer that `transformers` downloads with that model.

At inference the model still predicts the next token. `TokenPredictor` in `visualizer.py` spreads that out:

1. Call `chat.completions.create` with `temperature=0` and `seed=42` so the choice stays as fixed as the API allows.
2. `logprobs=True` and `top_logprobs=3` ask the API for the highest-probability tokens at this step, with their log probabilities.
3. `stream=True` records each token as it arrives.
4. networkx and matplotlib draw the chosen token and the other two candidates from that step.

The meeting-minutes section connects an open speech model to a text model: upload meeting audio (the notebook attaches Google Drive), transcribe it, then turn the transcript into minutes. You supply the audio.

Three training libraries are named this week and used later. This week only needs the names:

- `peft`: parameter-efficient fine-tuning. Train a small set of new weights, and leave the original model frozen. Week 7's QLoRA is in this family.
- `trl`: Transformer Reinforcement Learning. It organizes language-model training, including supervised fine-tuning and preference training.
- `accelerate`: runs the same training code on GPUs, and handles device placement and mixed precision.

## How to use each day

### Day 1: account and runtime

1. Create an account at huggingface.co. Later Colab labs store the HF token as a notebook secret so they can pull gated models.
2. Open the course link at colab.research.google.com. The free tier is usually enough.
3. When you pick a GPU runtime, prefer a T4.

Colab runs a notebook on a machine with a GPU, shares that notebook, and connects to Google services such as Drive.

### Day 2: pipeline

Create a pipeline from a task name, pass a sentence or a paragraph, and get a structured result. The end of the Colab lists every task name. This layer is how you check that an open model can do the job before you drop down to the tokenizer and the model.

### Day 3 and Day 4: tokenizer and model

Day 3 compares how different models cut the same sentence. Different vocabularies produce different token counts and different split points. Day 4 loads the weights and runs one forward pass on a T4. A pipeline is those two steps packed together.

### Day 5: see the sampling, then make minutes

Run local `visualizer.py` first. It uses the OpenAI API so the logprobs are visible, and it builds the picture that generation is repeated next-token choice. Then, in Colab, turn audio into meeting minutes. The minutes step is still: put the transcript in the prompt, and let the model write a summary.

## What each library does

- `huggingface_hub`: downloads and uploads models and datasets, and handles tokens and repos.
- `datasets`: loads a dataset by its Hub name. Week 6's Amazon products use it.
- `transformers`: models, tokenizers, and `pipeline`.
- `peft`: fine-tuning that trains a small number of added parameters.
- `trl`: the language-model training loop.
- `accelerate`: multi-GPU and mixed precision.
- Google Colab: cloud Jupyter with a GPU. Free T4, paid A100.
- `openai`: `visualizer.py` uses it to stream tokens and top logprobs.
- `networkx` and `matplotlib`: draw the candidate tokens.
