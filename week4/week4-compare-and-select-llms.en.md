# Week 4: Compare and Select LLMs

This week is about choosing a model for a business task, and making that choice with an experiment you can time. The experiment asks a model to rewrite a piece of Python into faster C++ (Day 5 rewrites it into Rust), then compiles and runs it, and compares the speedup.

The days build in this order:

1. Day 1: compare features and cost, and look at scaling laws and the limits of benchmarks.
2. Day 2: five leaderboards worth using, and blind arenas.
3. Day 3: the five steps of an AI engineer, and Python-to-C++ with frontier closed models.
4. Day 4: the same task with open models (Ollama, Groq, OpenRouter).
5. Day 5: change the target language to Rust, and separate technical metrics from business metrics.

Code for this week:

- Labs: `day1.ipynb` through `day5.ipynb`
- Tell the model which compilers this machine has: `system_info.py`
- Gradio styling: `styles.py`

## Concepts

Look at two columns when you choose a model.

Capability:

- Open or closed. Closed models are reached through an API. Open models can run locally or on hosted inference such as Groq.
- Chat, reasoning, or a hybrid. A reasoning model spends more tokens on internal steps.
- Release date and knowledge cutoff. Facts after the cutoff were not in the training data.
- Parameter count and number of training tokens.
- Context window length.

Cost:

- Whether the license allows commercial use.
- Inference cost: API price per token, or electricity and rental for your own GPU.
- Training or fine-tuning cost.
- How much glue code the product needs, and how long until it can ship.
- Rate limits, generation speed, and time to first token.

The Chinchilla scaling law (DeepMind, 2022) says that, for a fixed compute budget, parameter count and training tokens should grow together. Growing only the model, or only the data, wastes compute. The rule of thumb is about 20 training tokens per parameter. GPT-3 has 175 billion parameters and relatively less data. Chinchilla uses about 70 billion parameters and more data, and it beat GPT-3. The law is a rule for training a base model efficiently. It is not a knob you turn when you call an API.

The course names six hard benchmarks that still separate strong models:

- HLE (Humanity's Last Exam): very hard questions written by experts across fields.
- GPQA: graduate-level science questions whose answers are not sitting on a search engine.
- MMLU-Pro: a harder multi-subject multiple-choice exam.
- AIME: a hard math contest.
- LiveCodeBench: programming problems that keep being added, so memorized test sets matter less.
- MuSR: multi-step soft reasoning, such as puzzles and scenarios.

A benchmark score is not enough to choose a model. Common problems:

- Training-data contamination: the questions and answers may already be in the training set.
- Prompts, temperature, and eval scripts differ across labs, so scores are hard to compare.
- One exam covers one skill.
- Questions with a single correct answer do not measure tone, judgment, or whether the business actually improved.
- Saturation: the top models all sit near a perfect score, and the board stops separating them.
- Overfitting the leaderboard: the model is tuned for this exam, and real tasks get worse.
- A model may recognize that it is being evaluated.

Human blind tests fill a different gap. An arena (LM Arena, often written arena.ai) shows a person two anonymous answers and asks which is better, then ranks models with an Elo-style score. That measures preference. An automatic exam measures something else.

Leaderboards are a candidate list. They are not the final decision:

- Artificial Analysis: https://artificialanalysis.ai/ The overview chart is Intelligence Index against cost per task. The coding chart is Coding Agent Index against cost per task. Prefer models that land in the region that is both strong and cheap. Eval notes: https://artificialanalysis.ai/evaluations
- Vellum: https://www.vellum.ai/llm-leaderboard Context length and API price are on the same page.
- Scale SEAL: https://labs.scale.com/leaderboard
- Hugging Face Open LLM Leaderboard: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard Open models.
- Live Bench: https://live-bench.ai/ Questions keep changing, which reduces contamination.

An AI engineer does two jobs: understand the data and the problem (data science), and build the system (software engineering). The emphasis can fall three ways: make the system faster (systems), make the model more accurate (model tuning), or build a product on top (applications).

Five steps for choosing:

1. State the business problem and the number that means success. A chat reply that happens to contain a sales figure is not yet a business problem.
2. Prepare: use the boards above to shortlist candidates. Check window, price, license, and knowledge cutoff.
3. Select: build a small prototype and compare models on your task. The leaderboard is the shortlist.
4. Customize: RAG in Week 5, fine-tuning in Weeks 6 and 7, agents in Week 8.
5. Productize: deploy, rate-limit, monitor, and manage keys.

Evaluation has two layers:

- Model metrics: loss, perplexity, accuracy, precision, AUC. These are easier to optimize and farther from the business.
- Business metrics: in this week's experiment, seconds to finish the same computation, and the speedup. Also ROI, time saved, and whether customers are satisfied.

## Experiment: rewrite Python into faster local code

The task is to have a model turn a piece of Python into high-performance C++, then compile and run it on your machine. Compiling is optional. You can also paste the code into https://www.programiz.com/cpp-programming/online-compiler/ and see a rough difference. `system_info.py` collects the operating system and compilers and puts them in the prompt, so the generated code matches this machine.

Day 3 uses the more expensive closed models: GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, and Grok 4. To spend less, switch to something in the gpt-5-nano class. Day 4 switches to open models: local Ollama, plus hosted open models on Groq and OpenRouter.

One recorded C++ speedup against the original Python, from Ed's experiment:

- Gemini 2.5 Pro: 1440x
- Grok 4: 1060x
- gpt-oss 20B: 238x
- GPT-5: 233x
- Claude Sonnet 4.5: 184x
- Qwen3 Coder 30B: 168x
- DeepSeek Coder v2: 168x
- gpt-oss 120B: 14x
- Qwen 2.5 Coder: compile or run failed

Day 5 changes the target to Rust, and the ranking changes. Several closed models fail. gpt-oss 120B becomes the fastest tier (recorded time about 0.000304), then Grok 4, then gpt-oss 20B. The same models, a different target language, and the order can flip. That is why the prototype has to run on your task.

In products, model use falls into about three patterns: a chat layer around someone else's product (language learning, coding assistants), a vertical product (law, hiring, health, data platforms), and an agent that calls tools (coding agents, general agents).

## What each library does

This week adds almost no new model library. It reuses the Week 2 clients.

- The `openai` client, plus Groq, OpenRouter, and Ollama compatible endpoints: the same rewrite task sent to different models.
- `subprocess` and `system_info.py`: detect local compilers, and compile and time the result when you want to.
- `gradio` and `styles.py`: styling, if you turn the comparison into a small UI.
- The leaderboard sites are not libraries. They produce the candidate list. The timed experiment makes the final choice.
