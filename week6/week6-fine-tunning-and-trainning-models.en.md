# Week 6: Fine-Tuning and Training Models

This week starts the capstone, The Price is Right: given a product description, predict its price. The data is product metadata from an Amazon reviews dataset. The predicted price is later handed to the Week 8 bargain agent. Price is a regression problem, and traditional machine learning can do it. The course uses a generative model because the error is measurable in dollars, and the model already understands brands and specs in language.

The days build in this order:

1. Day 1: filter clean products out of the raw catalog and push them to the Hugging Face Hub.
2. Day 2: use an LLM to rewrite messy descriptions into one format.
3. Day 3: set the metrics, then build a constant baseline, a random forest, and XGBoost.
4. Day 4: train a neural network in PyTorch, then see how accurate an untrained frontier model is at zero-shot estimation.
5. Day 5: fine-tune GPT-4.1-nano into a private model that quotes a price.

Code is in `pricer/`:

- Parse raw records: `parser.py`, `loaders.py`
- One product: `items.py`
- Batch rewrite: `preprocessor.py`, `batch.py`
- Scoring: `evaluator.py`
- Neural network: `deep_neural_network.py`

You can use the datasets the instructor already prepared and skip the preprocessing bill:

- Lite: https://huggingface.co/datasets/ed-donner/items_lite
- Full: https://huggingface.co/datasets/ed-donner/items_full

The raw data is at https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## Concepts

Training changes model parameters from inputs and outputs. After training, a useful output on inputs the model has not seen is called generalization. Large language models already generalize well, so zero-shot (no fine-tuning, just ask "what does this cost") is already decent.

Training a model with hundreds of billions of parameters from scratch costs tens to hundreds of millions of dollars. The practical approach is transfer learning: start from a pretrained model and continue training for a short time on your task data. That continuation is fine-tuning. This week fine-tunes a frontier closed model through its API. Week 7 fine-tunes open weights.

One example is an `Item`: title, category, price, plus optional weight, summary, and full description. The prompt sent to the model looks like this:

```text
What does this cost to the nearest dollar?

(rewritten description)

Price is $123.00
```

During training the price stays at the end, so the model learns to continue the description as `Price is $...`. At test time `test_prompt()` cuts the price off and leaves the text through `Price is $`, and you read the number the model generates next.

Evaluation is a business metric, not loss alone. `evaluator.Tester` calls your predictor on a batch of products, parses a number, and compares the absolute error with the true price.

- Mean absolute error, in dollars. On the full dataset the random forest in the course is about $56.40.
- MSE: mean squared error. Large mistakes are amplified.
- r²: how much of the variation in price the predictions explain.
- Color: an error under $40, or under 20% of the true price, is green. Under $80 or 40% is orange. The rest is red.

The default sample is 200 items and 5 threads. If the prediction is a string, the code strips `$` and commas and takes the first number.

## What each day does

### Day 1: cleaning

Read title, description, features, and details from the raw JSON of a category on the Hub. `scrub` joins them into one text. `get_weight` tries to parse a weight. Drop records with no price, an absurd price, or a poor body. Cleaning often moves the final error more than a later hyperparameter search. Then `Item.push_to_hub` pushes train, validation, and test splits.

### Day 2: rewrite into one format

Raw descriptions are long and messy. `Preprocessor` asks an LLM for a short title, category, brand, one sentence of description, and one sentence of details, with no part numbers. The default model is `groq/openai/gpt-oss-20b`.

`Batch` writes products in groups of 1,000 as JSONL, sends them to a batch API, and writes the summaries back onto each `Item` when the batch finishes. The lite set is about 20,000 training rows and costs under $1 to preprocess. The full set is about 800,000 rows and about $30. You can skip the whole step and load the instructor's Hub dataset the next day. `LITE_MODE = True` selects the lite set.

### Day 3: baselines

A weak model is the control. Without it you cannot tell whether the neural network or the LLM is actually better.

- Constant baseline: always guess the training-set mean or median.
- Bag of words plus random forest: the feature is how often each word appears. A decision tree is a stack of ifs. One tree overfits. A random forest builds 100 trees, each on a subset of rows and a subset of words, and averages them. On the full 800,000 rows the instructor's run took about 15 hours and reached a mean error of about $56.40.
- XGBoost: also many trees, built one after another, each correcting the previous tree's error with gradient descent. It is usually faster and generalizes better. If the import fails on a Mac, run `brew install libomp` first. This section can be skipped.

### Day 4: neural network and a zero-shot LLM

`DeepNeuralNetwork` is built in PyTorch. The input is a text vector. The defaults are 10 layers, hidden width 4096, and dropout 0.2, with residual blocks in between. `DeepNeuralNetworkRunner` trains for about 5 epochs, saves `deep_neural_network.pth`, and runs inference on one description. This is for intuition. It is not the final system of the week. Week 8 downloads the weight file and uses it again.

Then, with no training, hand the description to a frontier model and let it quote a price from product knowledge it saw in pretraining. The instructor's conclusion: frontier models are already strong at zero shot. The next day's fine-tune squeezes the error further.

### Day 5: fine-tune GPT-4.1-nano

1. Write the examples as JSONL. Each line is one conversation: the user message is the prompt with the price removed, and the assistant message is `Price is $` plus the number. Upload it to OpenAI.
2. Start a fine-tuning job. The base model is GPT-4.1-nano. OpenAI suggests 50 to 100 examples is enough to try. The instructor used 20,000 and spent about $3.42. For practice, 100 examples is enough.
3. The finished job returns a model name with a suffix. Testing still uses Chat Completions. Only the `model` argument changes to that name. `Tester` compares dollar error against the zero-shot model and the random forest.

After fine-tuning, the model follows the format more reliably: it more often emits only a price, and the prices sit closer to this Amazon catalog. It does not gain facts that were absent from the training data.

## What each library does

- `datasets`: `load_dataset` reads the raw Hub data and the curated `items_lite` and `items_full`. `DatasetDict.push_to_hub` pushes train, validation, and test back.
- `pydantic`: validates `Item` fields, and `model_dump` / `model_validate` convert to and from one Hub row.
- `litellm`: `Preprocessor` calls the Groq model through one interface and accumulates tokens and cost.
- OpenAI Batch and Files APIs: `batch.py` uploads JSONL, submits the batch, and fetches the rewrites. Fine-tuning day uses the Fine-tuning API.
- `torch`: the hand-written deep network, residual blocks, and saving and loading weights.
- `scikit-learn`: the random forest, plus `mean_squared_error` and `r2_score`.
- `xgboost`: the gradient-boosted tree baseline.
- `pandas` and `plotly`: the error scatter plot and the error trend.
- `tqdm` and `ThreadPoolExecutor`: run the 200-item eval in parallel and show progress.
