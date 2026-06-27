# This week we are start Training and Fine-tuning models

- Training is the core idea of AI
- And Training is the heart of AI
- It involves setting the parameters of a model based on inputs and outputs.
- After training,the model is given new inputs("unseen data") and it will output the correct outputs("predictions").

The ability of a model to make good predictions is unseen data is called "Generalization".

LLMs are remarkably good at generalization.

- Training a multi-billion parameter model from scratch would cost tens to hundreds of millions of dollars.
- Instead, we take advanage of Transfer learning
- We take a pretrained model as base, and use additional training data to fine-tune it for our specific task.

This week is focusing on fine-tuning frontier models.

## The Capstone project - The price is right

Given a description of a product, predict its price.
- The true value of the product (this week and next week's scope)
- This will ultimately be incorporated in an Agentic Solution that will hunt for bargains (week8's scope)
- We'd typically use a Regression model to predict prices, but there are good reasons to try Gen AI
  - Clear and tangible eval
  - Take advantage of an LLM's understanding of language
  - Spoiler alert: the frontier models are already great at this!

### THE ORDER OF PLAY
- Week 6: Data curation, evaluation, front-er models
  - DAY 1: Data curation
  - DAY 2: Data pre-processing
  - DAY 3: Baselines Models and traditional ML
  - DAY 4: Neural Networkds and Frontier LLMs
  - DAY 5: Fine-tuned Frontier LLMs
- Week 7: Fine-tune Open-source LLMs
- Week 8: Deployed model, RAG, Agentic Product

### Finding datasets
- Kaggle
- Hugging Face datasets - recommended
- Google Dataset Search
- My proprieray data

### How will we evaluate performance?
- Model-centric: "Loss"
- Business-centric: Actual difference in price

