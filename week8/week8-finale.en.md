# Week 8 Finale: Deployment, RAG, and the Bargain Agent

This week packs the previous seven weeks into a system that finds bargains on its own. It scans deal RSS feeds, estimates a true price, pushes a notification to a phone when the discount is large enough, and shows logs and a results table in Gradio.

The price estimate is a weighted mix of three specialists:

- Specialist: the open model fine-tuned in Week 7, deployed on Modal.
- Frontier: GPT plus the prices of similar products in Chroma, which is RAG.
- Neural network: the Week 6 PyTorch weights, inference only.

The days build in this order:

1. Day 1: deploy the fine-tuned model on Modal and wrap it as SpecialistAgent.
2. Day 2: turn the training set into a vector store, and add FrontierAgent and EnsembleAgent.
3. Day 3: ScannerAgent reads RSS. MessagingAgent pushes with Pushover.
4. Day 4: PlanningAgent wires the pieces together. Another version lets the model decide when to call the three tools.
5. Day 5: the Gradio UI, `price_is_right.py`.

Code for this week:

- Remote services: `hello.py`, `llama.py`, `pricer_ephemeral.py`, `pricer_service.py`, `pricer_service2.py`
- Orchestration: `deal_agent_framework.py`
- UI: `price_is_right.py`, `log_utils.py`
- Agents, under `agents/`: `specialist_agent.py`, `frontier_agent.py`, `neural_network_agent.py`, `ensemble_agent.py`, `scanner_agent.py`, `messaging_agent.py`, `planning_agent.py`, `autonomous_planning_agent.py`
- Deals already surfaced are stored in `memory.json`
- The vector store directory is `products_vectorstore`

## Concepts

Modal runs a Python function inside their GPU container. A local call to `pricer.price.remote(description)` does not load the large model on your machine. The first image build can take about 10 minutes. After that, a cold start is about 30 seconds, and a container that is still awake answers in about 2 seconds. `MIN_CONTAINERS = 1` keeps one container up so latency stays near 2 seconds, and it keeps spending credits. You can also set `scaledown_window` to 1200 seconds and keep it warm for 20 minutes.

Two kinds of deployment:

- Ephemeral app: when the notebook stops, the container goes away. Used to try `hello` and the first Llama load.
- `uv run modal deploy`: a deployed app you can call later. `SpecialistAgent` connects with `modal.Cls.from_name("pricer-service", "Pricer")`.

The Hugging Face token in Modal is a secret named `huggingface-secret`, with key `HF_TOKEN`. If the code says `hf-secret`, match it to the name in the first column of the dashboard. You can pin a region, for example `@app.function(region="eu")`, at a slightly higher cost.

The three pricers see the same rewritten description:

1. Specialist sends the description to Modal and returns the fine-tuned model's price.
2. Frontier encodes the description to 384 dimensions with `all-MiniLM-L6-v2`, fetches the 5 nearest training products and their true prices from Chroma, puts them in the prompt, and asks `gpt-5.1` for a single number. This is RAG at inference time. There is no extra training.
3. The neural network loads Week 6's `deep_neural_network.pth` and runs one local forward pass. The weights are in https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK

`EnsembleAgent` combines them with fixed weights. It does not train another regression:

```text
0.8 * frontier + 0.1 * specialist + 0.1 * neural_network
```

The frontier model plus RAG carries most of the weight because the zero-shot frontier model was already strong in Week 6, and the true prices of similar products add a concrete price level. The fine-tune and the neural network supply two estimates with different bias.

A scan produces a `Deal`: description, listed price, and URL. Estimate minus listed price is `discount`. An `Opportunity` holds all three. `PlanningAgent` keeps the best deal whose discount is over $50 and sends the notification.

Memory is a list of URLs in `memory.json`. The next scan passes links that were already notified to the scanner, so they are not pushed again. `reset_memory` keeps only the first two entries in the file, for the demo.

There are two planners:

- `PlanningAgent`: the order is fixed in code. Scan, price the first 5 deals, sort by discount, notify when the threshold is crossed.
- `AutonomousPlanningAgent`: registers `scan_the_internet_for_bargains`, `estimate_true_value`, and `notify_user_of_deal` as tools. The model decides which to call and how many times. This is the Week 2 tool-calling pattern in full.

## What each day does

### Day 1: Modal and Specialist

1. Create a token at modal.com and run `uv run modal token set --token-id ... --token-secret ...` locally. On Windows, if the virtual environment cannot see `.modal.toml` in the user profile, move that file into the lab directory, or put `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`.
2. `hello.py` checks that a function runs in the cloud, including the version pinned to a European region.
3. Create a secret in the Modal dashboard: name `huggingface-secret`, key `HF_TOKEN`.
4. Deploy the pricer service. The preprocessor can default to Llama. To switch to Groq, set `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b` in `.env`.
5. `SpecialistAgent.price` is a remote call and nothing more.

### Day 2: vector store and ensemble

Encode the training set with sentence-transformers. The full set is about 800,000 rows and took the instructor about 30 minutes on a GPU. Use lite if that is too long. The Chroma collection is named `products`, and the metadata includes the price. t-SNE plus Plotly draws the points again, colored by category, the same check as in Week 5.

`FrontierAgent.find_similars` takes 5 hits. `make_context` pastes "similar product plus price" into the user message. `EnsembleAgent` first uses `Preprocessor` to turn messy RSS text into the Week 6 description, then asks the three specialists.

### Day 3: scan and notify

`ScannerAgent` pulls deal RSS feeds, uses a smaller GPT to turn each item into a product description and a price, and skips URLs already in `memory`. The output is forced into JSON with the Pydantic model `DealSelection`.

Pushover: sign up at https://pushover.net/, create an application, and write the user key and API token into `.env` as `PUSHOVER_USER` and `PUSHOVER_TOKEN`. Install the client on the phone. `MessagingAgent` has a model write a short notification, then POSTs it to Pushover.

### Day 4: planning

Walk the tool JSON with fake functions first, then replace them with the real scanner, ensemble, and messenger. `DealAgentFramework.run` calls `planner.plan(memory)` and, when there is a result, appends it to `memory.json`.

The fixed workflow uses `DEAL_THRESHOLD = 50`. In the autonomous version the model decides whether to notify. The tool implementations still scan, estimate, and push.

### Day 5: UI

`price_is_right.py` uses Gradio Blocks. One panel scrolls the logs. One table lists the product, listed price, estimate, discount, and link. Logs are pulled from a logging queue, so `self.log` inside an agent shows up on the page. Running the file loops on `DealAgentFramework.run`.

In the instructor's other repo, tech2ai, segment 4 replaces the autonomous planner with the OpenAI Agents SDK and connects MCP servers. That is an extension after this course, not the code this week submits.

## What each library does

- `modal`: builds the image, declares the GPU function, reads the Hugging Face secret, and, after `deploy`, calls it remotely with `Cls.from_name`.
- `huggingface_hub` and the Modal secret: the container downloads the Week 7 base and adapter on startup.
- `sentence_transformers`: `all-MiniLM-L6-v2`, turning a product description into a 384-dimensional vector locally.
- `chromadb`: `PersistentClient` stored in `products_vectorstore`, collection name `products`. A query returns documents and the price in metadata.
- `openai`: FrontierAgent's `gpt-5.1`, plus the smaller models in the scanner and messenger.
- `torch`: loads `deep_neural_network.pth` for inference.
- `litellm`: `Preprocessor` rewrites a raw description into the shared description format.
- `pydantic`: `Deal`, `DealSelection`, and `Opportunity`, which validate the JSON the model returns.
- `feedparser`: the scanner reads DealNews RSS. Parsing is in `ScrapedDeal.fetch` in `agents/deals.py`.
- `requests`: MessagingAgent calls the Pushover HTTP API.
- `gradio`: the final UI.
- t-SNE from `sklearn`, and `plotly`: only to see whether vectors cluster by category.
