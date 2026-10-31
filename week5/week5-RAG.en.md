# Week 5: RAG

This week builds an internal Q&A assistant for Insurellm, a fictional insurance-tech company. When employees ask about products, employees, contracts, or company facts, the answer has to be accurate and the solution has to stay cheap. The method is RAG: find the relevant passages in a knowledge base, put those passages into the prompt, and let the generative model answer from them.

The knowledge base is in `knowledge-base/`. It is all markdown, split by folder into `products`, `employees`, `contracts`, and `company`.

The days build in this order:

1. Day 1: hard keyword match, and why RAG puts context into the prompt.
2. Day 2: chunk, embed, store in Chroma, and plot the vectors.
3. Day 3: retrieve from the vector store, then call a chat model.
4. Day 4: use a test set to score retrieval and answer quality separately.
5. Day 5: leave LangChain, let an LLM chunk documents, rewrite the question, and rerank the hits.

Code for this week:

- Basic ingest: `implementation/ingest.py`
- Basic answering: `implementation/answer.py`
- Chat UI: `app.py`
- Evaluation: `evaluation/test.py`, `evaluation/eval.py`, `evaluation/tests.jsonl`
- Advanced ingest and answering: `pro_implementation/ingest.py`, `pro_implementation/answer.py`

## Concepts

RAG is Retrieval Augmented Generation. The model weights do not contain this company's internal documents. On every question, the program searches the knowledge base, appends the relevant source text to the prompt, and the model generates an answer. In this course the retrieved text goes in the system prompt. The user's question stays in the user prompt.

With no RAG, the model can only use public knowledge from training, so company contracts and employee files come out wrong or invented. Stuffing the entire knowledge base into every prompt also fails: tokens cost money, the context window is finite, and irrelevant passages distract the answer. RAG inserts only the closest few passages.

Two phases:

1. Ingest, done offline, and redone when documents change. Read documents, split them into chunks, turn each chunk into a vector, and store the vector with the original text and metadata.
2. Query, done on every user question. Turn the question into a vector, find the nearest chunks, paste them into the prompt, and call the chat model.

An embedding is a list of floats made from a piece of text. Text with similar meaning has nearby vectors. Retrieval compares vector distance, so "who founded the company" can hit a passage that says founder, even when the question does not contain the document's original words.

A generative model and an embedding model are different:

- An autoregressive model looks at earlier text and predicts the next token. It writes the answer. GPT is in this family.
- An autoencoding model reads the whole input and encodes it. BERT, OpenAI Embeddings, and Hugging Face sentence-transformers are in this family. The embedding model produces the vector. The vector database does not.

A token and a vector are also different. A token is a small piece of split text, an id in a vocabulary, and that id does not compare meaning. A vector is the list of numbers that represents meaning. Text is split into tokens first. The embedding model then folds them into one vector.

`tiktoken` only counts tokens with a model's vocabulary, so you can estimate length and cost. It does not produce the vectors RAG searches with.

- `encoding_for_model("gpt-4.1-nano")` looks up the model name, then returns the matching tokenizer. The tables are in the installed `tiktoken/model.py`: `MODEL_TO_ENCODING` for exact names and `MODEL_PREFIX_TO_ENCODING` for prefixes.
- `get_encoding("o200k_base")` asks for the tokenizer directly. `o200k_base` is a tokenizer name, not a model name. The `o` comes from the GPT-4o generation, `200k` means a vocabulary of about 200,000, and `base` is the standard version. GPT-4o, GPT-4.1, and GPT-5 point at it. Earlier GPT-4 uses `cl100k_base` (about 100,000).
- The installed tiktoken does not recognize `gpt-6-luna`. The unmerged [tiktoken PR 616](https://github.com/openai/tiktoken/pull/616) proposes mapping the GPT-6 family to `o200k_base` as well. For a local count, call `tiktoken.get_encoding("o200k_base")`. Leave `MODEL` as a model name. `encoding_for_model` only accepts model names.

A chunk is the unit of retrieval. A whole document is long, wastes tokens in the prompt, and drags in unrelated sentences. Chunks that are too small drop context. A common compromise is a fixed character count with overlap between neighbors, so a sentence is less likely to be cut on the boundary. The Day 2 notebook uses 1,000 characters and an overlap of 200. `implementation/ingest.py` uses 500 characters and an overlap of 200. Overlap means the same sentence can sit in two chunks, which makes a miss less likely and stores more.

Metadata is a label on the chunk, such as the source path and `doc_type`. Retrieved results can show the source. Evaluation and filters use it too.

Temperature controls how the next token is drawn. 0 means the highest-probability token is preferred, which fits factual Q&A. Tone comes from the system prompt. A temperature of 0 still does not guarantee identical text on every run.

## How one RAG pipeline is wired

Basic version (Day 2 ingest plus Day 3 answering):

1. `DirectoryLoader` and `TextLoader` read the `.md` files in each folder and set `doc_type` on each document.
2. `RecursiveCharacterTextSplitter` chunks them. It splits on paragraphs, newlines, and spaces, in that order, and tries to keep a sentence intact.
3. Pick an embedding model and turn each chunk into a vector. The Day 2 and Day 3 notebooks use Hugging Face `all-MiniLM-L6-v2` (384 dimensions, local, free). `implementation/` switches to OpenAI `text-embedding-3-large`. Ingest and query for the same store must use the same embedding model, so the dimensions and the space match.
4. `Chroma.from_documents(...)` writes the local directory `vector_db`. If a store already exists, delete the collection and rebuild it, so old and new vectors are not mixed.
5. On a question, `vectorstore.as_retriever()` encodes the question and returns the nearest chunks. `implementation/answer.py` sets `RETRIEVAL_K = 10`.
6. Join the chunk text into `{context}` in the system prompt.
7. `ChatOpenAI` generates the answer from the history and the current question. If it does not know, it says so.

`app.py` uses Gradio to show the conversation in one column and the retrieved source text in the other.

The advanced version (Day 5, `pro_implementation/`) adds an LLM before and after retrieval, and stores vectors in `preprocessed_db`:

1. Skip LangChain for loading. Walk `knowledge-base/**/*.md` directly.
2. Let an LLM decide the chunks. Each chunk has three parts: a headline (a short title that is easy to retrieve), a summary (a few sentences), and `original_text` (the source, unchanged). The text stored in the vector database is those three parts joined together. The prompt asks for full coverage and about 25% overlap, or about 50 words, between chunks.
3. Compute vectors with the OpenAI embeddings API and write them with Chroma's `PersistentClient`.
4. On a question, rewrite the user's question into a shorter, more specific search query (query rewriting). Retrieve once with the original question and once with the rewrite (`RETRIEVAL_K = 20`), then merge and drop duplicates.
5. Ask an LLM to rerank every candidate. Keep the top `FINAL_K = 10` passages for the prompt.
6. Include the source path in the generated answer so it can be checked.

## What each day covers

### Day 1: brute-force RAG

No vectors. Load employee surnames and product names into a dictionary. Split the question on spaces. If a word is exactly a dictionary key, put that whole document into the system prompt. `Who is Lancaster and what is carllm?` hits. A paraphrase misses. The point is the smallest RAG loop: retrieve, build the prompt, call `openai.chat.completions`, chat in Gradio. Semantic retrieval is what removes the requirement that the question contain the original word.

### Day 2: chunks, vectors, visualization

Use `tiktoken` first to see roughly how many tokens the whole knowledge base is, and whether it could fit in one context window. Then LangChain loads, chunks, embeds, and writes Chroma. The notebook has about 76 files and 413 chunks.

Visualization is not a retrieval step. Take every vector, compress it to 2D or 3D with t-SNE, and color the Plotly chart by `doc_type`. The plot shows whether documents of the same type sit together. t-SNE tries to keep points that were close in the original space close on the plot. The axes have no business meaning.

### Day 3: retrieve, then generate

Two LangChain objects show up often:

- Retriever: wraps the embedding model and the vector store. `retriever.invoke(question)` returns a list of documents.
- LLM: wraps the chat model. `llm.invoke(messages)` returns the answer.

Calling the LLM alone, with no retrieved passages, does not give it Insurellm's internal facts.

### Day 4: evaluation

The course treats evaluation as more important than the choice of framework. If you change chunk size, embedding model, or how many chunks you retrieve, read the numbers on the same test set.

The test set is `evaluation/tests.jsonl`, 150 questions. Each row has question, keywords, reference_answer, and category. The categories are roughly: direct_fact 70, temporal 20, spanning 20, comparative 10, numerical 10, relationship 10, holistic 10. Keywords are words that should appear in the retrieved context. They score retrieval. They are not an answer for the model to memorize.

Retrieval metrics, computed per keyword and then averaged:

- MRR: the reciprocal of the rank of the first hit. Rank 1 scores 1, rank 2 scores 1/2, a miss scores 0. Higher means the relevant chunk is near the top.
- nDCG: whether relevant chunks are ranked higher. A relevant chunk placed later is discounted.
- Keyword coverage: what percentage of this question's keywords appear in at least one retrieved chunk. It is close to recall.

Answer metrics use a second LLM as a judge, scored from 1 to 5 against the reference answer:

- accuracy: whether the facts are right. A factual error scores 1.
- completeness: whether the information in the reference answer is included.
- relevance: whether the answer stays on the question that was asked, without extra material.

Strong retrieval and a weak answer points at the prompt or the generative model. An invented answer whose retrieved passages contain no support points at retrieval, or at the model ignoring "say so if you do not know."

### Day 5: ten advanced techniques

The notebook lists these directions. This week's code actually does chunk-size experiments, document pre-processing, query rewriting, and reranking. The rest is a list, and the main code this week does not implement it.

1. Chunking: change chunk size and overlap, and compare on the Day 4 test set.
2. Encoder: swap the embedding model, and choose with the same test set.
3. Prompt: state the role, today's date, the retrieved context, and the conversation history.
4. Document pre-processing: before ingest, have an LLM write a title and a summary, and encode those together with the original text.
5. Query rewriting: turn a spoken question into a short query that retrieves better.
6. Query expansion: turn one question into several retrieval queries.
7. Re-ranking: retrieve extra hits with vectors, then have an LLM reorder them and drop the weak ones.
8. Hierarchical RAG: retrieve a summary first, then drill into the original text.
9. Graph RAG: follow relationships between similar documents and pull more content.
10. Agentic RAG: let an agent decide whether to search a vector store, query SQL, or use another tool.

Day 4 also states the limits of RAG. It adds knowledge in the prompt, and the result is empirical. The chunking strategy changes the score a lot. When retrieval misses, the model suddenly says it does not know. An architecture that sounds reasonable can still score poorly.

## What each library does

Only the libraries this week actually uses.

Reading files and configuration:

- `pathlib` and `glob`: find files under `knowledge-base`. Day 1 and the pro version read the text with them directly.
- `load_dotenv` from `python-dotenv`: reads `OPENAI_API_KEY` and other secrets from `.env`.
- `tiktoken`: counts tokens with a model vocabulary. See Concepts above.
- `numpy`: stacks the Chroma vectors into a matrix for t-SNE.

LangChain is used on Day 2, Day 3, and in `implementation/`. It gives one interface for loading, splitting, vector stores, and chat models, which is fast to start with. Once the model APIs themselves are simple, the extra layer is heavy, and some wrappers go stale. Day 5 drops it so chunking and retrieval stay under direct control.

- `DirectoryLoader` and `TextLoader` (`langchain_community.document_loaders`): read text by directory into a `Document` (body plus metadata).
- `RecursiveCharacterTextSplitter` (`langchain_text_splitters`): recursive splitting by character count. `MarkdownTextSplitter` in the same package is commented out in `ingest.py`. Splitting on markdown structure is the alternative.
- `HuggingFaceEmbeddings` (`langchain_huggingface`): runs sentence-transformers locally. This week uses `all-MiniLM-L6-v2`.
- `OpenAIEmbeddings` (`langchain_openai`): calls OpenAI's embedding API. The alternative this week, and the choice in `implementation/`, is `text-embedding-3-large`.
- `ChatOpenAI` (`langchain_openai`): calls the chat model. The basic version uses `gpt-4.1-nano` at temperature 0.
- `Chroma` (`langchain_chroma`): the LangChain wrapper around Chroma. `from_documents` embeds and writes. `as_retriever` queries.
- `SystemMessage`, `HumanMessage`, and `convert_to_messages` (`langchain_core`): pack the system message, history, and current question into LangChain messages.

Vector store and visualization:

- Chroma: a local vector database. Data lives in `vector_db/` or `preprocessed_db/`. It stores vectors and original text, and queries by distance. The embedding model has to compute the vectors first.
- `chromadb.PersistentClient`: opens the same database without LangChain. The pro version writes with `collection.add` and queries with `collection.query`.
- `sklearn.manifold.TSNE`: compresses hundreds of dimensions to 2D or 3D, for the plot only.
- `plotly.graph_objects`: draws an interactive scatter plot.

Generation, structure, and UI:

- `openai.OpenAI`: the official SDK. Day 1 uses its chat completions. The pro version uses `embeddings.create`.
- `litellm.completion`: one interface for models from different providers. Pro ingest chunks with `openai/gpt-4.1-nano`. The answer script switches to `groq/openai/gpt-oss-120b`. Structured output uses `response_format` set to a Pydantic model.
- `pydantic.BaseModel`: the JSON shape the LLM must return. Chunking uses `Chunk` / `Chunks`. Reranking uses `RankOrder`. Evaluation uses `RetrievalEval`, `AnswerEval`, and `TestQuestion`.
- `tenacity.retry`: retries a failed call with exponential backoff. The pro version uses it for rate limits.
- `multiprocessing.Pool`: chunks many documents in parallel. On rate limits, set `WORKERS` to 1.
- `tqdm`: the chunking progress bar.
- `gradio`: the chat UI. Day 1 and Day 3 use `ChatInterface`. `app.py` uses `Blocks`, with the conversation on one side and the retrieved passages and sources on the other.

Vector stores mentioned in the course and not wired into this week's main code: open-source Qdrant and FAISS (in memory), hosted Pinecone and Weaviate, and Postgres, Mongo, and Elasticsearch with vector support. Embedding models also mentioned: word2vec, BERT, and Google `gemini-embedding-001`.

## When to use the basic version and the advanced version

The basic version is the one to get running first: LangChain chunking, one embedding model, Chroma, then generate from the retrieved passages. After you change `chunk_size`, `chunk_overlap`, or the embedding model in `implementation/ingest.py`, ingest again, then run `evaluation/eval.py`.

The advanced version hands "how to cut" and "how to rank" to an LLM. Retrieval can be more accurate. Ingest and every question are slower and more expensive. It retrieves 20 passages before reranking and keeps 10 afterward, so a passage the vector search ranked too low still has a chance to be moved up by the LLM.

The embedding model used at query time has to be the one used at ingest time. Changing the model means rebuilding the vector store. Hugging Face's 384-dimensional vectors and OpenAI's higher-dimensional vectors cannot be queried together in the same collection.
