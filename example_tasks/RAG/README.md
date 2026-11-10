# Example: retrieval-augmented question answering

This example answers a question from Lit2Vec paragraphs and cites the paragraphs it used. The notebook [`RAG.ipynb`](RAG.ipynb) is the implementation behind Section 3.15.1 of the paper.

## What the notebook does

1. Loads record JSON files and collects every paragraph with its 1024-dimensional embedding and the paper's metadata (title, authors, year, DOI).
2. Builds a FAISS `IndexFlatL2` over the paragraph vectors (exact search).
3. Encodes the question with `intfloat/e5-large-v2` and retrieves the top-k paragraphs.
4. Sends the paragraphs, labelled with their IDs (for example `9587808P32`), and the question to GPT-4o. The model is asked to answer in one paragraph and to cite the paragraph IDs in parentheses.
5. Prints the answer and a reference list built from the metadata of the cited papers.

## What you need

- Records with `paragraphs`, `embeddings` and `metadata`. These are the files the pipeline writes to `results/pipeline_output/` (see the quick start in the main [README](../../README.md)). The notebook reads its records from the folder in the `load_json_files(...)` call. At the moment that points to `example_docs/lit2vec_1K_sample_records`, which is not part of this repository, so change the path to your own folder of reconstructed records. Running `make repro` produces three records, which is enough to see the mechanics but not enough for a meaningful answer.
- The Python packages in the top-level `requirements.txt` (FAISS, `sentence-transformers`, `openai`, `python-dotenv`).
- An OpenAI API key, either in the environment (`export OPENAI_API_KEY=...`) or in a `.env` file in this folder. The `.env` file is ignored by git.

Answers come from GPT-4o and are only as good as the retrieved paragraphs. Check the cited paragraphs before you rely on a statement.
