# Example: semantic paper recommendation

This example recommends papers from abstract embeddings. The notebook [`lit2vec_recommender.ipynb`](lit2vec_recommender.ipynb) is the implementation behind Section 3.15.2 of the paper. It supports two ways of querying:

- by corpus ID, to find papers similar to a known paper;
- by free text, to find papers related to an idea.

Results can be filtered by publication year, by the confidence of a predicted subfield, and by open-access status. Optionally the top results are reranked with maximal marginal relevance (MMR) to make them more diverse.

## What you need

- Records with an `abstract_embedding` and `metadata`. For the subfield filter they also need `predicted_subfield`, which the pipeline does not fill. You can generate it with `python -m src.subfield_predictor` (see the main [README](../../README.md)). The notebook reads its records from the folder in the `load_json_files(...)` call. At the moment that points to `example_docs/lit2vec_1K_sample_records`, which is not part of this repository, so change the path to your own folder of reconstructed records.
- The Python packages in the top-level `requirements.txt` (FAISS, `sentence-transformers`).

## Usage

```python
# Papers similar to a known paper
rec1 = recommend_by_corpus(
    corpus_id="123456",
    top_k=5,
    use_mmr=True,
    year_range=(2015, 2025),
    min_subfield=("Biochemistry", 0.6),
    open_access=True,
)

# Papers related to a free-text idea
rec2 = recommend_by_text(
    "green tea polyphenols mitigating oxidative stress in neurodegeneration",
    top_k=5,
    use_mmr=True,
)
```

The subfield names are the 18 labels of the Lit2Vec subfield classifier (for example `Biochemistry`, `Materials Science`, `Organic Chemistry`).

## How it works

- Embedding model: `intfloat/e5-large-v2`. Free-text queries are encoded with the `query: ` prefix.
- Search: FAISS `IndexFlatIP` over L2-normalised abstract embeddings, so the inner product equals the cosine similarity.
- Diversification: MMR with a relevance weight of 0.6 in the notebook examples.
