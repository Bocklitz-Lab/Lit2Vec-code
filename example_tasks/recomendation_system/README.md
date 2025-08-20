# 🔬 Semantic Paper Recommendation (1K Sample)

This project demonstrates a lightweight **semantic paper recommendation system** using abstract-level embeddings, FAISS similarity search, and MMR reranking. It supports:

- **Finding similar papers by corpus ID**  
- **Free-text search recommendations**  
- **Optional filtering by year, subfield prediction, open access**  
- **MMR-based diversity reranking**

---

## ⚙️ Installation

Install required Python packages (preferably in a virtual environment):

```bash
pip install numpy faiss-cpu sentence-transformers openai python-dotenv tqdm pandas
````

## 🚀 How to Use

Open and run the `lit2vec_recommender.ipynb` notebook.

### 1. Recommend by Known Paper

```python
rec1 = recommend_by_corpus(
    corpus_id="123456",
    top_k=5,
    use_mmr=True,
    year_range=(2015, 2025),
    min_subfield=("Neuroscience", 0.6),
    open_access=True
)
```

### 2. Recommend by Free-Text Idea

```python
rec2 = recommend_by_text(
    "green tea polyphenols mitigating oxidative stress in neurodegeneration",
    top_k=5,
    use_mmr=True
)
```

---

## 🧠 Key Components

* **Embedding model:** `intfloat/e5-large-v2` from SentenceTransformers
* **Similarity search:** FAISS `IndexFlatIP` with L2-normalized embeddings
* **Diversification:** Maximal Marginal Relevance (MMR)
* **Metadata filters:** year range, open access flag, subfield confidence

---

## 📄 License

MIT License — feel free to adapt or extend this system for your own research or applications.

---

## 👨‍💻 Author

This example was built to showcase semantic search in scientific literature using embeddings. If you found it useful, feel free to reach out or contribute!
