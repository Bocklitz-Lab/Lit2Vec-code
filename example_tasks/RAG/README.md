# 📚 Lit2Vec: Citation-Aware RAG System for Scientific Literature

this pipeline uses a collection of structured JSON files to generate fluent scientific answers with **paragraph-level inline citations** and full references. It’s built for tasks like:
- Literature review
- Question answering over chemistry/biomedical papers
- Semantic retrieval and citation traceability

---

## 🚀 Features

- Parses and indexes paragraph-level data from JSON files
- Uses precomputed SentenceTransformer embeddings (`intfloat/e5-large-v2`)
- Semantic search via FAISS
- Answer generation with GPT-4o (OpenAI API v1.0+)
- Inline citations like `(9587808P32)` and full references like academic papers
- Supports multi-paragraph, multi-document answers

---

## 📁 Project Structure

```bash
.
├── lit2vec_1K_sample_records/  # Folder with JSON documents (S2ORC format)
├── RAG.ipynb                   # RAG pipeline with semantic retrieval + generation
└── README.md
````

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone git@github.com:Bocklitz-Lab/Lit2Vec-code.git
cd lit2vec
```

2. **Set your OpenAI API key**

You can either:

* Set it as an environment variable:

 ```bash
 export OPENAI_API_KEY=sk-...
 ```
* Or use a `.env` file:

 ```ini
 OPENAI_API_KEY=sk-...
 ```


3. **Put sample records inside folder**

Copy the provided JSON sample dataset into:

```
example_docs/lit2vec_1K_sample_records
```

Your folder tree should look like:

```
example_docs/
└── lit2vec_1K_sample_records/
    ├── 37254803.json
    ├── 9587808.json
    └── ...

```
---

## 🧠 How It Works

### 1. Preprocess JSON files

* Extract paragraph texts and precomputed embeddings
* Attach metadata (title, authors, corpus ID, DOI)
* Store them in a searchable vector index (FAISS)

### 2. Semantic Retrieval

* User query is encoded using `intfloat/e5-large-v2`
* FAISS returns top-k most relevant paragraphs

### 3. Answer Generation with Citations

* Retrieved paragraphs are passed to GPT-4o
* The model is prompted to cite specific paragraph IDs (e.g., `9587808P2`)
* A full reference list is automatically generated from metadata

---

## ✨ Example Output

**Query:**

> What are the benefits of EGCG in oxidative stress?

**Answer:**
EGCG, a polyphenolic compound found in green tea, exhibits potent antioxidant properties due to its two triphenolic groups, making it the most effective among green tea polyphenols (9587808P6). These properties enable EGCG to block the induction of nNOS and iNOS, scavenge free radicals, and attenuate lipid peroxidation, supporting its potential neuroprotective effects (9587808P7). The study demonstrates that EGCG significantly reduces NADPH-d/nNOS reactivity and preserves motor neuron survival following peripheral nerve injury (9587808P2, 9587808P32).

**References:**

```
[9587808] I. Wei, H. Tu, Chih-Chia Huang, Mang-Hung Tsai, C. Tseng, J. Shieh (2011). *(-)-Epigallocatechin gallate attenuates NADPH-d/nNOS expression in motor neurons of rats following peripheral nerve injury*. https://doi.org/10.1186/1471-2202-12-52
```

---

## 🧪 Requirements

* Python 3.8+
* OpenAI Python SDK (`openai>=1.0.0`)
* FAISS (`faiss-cpu`)
* SentenceTransformers
* numpy, dotenv, etc.



---

## 📄 License

This project is licensed under the MIT License.

---

## ✉️ Contact

Questions, ideas, or want to contribute? Open an issue or reach out!
