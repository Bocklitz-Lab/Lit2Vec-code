# Contributing to Lit2Vec

🎉 Thanks for your interest in contributing to **Lit2Vec** — a CC-licensed, reproducible corpus for chemistry text mining and retrieval-augmented applications.

We welcome contributions that improve the codebase, expand dataset coverage, enhance documentation, or suggest new use cases. Please read the guidelines below before submitting code or data.

---

## 👥 Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and inclusive. Toxic behavior will not be tolerated.

---

## 🧩 Ways to Contribute

### 🚀 Code
- Fix bugs or add tests
- Improve performance or modularity
- Add support for new LLM embeddings
- Improve or extend the pipeline stages

### 🗃️ Data
- Suggest additional open-access chemistry sources
- Contribute metadata enrichments (e.g., subfield corrections)
- Improve license mappings or source audits

### 📚 Docs
- Clarify usage in the README
- Expand schema field documentation
- Add new example notebooks or tutorials

### 🧪 Research
- Suggest evaluation benchmarks or retrieval tasks
- Report findings or downstream results using Lit2Vec

---

## 🛠️ Getting Started

### 1. Fork the repository

```bash
git clone https://github.com/YOUR_USERNAME/lit2vec.git
cd lit2vec
conda env create -f environment.yml
conda activate lit2vec
````

### 2. Run the pipeline

You can test the pipeline on a small sample set using:

```bash
make sample  # or run: python scripts/run_sample_pipeline.py
```

### 3. Run tests

```bash
pytest tests/
python scripts/schema_check.py --path samples/1k-records.jsonl
python scripts/license_audit.py --path samples/1k-records.jsonl
```

---

## 📂 Project Structure

```
lit2vec/
├── pipelines/        # Modular data processing stages
├── scripts/          # Helpers for license checking, schema, etc.
├── schema/           # JSON schema and field-level docs
├── tests/            # Unit tests
├── notebooks/        # Demos and figures
├── configs/          # Model and pipeline config
├── Makefile          # Reproducibility commands
└── Dockerfile        # Containerized build
```

---

## ✅ Contribution Checklist

Before submitting a pull request:

* [ ] Your code passes `pytest` and `black .`
* [ ] All JSON records validate against `record.schema.json`
* [ ] You ran `scripts/license_audit.py` on any new records
* [ ] If you added data, it **does not include** any ND (no-derivatives) licenses
* [ ] You documented any new scripts or configs
* [ ] You updated relevant README sections or notebooks if applicable

---

## 🔏 Data Licensing Rules

> **⚠️ Important:** This project distributes full-text data **only if** the license allows derivative works.

Any submitted data must:

* Come from an **open-access source**
* Be under a **CC-BY, CC0, CC-BY-SA, or CC-BY-NC** license
* **NOT** be CC-BY-ND, CC-BY-NC-ND, or copyrighted without permission

Use `scripts/license_audit.py` to test compliance on any new additions.

---

## 📢 Reporting Issues

Found a bug, licensing conflict, or incorrect subfield label?

1. Search existing issues first
2. Open a new issue and describe:

   * What you expected
   * What happened
   * How to reproduce (if applicable)
   * Affected files or records

Bug reports, licensing concerns, and suggestions are all welcome.

---

## 💬 Need Help?

If you're unsure whether your contribution is welcome, just open a draft pull request or start a discussion.

For deeper collaboration or publication proposals, contact us at:
📧 [your.email@institution.edu](mailto:your.email@institution.edu)

---

## ✨ Acknowledgment

This project was built with openness and reproducibility in mind. Your contributions help make chemistry research more FAIR and accessible to the community.

Thanks for helping improve **Lit2Vec**!

—
The Lit2Vec team

```

---

### Tips:
- Link this from the main `README.md`.
- Pair it with a `CODE_OF_CONDUCT.md` (I can provide one too if you need).
- Use GitHub labels like `good first issue`, `license-needed`, or `help wanted` to encourage contributions.

Let me know if you’d like a **GitHub Discussions template**, a **PR checklist**, or a **pre-push hook** to auto-run schema and license validation.

