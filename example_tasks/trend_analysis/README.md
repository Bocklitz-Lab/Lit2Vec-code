# Trend Analysis with Lit2Vec

This example demonstrates how to use the Lit2Vec corpus for fine-grained trend analysis in chemistry, focusing on materials and techniques mentioned in experimental sections of papers.

The analysis reproduces the results shown in our paper's supplementary material (see the `Trend Analysis` subsection), and illustrates one of the key downstream applications enabled by Lit2Vec's structured full text and paragraph-level semantic representations.

---

## 🔍 What It Does

- Extracts yearly trends in:
  - SERS with **gold** or **silver** nanoparticles
  - Raman spectroscopy with **785 nm** or **532 nm** excitation
- Restricts analysis to **relevant sections** (e.g. Methods, Instrumentation, Results)
- Applies proximity filters to ensure co-occurrence in context
- Generates:
  - A CSV of per-document flags
  - A CSV of yearly trend aggregations
  - Trend plots for use in publications or presentations

---

## ⚙️ Requirements

```bash
pip install pandas matplotlib seaborn duckdb orjson
```

---

## 🚀 How to Run

1. **Extract features from JSON corpus:**

```bash
python trend_analysis.py /path/to/lit2vec/json/folder
```

This scans all `.json` files, computes trend-relevant flags per paper, aggregates them by year, and saves:

* `doc_level_flags.csv`
* `yearly_trends.csv`

2. **Generate plots:**

```bash
python plot_yearly_trends.py
```

This creates publication-quality plot:

* `chem_trends_combined_2plots.png`

---

## 📄 Citation

If you use this code or figures derived from it, please cite:

> Lit2Vec: A FAIR, derivative-permissive corpus of 582k chemistry papers with semantic paragraph embeddings and structured metadata
> *Submitted to Nature Scientific Data, 2025*

---

## 🔗 Related Resources

* 📘 [Main Lit2Vec Repository](https://github.com/Bocklitz-Lab/Lit2Vec-code)
* 📦 Dataset DOI: *coming soon*

---

## 📬 Questions?

Feel free to open an issue or contact us via the GitHub repo.

