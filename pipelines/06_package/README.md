# Packaging: Final Dataset Export

This stage assembles all processed components into a structured, versioned release for public distribution.

---

## 🎯 Goal

Produce a reproducible, license-compliant dataset release that includes:

- Normalized full-text records (JSONL + Zstandard)
- Embedding shards (FAISS or binary format)
- Metadata (manifest, stats, license map)
- Schema, checksums, and sample subsets

Each release is versioned (e.g., `v1.0.0`) and archived via Zenodo/Hugging Face Datasets.

---

## 📂 Output Structure

```

lit2vec-v1.0.0/
├── data/
│   ├── records-00000.jsonl.zst
│   ├── records-00001.jsonl.zst
│   └── ...
├── embeddings/
│   ├── paragraphs-00000.fbin
│   └── ...
├── stats/
│   ├── global.json
│   ├── licenses.json
│   └── validation\_report.html
├── schema/
│   └── record.schema.json
├── samples/
│   └── 1k\_sample.jsonl
├── checksums.txt
├── manifest.json
└── README.md  ← this file

````

---

## 📦 Data Formats

- `*.jsonl.zst` — JSON Lines compressed with Zstandard (`zstd -T0 -19`)
- `*.fbin` — Embeddings in binary float32 format (`n_chunks × 1024 dim`)
- `record.schema.json` — JSON Schema describing all fields
- `manifest.json` — file sizes, shard IDs, and types
- `checksums.txt` — SHA256 for all files (reproducibility)
- `stats/` — license counts, subfield counts, quality metrics

---

## 🔐 Licensing

Each record includes:
- `license_unpaywall`, `license_crossref`, `license_openalex`
- A resolved `license_final` string (e.g., `CC-BY-4.0`)
- A boolean `license_allows_derivatives`

❗ Records that do **not** permit redistribution of modified content (`ND` licenses) are **excluded from full text output** — only metadata is preserved.

---

## 🧪 Validation

- All outputs are schema-validated (`schema_check.py`)
- Embedding alignment checked against record UUIDs
- Global stats recomputed to ensure consistency with earlier stages

---

## ⚙️ Reproduce

Run the full packaging pipeline:

```bash
make package
````

Or directly:

```bash
python pipelines/06_package/build_release.py \
  --input_records=data/normalized/ \
  --input_embeddings=embeddings/ \
  --output_dir=release/lit2vec-v1.0.0 \
  --config=configs/default.yaml
```

---

## 🔄 Versioning

Releases follow semantic versioning:

* `v1.0.0` — initial release (582,683 records, 28.8M chunks)
* `v1.1.0` — added subfield confidence scores
* `v2.0.0` — changed embedding model (E5 → E5-v2), schema change

---

## 📤 Upload Targets

* [ ] ✅ Zenodo (via GitHub tag → DOI)
* [ ] ✅ Hugging Face Datasets (`lit2vec/lit2vec`)
* [ ] 🔄 S3 mirror (optional)

---

## 📎 See Also

* [schema/record.schema.json](../schema/record.schema.json)
* [configs/default.yaml](../../configs/default.yaml)
* [scripts/license\_audit.py](../../scripts/license_audit.py)
* [stats/global.json](./stats/global.json)

---

*Last updated: 2025-08-14*

