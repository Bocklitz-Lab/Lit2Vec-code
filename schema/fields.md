# 📄 Lit2Vec Dataset — Field Reference

This document describes every field in a Lit2Vec dataset record. Each record is a JSON object conforming to [`record.schema.json`](./record.schema.json), containing normalized text, metadata, license flags, and paragraph embeddings.

---

## 🧩 Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `string` | Unique stable ID for the record (e.g., hash of DOI or OpenAlex ID). |
| `title` | `string` | Normalized title of the publication. |
| `abstract` | `string \| null` | Abstract text if available; `null` otherwise. |
| `authors` | `array[string]` | List of author names in publication order. |
| `published_date` | `string (YYYY-MM-DD)` | Publication date in ISO format. |
| `doi` | `string \| null` | Digital Object Identifier if known; otherwise `null`. |
| `source` | `string` | Text source (e.g., `openalex`, `unpaywall`, or publisher name). |

---

## 🔖 License Fields

| Field | Type | Description |
|-------|------|-------------|
| `license_final` | `string` | Final resolved license (e.g., `CC-BY-4.0`). |
| `license_allows_derivatives` | `boolean` | Whether the license permits redistribution of derivative works. |
| `license_sources` | `object` | Raw license data from each source:<br> - `unpaywall`: string or `null` <br> - `crossref`: string or `null` <br> - `openalex`: string or `null` |

---

## 🧪 Classification & Labels

| Field | Type | Description |
|-------|------|-------------|
| `subfield` | `string` | Predicted chemistry subfield label (see list below). |
| `classifier_confidence` | `number` (0–1) | Confidence score from the subfield classifier. |
| `summary_tldr` | `string \| null` | Optional short summary (TL;DR) generated via LLM. |

---

## 📄 Full Text & Embeddings

| Field | Type | Description |
|-------|------|-------------|
| `paragraphs` | `array[string]` | List of normalized paragraphs from the full text. |
| `paragraph_embeddings` | `array[array[float]]` | List of 1024-dim float vectors, aligned with `paragraphs`. |

---

## 🧠 Embedding Provenance

| Field | Type | Description |
|-------|------|-------------|
| `embedding_model` | `string` | Hugging Face model ID (e.g., `intfloat/e5-base-v2`). |
| `embedding_version` | `string` | Commit hash or version ID of the model/tokenizer used. |

---

## 🛠 Optional / Advanced Fields

| Field | Type | Description |
|-------|------|-------------|
| `license_conflict_flag` | `boolean` | Optional flag if license sources disagree. |
| `ingest_source_url` | `string` | Optional original URL for source document. |

---

## 🧪 Subfield Label Vocabulary

Possible values for `subfield` (18 total):

- `analytical_chemistry`  
- `biochemistry`  
- `chemical_engineering`  
- `computational_chemistry`  
- `electrochemistry`  
- `environmental_chemistry`  
- `inorganic_chemistry`  
- `materials_chemistry`  
- `medicinal_chemistry`  
- `nanochemistry`  
- `natural_products`  
- `nuclear_chemistry`  
- `organic_chemistry`  
- `organometallic_chemistry`  
- `physical_chemistry`  
- `polymer_chemistry`  
- `theoretical_chemistry`  
- `others` *(catch-all)*

---

## 📎 Example Record (truncated)

```json
{
  "record_id": "f6c62e...",
  "title": "A novel catalyst for selective CO2 reduction",
  "abstract": "We report a catalyst based on...",
  "authors": ["Alice Smith", "Bob Jones"],
  "published_date": "2023-04-18",
  "doi": "10.1000/xyz123",
  "source": "openalex",
  "license_final": "CC-BY-4.0",
  "license_allows_derivatives": true,
  "license_sources": {
    "unpaywall": "cc-by",
    "crossref": "CC-BY-4.0",
    "openalex": null
  },
  "subfield": "inorganic_chemistry",
  "classifier_confidence": 0.945,
  "summary_tldr": "A cobalt-based catalyst selectively reduces CO2 to ethanol under mild conditions.",
  "paragraphs": [
    "Carbon dioxide reduction is a major goal...",
    "In this study, we design a cobalt-based complex..."
  ],
  "paragraph_embeddings": [
    [0.021, -0.113, ..., 0.054],
    [0.035, -0.098, ..., 0.066]
  ],
  "embedding_model": "intfloat/e5-base-v2",
  "embedding_version": "b7e3baf"
}
