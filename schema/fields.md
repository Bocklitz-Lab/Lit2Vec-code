# Lit2Vec record fields

This document describes the fields of a Lit2Vec record as defined in [`record.schema.json`](record.schema.json) (JSON Schema Draft 2020-12). A complete example is [`sample_record.json`](sample_record.json). Each record is one JSON file named after its corpus ID.

The full reconstructed records are not redistributed (see the main [README](../README.md)). This schema describes the records that the pipeline produces when you rebuild the corpus locally.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Schema version of the record (currently `"1.0"`). |
| `corpus_id` | string or integer | yes | Semantic Scholar corpus ID. Both types occur in practice, so cast before joining. |
| `metadata` | object | yes | Bibliographic metadata from the Semantic Scholar Academic Graph (see below). |
| `abstract` | string | yes | Abstract text. Empty if the source has no abstract. |
| `fulltext` | string | no | Full text normalised to Markdown (title as `#`, section headers as `##` or `###`). It is used to build the paragraphs and is not part of any public release. |
| `paragraphs` | object | yes | Chunk texts keyed by paragraph ID (see below). |
| `embeddings` | object | yes | 1024-dimensional vectors keyed by the same paragraph IDs. |
| `abstract_embedding` | array | yes | 1024-dimensional embedding of the abstract. |
| `tldr` | string | yes | Short summary of the abstract. |
| `predicted_subfield` | object | yes | Subfield label to score. |
| `unpaywall_license` | object or null | no | Raw license payload from Unpaywall. |
| `crossref_license` | object or null | no | Raw license payload from Crossref. |
| `openalex_license` | object or null | no | Raw license payload from OpenAlex. |

`abstract_embedding`, `tldr` and `predicted_subfield` depend on the abstract. For records without a usable abstract they are empty or absent, and the schema validation reports such records as incomplete (see Section 3.3 of the paper).

## `metadata`

| Field | Type | Required | Description |
|---|---|---|---|
| `_id` | string | no | Internal database ID. |
| `corpusid` | string or integer | no | Same value as `corpus_id`. |
| `externalids` | object | no | External identifiers (DOI, PubMed ID, and so on). |
| `url` | string (URI) | no | Semantic Scholar page of the paper. |
| `title` | string | yes | Title. |
| `authors` | array of objects | yes | Authors, each with `name` and optionally `authorId`. |
| `venue` | string | yes | Venue name. |
| `publicationvenueid` | string or null | no | Semantic Scholar publication-venue ID. |
| `year` | integer | yes | Publication year (1800–2100). |
| `referencecount`, `citationcount`, `influentialcitationcount` | integer or null | no | Counts from Semantic Scholar. |
| `isopenaccess` | boolean | no | Open-access flag from Semantic Scholar. |
| `publicationdate` | string or null | no | ISO date, for example `"2016-08-10"`. |
| `s2fieldsofstudy` | array of objects | no | Field-of-study labels from Semantic Scholar (`category`, `source`). |
| `publicationtypes` | array or null | no | Document types, for example `JournalArticle`. |
| `journal` | object or null | no | Journal name, volume and pages as given by Semantic Scholar. |

## `paragraphs` and `embeddings`

- Keys have the form `<corpus_id>P<n>`, where `n` counts the chunks of the paper from 0 (for example `2257319P0`).
- Each value in `paragraphs` is a chunk of about 170 tokens on average (the chunker targets 200), stored with the prefix `passage: ` that the e5 model expects. The vector for a paragraph was computed from exactly this string.
- `embeddings` has the same keys as `paragraphs`. Each vector has 1024 float32 values and is L2-normalised, produced by `intfloat/e5-large-v2`.

## `abstract_embedding`

A 1024-dimensional, L2-normalised vector of the string `passage: <abstract>`, computed with the same model as the paragraph embeddings.

## `predicted_subfield`

An object mapping subfield labels to scores between 0 and 1. It contains every label of the 18-class vocabulary whose score reaches the threshold of 0.5, so a record can carry several labels or none. The scores come from the Lit2Vec subfield classifier and are model predictions, not source metadata.

## `tldr`

A short summary, usually one or two sentences, generated from the abstract by the fine-tuned DistilBART model.

## License fields

`unpaywall_license`, `crossref_license` and `openalex_license` hold the raw payloads collected for the paper's DOI. They are the input of the license screening (`src/LicenseValidator.py`), which normalises them and resolves a single license. Check the license of a record before you share any of its content.
