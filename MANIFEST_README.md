# Lit2Vec record-level manifest

`lit2vec_manifest.csv.gz` lists every record in the Lit2Vec study corpus (one row per paper), built from the Semantic Scholar Academic Graph (S2AG) release **2024-12-31**. It holds identifiers, the license evidence and decision for each paper, and integrity hashes. It contains no full text, abstracts, paragraphs or embeddings.

## What it is for

- **Reconstruction:** the exact list of papers to rebuild, keyed by `corpus_id` and DOI.
- **License audit (frozen license-resolution metadata):** for each paper, the license value reported by Crossref, Unpaywall and OpenAlex at screening time (after normalisation), the resolved license, which sources it rests on, and the date of the check. Later queries to these services may return different answers, so this table is the reference for the study's screening decisions.
- **Verification:** SHA-256 hashes of fields in each record, computed with `src/RecordHasher.py` (canonical JSON, sorted keys). A rebuilt record matches the study record when its hashes are equal. The hashes of the three license payloads only match if your license metadata is the same as in the study.

## Columns

| Column | Meaning |
|---|---|
| `corpus_id` | Semantic Scholar corpus id |
| `doi` | DOI(s), lower-case; several are separated by `;` |
| `s2ag_release` | S2AG release the record was built from |
| `license_crossref`, `license_unpaywall`, `license_openalex` | License reported by each source, normalised by `src/LicenseValidator.py` (e.g. `cc-by`, `cc-by-nc`, `missing`) |
| `resolved_license` | Final license after resolution |
| `license_source` | Sources the decision rests on (`all`, or e.g. `crossref+openalex`) |
| `license_status` | `pass` = at least two sources agree on an accepted open license and none contradicts them |
| `license_checked_at` | Date the license check was run (YYYY-MM-DD) |
| `n_paragraphs` | Number of paragraph chunks in the record |
| `sha256_metadata`, `sha256_fulltext`, `sha256_paragraphs` | Hashes of the record fields |
| `sha256_crossref_license`, `sha256_unpaywall_license`, `sha256_openalex_license` | Hashes of the raw license payloads as stored in the record |

## Rebuilding and checking

1. Rebuild records with the Lit2Vec code from S2AG 2024-12-31, keeping only the `corpus_id`s listed here.
2. Hash the rebuilt records with `src/RecordHasher.py` and compare them with the `sha256_*` columns (or the full reports with `src/ValidationComparator.py`).

Full per-record validation reports, including all ten checks, are in `validation/`.

License: CC-BY 4.0. Cite the Lit2Vec paper when you use this file.
