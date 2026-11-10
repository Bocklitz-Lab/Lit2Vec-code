# Stage 06: compare record hashes

This stage checks that a rebuilt record is identical to a reference. It compares the per-field hashes that the validation stage (`05_Validate`) writes into each report with the hashes in a folder of reference reports. By default the reference reports are the ones in `ground_truth/validation/`.

```bash
make compare
# or
python -m pipelines.06_Compare.run --config configs/default.yaml
```

## What is compared

The validation stage hashes every top-level field of a record with SHA-256, after serialising it as JSON with sorted keys and no extra whitespace (`src/RecordHasher.py`). Fields listed under `VALIDATOR_CONFIG.hash_exclude_keys` are skipped. In the default configuration these are `abstract_embedding`, `corpus_id`, `embeddings`, `schema_version`, `tldr`, `predicted_subfield` and `validation_info`. What remains is `metadata`, `abstract`, `fulltext`, `paragraphs` and the three license payloads. The embedding vectors are therefore not part of this comparison. They are checked in the validation stage, which re-encodes a sample of paragraphs and compares cosine similarity.

## Configuration

The stage reads the `COMPARE_CONFIG` block of the YAML file:

| Key | Meaning |
|---|---|
| `ground_truth_dir` | Folder with the reference reports (`<corpus_id>.json`). |
| `generated_dir` | Folder with the reports from your run. |
| `ignored_fields` | Fields whose hashes are not compared. The default is `tldr` and `predicted_subfield`, which come from the enrichment models and not from the pipeline. |

## Behaviour

For every file in `ground_truth_dir` the stage looks for a file with the same name in `generated_dir`. A missing file is reported and counted. For each existing pair it prints the fields whose hashes differ, with the expected and the found value. At the end it prints a summary line. The stage does not exit with a non-zero status when hashes differ, so check its output (or its log in CI) and not its return code.

## Comparing with the published reports

The validation reports of the study corpus are on Hugging Face (see the main [README](../../README.md)). To compare records you rebuilt yourself, extract the reports for those records into a folder of their own and point `ground_truth_dir` at it. Do not point it at all 582,683 reports unless you rebuilt all of them, because every record without a counterpart in `generated_dir` is reported as missing. Reports written by earlier versions of the validator also contain hashes for `tldr` and `predicted_subfield`. The default `ignored_fields` skips them.

Hashes of the license payloads (`unpaywall_license`, `crossref_license`, `openalex_license`) only match if your metadata snapshot matches the one used for the reference. Live API responses change over time.
