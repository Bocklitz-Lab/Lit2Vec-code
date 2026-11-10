# Lit2Vec

Lit2Vec is a workflow for building a license-screened, full-text chemistry corpus from the Semantic Scholar Open Research Corpus (S2ORC) and preparing it for retrieval and text mining. It covers Markdown normalisation of the parsed articles, token-aware paragraph chunking, e5-large-v2 embeddings, multi-source license screening, and a set of validation checks. This repository holds the code, the record schema and the validation tooling that accompany the paper:

> Amiri M, Jafari JM, Mostafapour S, Bocklitz T. *Lit2Vec: A Reproducible Workflow for Building a Legally Screened Chemistry Corpus from S2ORC for Downstream Retrieval and Text Mining.* arXiv:2604.12498 (2026). https://doi.org/10.48550/arXiv.2604.12498

Paper: [arXiv:2604.12498](https://arxiv.org/abs/2604.12498) · Validation data: [Lit2Vec-dataset](https://huggingface.co/datasets/Bocklitz-Lab/Lit2Vec-dataset) · Models: [TL;DR](https://huggingface.co/Bocklitz-Lab/lit2vec-tldr-bart-model), [subfield classifier](https://huggingface.co/Bocklitz-Lab/lit2vec-subfield-classifier-model)

## Scope

In the study we ran the workflow on the 2024-12-31 release of the Semantic Scholar Academic Graph. Of 18.6 million chemistry-labelled papers, about 1.07 million had parsed full text, and 582,683 of those had license metadata that met our screening rule (see [Licensing](#licensing)). Their text was split into roughly 28.8 million chunks of at most 200 tokens and embedded with `intfloat/e5-large-v2` (1024 dimensions, L2-normalised, float32).

The corpus itself is not redistributed. It is built from sources with mixed licenses and redistribution terms, and it is large (277 GB compressed, about 825 GB uncompressed). What we publish is the means to rebuild it from the pinned upstream release (code, schema, validation reports) rather than a copy of the text and vectors. Reproducibility in this project means regenerating the corpus from a fixed upstream snapshot, not downloading it.

## What is released

| Material | Where |
|---|---|
| Code, record schema, example records, notebooks | this repository |
| Technical validation reports for all 582,683 records | [Bocklitz-Lab/Lit2Vec-dataset](https://huggingface.co/datasets/Bocklitz-Lab/Lit2Vec-dataset) (`validation/`, 59 archives, about 1 GB) |
| TL;DR supervision set (19,992 CC-BY abstracts with GPT-4o summaries) | [lit2vec-tldr-bart-dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset) |
| Subfield classification set (multi-label, 18 classes) | [lit2vec-subfield-classifier-dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-subfield-classifier-dataset) |
| TL;DR model (fine-tuned DistilBART) | [model](https://huggingface.co/Bocklitz-Lab/lit2vec-tldr-bart-model), [training code](https://github.com/Bocklitz-Lab/lit2vec-tldr-bart), [demo](https://huggingface.co/spaces/Bocklitz-Lab/lit2vec-tldr-bart-space) |
| Subfield classifier (MLP on e5 embeddings) | [model](https://huggingface.co/Bocklitz-Lab/lit2vec-subfield-classifier-model), [training code](https://github.com/Bocklitz-Lab/lit2vec-subfield-classifier), [demo](https://huggingface.co/spaces/Bocklitz-Lab/lit2vec-subfield-classifier-space) |

Not released: the reconstructed full texts, the abstracts of the study corpus, paragraph exports, and embedding vectors derived from them.

## Workflow and code

The numbers refer to the steps in Table 2 of the paper.

| Step | Code |
|---|---|
| 1–4. Download S2AG, filter to Chemistry, join abstracts and full text by `corpus_id` | described in Section 2.1 of the paper; see the note below |
| 5. Convert annotated full text to normalised Markdown | `src/PaperProcessor.py`, `pipelines/02_Normalize` |
| 6. Token-aware chunking (200 tokens, 20 overlap, chunks under 100 tokens merged) | `src/ChunkProcessor.py`, `src/recursive_token_chunker.py` |
| 7. Embedding with e5-large-v2 (`passage:` prefix) | `src/EmbeddingProcessor.py`, `src/AbstractEmbeddingProcessor.py` |
| 8. License metadata from Unpaywall, OpenAlex and Crossref, then resolution | `src/license_fetcher.py`, `src/LicenseValidator.py` |
| 9–10. Subfield labels and TL;DR summaries (optional) | `src/subfield_predictor.py`, `src/tldr_generator.py` |
| Technical validation | `src/technical_validation.py` and the `*Validator.py` modules |

In the study, steps 1–4 ran against a local MongoDB instance. Those bulk scripts are not part of this repository at the moment. What is here for acquisition is single-paper retrieval: `src/s2orc_fetcher.py` looks up one DOI in S2ORC (the dataset endpoint requires a Semantic Scholar API key), and `src/license_fetcher.py` collects the three license payloads for one DOI.

## Quick start

The example run needs Linux, Python 3.10 or newer (we tested 3.11) and an NVIDIA GPU. The embedding stage reads the GPU temperature through NVML, so it will not start without one. The first run downloads `intfloat/e5-large-v2` (about 1.3 GB) from the Hugging Face Hub.

```bash
git clone https://github.com/Bocklitz-Lab/Lit2Vec-code.git
cd Lit2Vec-code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make repro
```

`make repro` processes the three papers in `example_docs/` through metadata assembly, normalisation, chunking, embedding and validation. It then compares per-field record hashes with the reference reports in `ground_truth/validation/`. Records end up in `results/pipeline_output/` and reports in `results/validation/`. Individual stages can be run with `make metadata`, `make normalize`, `make chunk`, `make embed`, `make validate` and `make compare`. The `repro` and `production` profiles in `configs/` use the same stage code and differ only in their output paths. A CUDA 11.8 `Dockerfile` is included.

The hash comparison skips `tldr` and `predicted_subfield`, which come from the enrichment models and not from the pipeline, and it does not cover the embedding vectors. The validation stage checks embeddings separately by re-encoding a sampled paragraph per record and comparing cosine similarity with the stored vector.

### Enrichment

```bash
python -m src.tldr_generator --abstract "Your abstract text"
python -m src.subfield_predictor --abstract "Your abstract text"
```

Both download their weights from the Hugging Face Hub on first use. The TL;DR model reads up to 1,024 input tokens. The subfield classifier is multi-label and returns every class with a score of at least 0.5 (`--threshold` changes this). Regenerating enrichment with the released checkpoints is not guaranteed to reproduce the values stored in the original study corpus exactly.

## Record format

Each record is one JSON object:

```text
schema_version        "1.0"
corpus_id             Semantic Scholar corpus id
metadata              bibliographic metadata from S2AG
abstract              abstract text
fulltext              normalised Markdown
paragraphs            {"<corpus_id>P<n>": "passage: <chunk text>", ...}
embeddings            {"<corpus_id>P<n>": [1024 floats], ...}, same keys as paragraphs
abstract_embedding    [1024 floats]            (abstract-dependent)
tldr                  short summary             (abstract-dependent)
predicted_subfield    {"<subfield>": score}     (abstract-dependent)
unpaywall_license     raw Unpaywall payload
crossref_license      raw Crossref payload
openalex_license      raw OpenAlex payload
```

Paragraph text is stored with the `passage: ` prefix that e5 expects, and the vector for each paragraph was computed from that exact string. `corpus_id` is stored as a string in some records and as an integer in others, so cast it before joining. Records whose abstract is missing or very short have no abstract embedding, TL;DR or subfield labels. Field definitions are in [`schema/fields.md`](schema/fields.md) and [`schema/record.schema.json`](schema/record.schema.json), and a complete example is in [`schema/sample_record.json`](schema/sample_record.json).

## Validation reports

Each record in the study corpus has a report with ten checks: schema, metadata, license, text, chunks, embeddings, identifiers, subfield labels, summary scores, and per-field SHA-256 hashes. The reports are on Hugging Face as tar archives of up to 10,000 JSON files each, named after the corpus id.

```python
import json
import tarfile
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Bocklitz-Lab/Lit2Vec-dataset",
    repo_type="dataset",
    filename="validation/data_part_058.tar.gz",
)

with tarfile.open(path) as tar:
    member = next(m for m in tar if m.name.endswith(".json"))
    report = json.load(tar.extractfile(member))["validation_info"]

print(member.name, report["overall_status"])
print(report["checks"]["license_validation"]["metrics"])
```

The hashes cover `metadata`, `abstract`, `fulltext`, `paragraphs`, the three license payloads, `tldr` and `predicted_subfield`. If you rebuild records yourself, `src/RecordHasher.py` computes the same hashes and `src/ValidationComparator.py` compares two sets of reports. Keep in mind that license payloads change over time, so their hashes only match if your metadata snapshot matches the one used in the study.

The notebooks in `notebooks/` produce the validation figures in the paper. Several of them read the reports from a local MongoDB instance filled by `src/write_validation_results_to_mongodb.py`. The ratings from the two expert reviewers of the TL;DR and subfield annotations are in `annotation_validation/`.

## Example applications

The folders under `example_tasks/` show three uses of a locally reconstructed corpus: retrieval-augmented question answering over paragraph embeddings with FAISS and GPT-4o (`RAG`), abstract-level paper recommendation with FAISS and maximal marginal relevance (`recomendation_system`), and a trend analysis of gold versus silver substrates in SERS and of 785 nm versus 532 nm excitation in Raman papers (`trend_analysis`).

## Models

The TL;DR model is `sshleifer/distilbart-cnn-12-6` fine-tuned on the 19,992 CC-BY abstracts with GPT-4o-generated summaries. The subfield classifier is a two-layer MLP (256 units, ReLU, dropout 0.3) on e5-large-v2 abstract embeddings with a sigmoid output over 18 classes. On the held-out test set at a threshold of 0.5 it reaches micro-F1 0.81, weighted-F1 0.80 and macro-F1 0.75. It works best on frequent classes such as Biochemistry (0.92) and worst on rare ones such as Chemical Engineering (0.53). Both models learned from GPT-4o pseudo-labels, so they inherit that model's biases. Training details are in the paper and in the two training repositories.

## Licensing

License screening in Lit2Vec is based on metadata only. For each paper with a DOI, license information is collected from Unpaywall, OpenAlex and Crossref and normalised to a common vocabulary. A paper enters the study corpus when at least two sources agree on an open license and no source contradicts them. Accepted are CC0, public domain, and the CC-BY, CC-BY-SA, CC-BY-NC and CC-BY-NC-SA variants. Records with restrictive, conflicting or indeterminate licenses are excluded. In the study corpus 87.5% of records resolve to CC-BY and 10.7% to CC-BY-NC.

This is an auditable screening step, not legal clearance. A corpus you rebuild is a collection of works under different licenses, so no single license covers it. Check the license of each record before you share any text or derived data, and observe the non-commercial and share-alike conditions where they apply. Follow the terms of Semantic Scholar, Unpaywall, OpenAlex and Crossref for the data you download from them. The three example papers in `example_docs/` are CC-BY and remain attributed to their original authors.

The materials in this project are licensed as follows:

| Material | License |
|---|---|
| Code (this repository and the two training repositories) | MIT |
| Validation reports, schema and example records | CC-BY 4.0 |
| TL;DR and subfield supervision datasets | CC-BY 4.0 |
| Model weights | Apache-2.0 |

## Limitations

Lit2Vec covers text only. Figures, chemical structures and tables are not represented. Rebuilding the corpus depends on continued access to the pinned S2AG release and on live metadata services whose answers can change after the study. Abstract-based enrichment exists only for records with a usable abstract. The subfield labels and TL;DR targets come from GPT-4o and were checked by two experts on a sample, so they are navigation aids and not ground truth. Subfield predictions for rare classes are less reliable.

## Citation

Please cite the paper when you use the code, the validation data, the models or the supervision datasets:

```bibtex
@misc{amiri2026lit2vec,
  title         = {Lit2Vec: A Reproducible Workflow for Building a Legally Screened Chemistry Corpus from S2ORC for Downstream Retrieval and Text Mining},
  author        = {Amiri, Mahmoud and Jafari, Jamile Mohammad and Mostafapour, Sara and Bocklitz, Thomas},
  year          = {2026},
  eprint        = {2604.12498},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2604.12498},
  url           = {https://arxiv.org/abs/2604.12498}
}
```

If you build on the underlying data, please also credit S2ORC (Lo et al., 2020) and the authors of any CC-BY text you reuse. GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).

## Contact and acknowledgements

Questions about the code or the data: open an issue, or write to Mahmoud Amiri (mahmoud.amiri@uni-jena.de). Contributions are described in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Lit2Vec was developed in the Bocklitz Lab at the Leibniz Institute of Photonic Technology (Leibniz IPHT) and Friedrich Schiller University Jena. The work was funded by the BMFTR within the programme Photonics Research Germany (13N15466, 13N15710, 13N15715) and is part of the Leibniz Centre for Photonics in Infection Research (LPI). We thank the Allen Institute for AI and the Semantic Scholar team for S2ORC.
