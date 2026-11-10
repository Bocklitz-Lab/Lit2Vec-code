# Contributing to Lit2Vec

Lit2Vec is a research workflow for building a license-screened chemistry corpus from S2ORC, described in [arXiv:2604.12498](https://arxiv.org/abs/2604.12498). Bug reports, fixes, documentation improvements and new example tasks are welcome. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

Because the paper reports numbers produced by this code, changes that alter pipeline output (chunking, normalisation, license resolution, embeddings) need more care than ordinary refactoring. If your change is of that kind, open an issue and describe it before you write the patch.

## Where help is most useful

- Scripts for the bulk acquisition steps (download of the S2AG dumps, Chemistry filtering, joins by `corpus_id`), which are not in this repository yet. See the workflow table in the [README](README.md).
- Automated tests. There is currently no test suite; `make repro` is the only end-to-end check.
- Corrections to license normalisation or to the schema documentation in `schema/`.
- Further example tasks in `example_tasks/`, and notes on results you obtain with a reconstructed corpus.

## Setting up

Fork the repository, clone your fork, and install the dependencies:

```bash
git clone https://github.com/<your-username>/Lit2Vec-code.git
cd Lit2Vec-code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The embedding stage needs an NVIDIA GPU. The README's quick start has the details.

## Before you open a pull request

1. Run `make repro`. It processes the three papers in `example_docs/` and compares per-field record hashes with `ground_truth/validation/`. The hashes should match. If your change is meant to alter the output, say so in the pull request, explain why, and update the files in `ground_truth/` in the same commit.
2. Follow the style of the surrounding code. `pyproject.toml` configures `black` and `ruff` with a line length of 100, which is a good default for new code. Please do not reformat files you are not otherwise changing.
3. Pin any new dependency in `requirements.txt`.
4. Document new scripts and config keys, and update the README if the user-facing behaviour changes.
5. Do not commit API keys, `.env` files, or generated output from `results/`.

## Data and licensing

This repository does not distribute the Lit2Vec corpus, and contributions should not add any part of it. Please do not paste article text, abstracts or embedding vectors from a reconstructed corpus into pull requests or issues. Use the corpus id or DOI instead.

If you need article text as a test fixture or an example, use a paper under CC0 or CC-BY. Record its DOI and license in the pull request, keep the attribution, and avoid no-derivatives (ND) papers. Fixtures under non-commercial or share-alike licenses should be avoided as well.

Changes to the license screening rule (which licenses are accepted, or how disagreement between Unpaywall, OpenAlex and Crossref is handled) change the corpus described in the paper, so please discuss them in an issue first.

By submitting a pull request you agree that your contribution is released under the license of this repository (see `LICENSE`).

## Reporting issues

Use the issue templates on GitHub. The bug report template is for code problems. The data issue template is for a wrong license resolution, a malformed field, a subfield label you disagree with, or an embedding mismatch in the validation reports or supervision datasets. Search existing issues first, and include the corpus id or DOI of any affected record.

Report security problems and accidentally published credentials privately as described in [SECURITY.md](SECURITY.md).

## Contact

For questions about collaboration or larger changes, write to Mahmoud Amiri at mahmoud.amiri@uni-jena.de.
