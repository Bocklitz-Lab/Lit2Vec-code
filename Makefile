# ==== CONFIG ====
ENV_NAME=lit2vec
PYTHON=conda run -n $(ENV_NAME) python
PIP=conda run -n $(ENV_NAME) pip

# ==== ENVIRONMENT ====
.PHONY: setup
setup:
	conda env create -f environment.yml || true
	conda run -n $(ENV_NAME) python -m nltk.downloader punkt
	conda run -n $(ENV_NAME) python -m spacy download en_core_web_sm

# ==== PIPELINE STAGES ====
.PHONY: ingest
ingest:
	$(PYTHON) pipelines/01_ingest/run.py

.PHONY: normalize
normalize:
	$(PYTHON) pipelines/02_normalize/run.py

.PHONY: chunk
chunk:
	$(PYTHON) pipelines/03_chunk/run.py

.PHONY: embed
embed:
	$(PYTHON) pipelines/04_embed/run.py

.PHONY: validate
validate:
	$(PYTHON) pipelines/05_validate/run.py

.PHONY: package
package:
	$(PYTHON) pipelines/06_package/run.py

# ==== FULL PIPELINE ====
.PHONY: run
run: ingest normalize chunk embed validate package

# ==== CLEANUP ====
.PHONY: clean
clean:
	rm -rf output/ data/ tmp/ logs/

# ==== TESTING ====
.PHONY: test
test:
	conda run -n $(ENV_NAME) pytest tests/

.PHONY: lint
lint:
	conda run -n $(ENV_NAME) flake8 src/ tests/

.PHONY: typecheck
typecheck:
	conda run -n $(ENV_NAME) mypy src/

.PHONY: check
check: lint typecheck test

# ==== SCHEMA + LICENSE ====
.PHONY: schema
schema:
	$(PYTHON) scripts/schema_check.py --input output/records.jsonl --schema schema/record.schema.json

.PHONY: license-audit
license-audit:
	$(PYTHON) scripts/license_audit.py --input output/records.jsonl

# ==== DOCKER (optional) ====
.PHONY: docker-build
docker-build:
	docker build -t lit2vec .

.PHONY: docker-run
docker-run:
	docker run -it --rm -v $(PWD):/app -p 8888:8888 lit2vec

