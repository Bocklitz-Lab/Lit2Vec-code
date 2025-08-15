# ==== CONFIG ====
PYTHON=python
PIP=pip
CONFIG_FILE=./configs/default.yaml

# ==== DOCKER ====
.PHONY: docker-build
docker-build:
	docker build -t lit2vec .

.PHONY: docker-run
docker-run:
	@if [ "$$CI" = "true" ]; then \
		echo "Running in CI - no GPU"; \
		docker run --rm -v $(PWD):/workspace -w /workspace lit2vec; \
	else \
		echo "Running locally - with GPU"; \
		docker run --rm --gpus all -v $(PWD):/workspace -w /workspace lit2vec; \
	fi



# ==== PIPELINE STAGES ====
.PHONY: metadata
metadata:
	$(PYTHON) -m pipelines.01_Metadata.run --config $(CONFIG_FILE)

.PHONY: normalize
normalize:
	$(PYTHON) -m pipelines.02_Normalize.run --config $(CONFIG_FILE)

.PHONY: chunk
chunk:
	$(PYTHON) -m pipelines.03_Chunk.run --config $(CONFIG_FILE)

.PHONY: embed
embed:
	$(PYTHON) -m pipelines.04_Embed.run --config $(CONFIG_FILE) --mode both

.PHONY: validate
validate:
	$(PYTHON) -m pipelines.05_Validate.run --config $(CONFIG_FILE)

.PHONY: package
package:
	$(PYTHON) -m pipelines.06_Package.run --config $(CONFIG_FILE)

.PHONY: all
all: metadata normalize chunk embed validate package

# ==== FULL PIPELINE ====
.PHONY: run
run: metadata normalize chunk embed validate package

# ==== CI TEST ====
.PHONY: test-no-embed
test-no-embed: metadata normalize chunk validate package

# ==== CLEANUP ====
.PHONY: clean
clean:
	rm -rf ./results/pipeline/*
	rm -rf ./data/chunked_json
	rm -rf ./data/embeddings
