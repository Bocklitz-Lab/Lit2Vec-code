# Base image with conda and Python 3.10
FROM continuumio/miniconda3:latest

# Metadata
LABEL maintainer="Your Name <your@email.com>"
LABEL description="Lit2Vec: Open Chemistry Corpus with Paragraph Embeddings"

# Set working directory
WORKDIR /app

# Copy env file first for layer caching
COPY environment.yml .

# Create environment
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
SHELL ["conda", "run", "-n", "lit2vec", "/bin/bash", "-c"]

# Install any pip-only packages (if needed)
# RUN pip install some_package_not_in_conda

# Copy source code last (after env to preserve cache)
COPY . .

# Expose Jupyter and default run command
EXPOSE 8888

# Entrypoint: Jupyter Lab (or override in docker-compose)
CMD ["conda", "run", "--no-capture-output", "-n", "lit2vec", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
