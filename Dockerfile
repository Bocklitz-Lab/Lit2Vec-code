# Base image: CUDA 11.8 + cuDNN + Ubuntu 22.04 (GPU-ready)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (add make for your Makefile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential make \
    git curl wget vim unzip \
    ca-certificates libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3 for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Install Python deps via requirements (cached layer)
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# Download language models at build time
RUN python -m nltk.downloader punkt && \
    python -m spacy download en_core_web_sm

# Copy the rest of the project
COPY . /workspace

# Default command runs your pipeline via Makefile
CMD ["make", "run"]
