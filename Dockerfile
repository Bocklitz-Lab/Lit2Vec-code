# Base image: PyTorch with CUDA 11.8 and cuDNN, Python 3.10
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    unzip \
    ca-certificates \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Optional: Install Miniconda (not preinstalled in PyTorch image)
ENV CONDA_DIR=/opt/conda
RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment.yml into container
COPY environment.yml /tmp/environment.yml

# Create Conda environment
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Set default shell to bash with Conda activated
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate lit2vec" >> ~/.bashrc
ENV PATH /opt/conda/envs/lit2vec/bin:$PATH
ENV CONDA_DEFAULT_ENV=lit2vec

# Set working directory inside the container
WORKDIR /workspace

# Copy your project files (adjust if needed)
COPY . /workspace

# Default command (optional)
CMD ["python"]
