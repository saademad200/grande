FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install jupyterlab \
    && python3 -m pip install ipykernel

RUN python3 -m pip install --no-cache-dir typing_extensions==4.8.0 \
    && python3 -m pip install --no-cache-dir referencing==0.31.0 \
    && python3 -m pip install --no-cache-dir jsonschema==4.20.0 \
    && python3 -m pip install --no-cache-dir jupyterlab==4.0.6 \
    && python3 -m pip install --no-cache-dir ipykernel==6.25.0

# Copy the rest of the application
COPY . .

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# Create directories for results and data
RUN mkdir -p results data

# Set up Jupyter kernel and configure it
RUN python3 -m ipykernel install --user --name=grande --display-name="Python (GRANDE)"

# Set permissions for jupyter directories
RUN mkdir -p /root/.jupyter && \
    mkdir -p /root/.local/share/jupyter
# Expose port for Jupyter
EXPOSE 8888

# Set working directory for when container starts
WORKDIR /app