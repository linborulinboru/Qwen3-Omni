# Dockerfile for Qwen3-Omni with CUDA 12.8 (supporting RTX 5090 sm_120) - Transformers only (No vLLM)

ARG CUDA_VERSION=12.8.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

FROM base AS dev

WORKDIR /

RUN mkdir -p /data/shared/Qwen3-Omni
RUN mkdir -p /data/models

WORKDIR /data/shared/Qwen3-Omni/

FROM dev AS bundle_req

FROM bundle_req AS bundle_transformers

ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

# Install PyTorch with CUDA 12.8 support (supporting sm_120 for RTX 5090)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install transformers from source (latest version)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install git+https://github.com/huggingface/transformers

# Install AutoAWQ for AWQ quantization support (8-bit model)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install autoawq autoawq-kernels compressed-tensors

# Install flash-attention 2 (compiled for CUDA 12.8 and sm_120)
ARG BUNDLE_FLASH_ATTENTION=true
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        pip install flash-attn --no-build-isolation; \
    fi

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip3 install networkx==3.4.2
RUN --mount=type=cache,target=/root/.cache/pip pip3 install accelerate==1.10.1 qwen-omni-utils huggingface_hub[cli] modelscope_studio modelscope

# Install Gradio and audio processing libraries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
    gradio==5.44.1 \
    gradio_client==1.12.1 \
    soundfile==0.13.1 \
    librosa \
    av

# Clean up pip cache
RUN rm -rvf /root/.cache/pip

# Final update: Ensure all critical dependencies are up-to-date
# This step ensures compatibility between transformers, huggingface-hub, and compressed-tensors
RUN pip3 install --no-cache-dir -U transformers huggingface-hub compressed-tensors

# Set working directory for the application
WORKDIR /app

# Set environment variables
ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV CUDA_LAUNCH_BLOCKING=0

EXPOSE 8901
