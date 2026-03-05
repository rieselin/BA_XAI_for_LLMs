# ── Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 ────────────────────────────────
# Matches the CUDA version expected by PyTorch 2.x / unsloth
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCHDYNAMO_DISABLE=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── Upgrade pip ───────────────────────────────────────────────────────────────
RUN pip install --upgrade pip

# ── PyTorch (CUDA 12.1 wheel) ─────────────────────────────────────────────────
# Install before unsloth so it picks up the right CUDA build
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── bitsandbytes (GPU build) ──────────────────────────────────────────────────
RUN pip install bitsandbytes>=0.43.0

# ── unsloth (installs transformers, accelerate, peft, trl as deps) ────────────
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ── Remaining Python dependencies ─────────────────────────────────────────────
RUN pip install \
    fastapi \
    uvicorn[standard] \
    pydantic \
    numpy \
    shap \
    matplotlib \
    scipy \
    scikit-learn

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy application files
COPY server.py   .
COPY index.html . 

# ── (Optional) Pre-download the model weights at build time ───────────────────
# Saves cold-start time on first request. Comment out if you prefer lazy loading
# or if disk space in your build environment is constrained (~5 GB per model).
#
# RUN python -c "\
# from unsloth import FastLanguageModel; \
# FastLanguageModel.from_pretrained('unsloth/Llama-3.1-8B-Instruct-bnb-4bit', \
#     load_in_4bit=True, max_seq_length=8192); \
# FastLanguageModel.from_pretrained('unsloth/Llama-3.1-8B-bnb-4bit', \
#     load_in_4bit=True, max_seq_length=8192)"

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
