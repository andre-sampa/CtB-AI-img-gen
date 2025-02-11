# Use the official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0"  # CUDA architecture for Hopper (H100)

# Debug: Print starting message
RUN echo "🚀 Starting Docker build process..."

# Install system dependencies
RUN echo "🔧 Installing system dependencies..." \
    && apt-get update \
    && apt-get install -y \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        libgl1 \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && echo "✅ System dependencies installed."

# Install pip for Python 3.11
RUN echo "📦 Installing pip for Python 3.11..." \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && echo "✅ pip installed."

# Install PyTorch and other dependencies
RUN echo "📦 Installing Python dependencies..." \
    && pip install \
        torch==2.5.0 \
        transformers==4.44.0 \
        accelerate==0.33.0 \
        gradio>=4.44.1 \
        safetensors==0.4.4 \
        pillow==10.3.0 \
        invisible_watermark==0.2.0 \
        huggingface_hub[hf_transfer]==0.26.2 \
        sentencepiece==0.2.0 \
        numpy<2 \
    && echo "✅ Python dependencies installed."

# Clone and install diffusers from GitHub
RUN echo "📦 Cloning and installing diffusers from GitHub..." \
    && git clone https://github.com/huggingface/diffusers.git /tmp/diffusers \
    && cd /tmp/diffusers \
    && pip install . \
    && rm -rf /tmp/diffusers \
    && echo "✅ diffusers installed from GitHub."

# Set up cache directories for torch.compile
RUN echo "📦 Setting up cache directories for torch.compile..." \
    && mkdir -p /root/.inductor-cache /root/.nv /root/.triton \
    && echo "✅ Cache directories created."

# Set environment variables for torch.compile
ENV TORCHINDUCTOR_CACHE_DIR=/root/.inductor-cache
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Debug: Print completion message
RUN echo "🎉 Docker build completed successfully!"

# Set the entrypoint
ENTRYPOINT ["python3.11"]