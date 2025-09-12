# Dockerfile: RTX 4090 devcontainer with UV, JAX, and PyTorch (CUDA 12.x)

ARG CUDA_TAG=12.4.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=docker_dev_template
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies with Computer Vision additions
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates git procps htop \
        python3 python3-venv python3-pip python3-dev \
        build-essential cmake pkg-config \
        libjemalloc2 libjemalloc-dev \
        iproute2 net-tools lsof wget \
        # Computer Vision dependencies
        ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 libgtk-3-dev \
        # X11 support for GUI applications
        x11-apps xauth xvfb \
        # Video codec libraries
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        # Image format libraries
        libjpeg-dev libpng-dev libtiff-dev \
        # OpenGL support for visualization
        libgl1-mesa-glx libglu1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV package manager
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

WORKDIR /app

# Create venv managed by UV
RUN uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}"

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH="/workspace"

# Memory and allocator settings
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TCACHE_MAX=0 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

# GPU‑relevant environment
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True \
    JAX_PREALLOCATION_SIZE_LIMIT_BYTES=17179869184

# Computer Vision specific environment variables
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:0

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/weights \
    && chmod 755 /app/models /app/data /app/weights

# Bring in project descriptors and tests
COPY pyproject.toml /workspace/
COPY uv.lock* /workspace/
COPY .devcontainer/validate_gpu.py /app/validate_gpu.py
COPY .devcontainer/tests/ /app/tests/

# Resolve project dependencies with UV
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    cd /workspace && (uv sync --frozen --no-dev || (uv sync --no-dev && uv lock))

# CRITICAL FIX 2: Install PyTorch first to establish CUDA environment
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Installing PyTorch with CUDA 12.4..." && \
    uv pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# CRITICAL FIX 3: Install compatible CuDNN 9.8.0 to satisfy JAX requirements
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Upgrading CuDNN to 9.8.0 for JAX compatibility..." && \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12==9.8.0.69 || \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12>=9.8.0

# CRITICAL FIX 4: Install JAX after CuDNN upgrade with proper dependency resolution
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Removing any existing JAX installations..." && \
    (uv pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true) && \
    echo "Installing JAX with CUDA 12 support..." && \
    (uv pip install --no-cache-dir "jax[cuda12-local]>=0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
     || uv pip install --no-cache-dir "jax[cpu]>=0.4.26")

# Install Computer Vision specific packages
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Installing Computer Vision packages..." && \
    # Ensure latest Ultralytics and dependencies
    uv pip install --no-cache-dir --upgrade ultralytics==8.3.158 && \
    # Install tracking libraries
    uv pip install --no-cache-dir supervision>=0.17.0 lap>=0.4.0 && \
    # Install additional CV utilities
    uv pip install --no-cache-dir albumentations>=1.3.0 && \
    # Ensure opencv-contrib for advanced features
    uv pip install --no-cache-dir opencv-contrib-python-headless>=4.10.0

# REMOVED: Pre-download YOLO models (models will be downloaded on first use)
# This reduces build time and container size significantly
# Models will be cached in the mounted volumes during runtime

# Jupyter kernel support
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install ipykernel jupyter-client jupyterlab

# CUDA libs in path - include both system and package CUDA libraries
ENV LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Shell activation helper with updated library paths
RUN echo '#!/bin/bash' > /app/activate_uv.sh && \
    echo 'export VIRTUAL_ENV="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PATH="/app/.venv/bin:$PATH"' >> /app/activate_uv.sh && \
    echo 'export UV_PROJECT_ENVIRONMENT="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /app/activate_uv.sh && \
    echo 'export LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"' >> /app/activate_uv.sh && \
    echo '# Computer Vision environment' >> /app/activate_uv.sh && \
    echo 'export YOLO_VERBOSE=false' >> /app/activate_uv.sh && \
    echo 'export OPENCV_LOG_LEVEL=ERROR' >> /app/activate_uv.sh && \
    echo 'cd /workspace' >> /app/activate_uv.sh && \
    chmod +x /app/activate_uv.sh && \
    echo 'source /app/activate_uv.sh' > /etc/profile.d/10-uv-activate.sh && \
    echo 'source /app/activate_uv.sh' >> /root/.bashrc && \
    chmod +x /etc/profile.d/10-uv-activate.sh

# Enhanced healthcheck with CV components (updated without YOLO model check)
RUN echo '#!/bin/bash' > /app/healthcheck.sh && \
    echo 'source /app/.venv/bin/activate' >> /app/healthcheck.sh && \
    echo 'echo "=== Environment Check ==="' >> /app/healthcheck.sh && \
    echo 'python --version' >> /app/healthcheck.sh && \
    echo 'echo "=== CuDNN Version Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import torch; print(f\"PyTorch CuDNN: {torch.backends.cudnn.version()}\")" || echo "PyTorch CuDNN check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== JAX Device Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import jax; print(f\"JAX devices: {jax.devices()}\")" || echo "JAX device check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== Computer Vision Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import cv2; print(f\"OpenCV: {cv2.__version__}\")" || echo "OpenCV check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "from ultralytics import YOLO; print(\"YOLO package: OK\")" || echo "YOLO package check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "import roboflow; print(f\"Roboflow: {roboflow.__version__}\")" || echo "Roboflow check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== GPU Validation ==="' >> /app/healthcheck.sh && \
    echo 'python /app/validate_gpu.py --quick' >> /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

WORKDIR /workspace
CMD ["bash", "-l"]
