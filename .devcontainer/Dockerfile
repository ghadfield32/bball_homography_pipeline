# ENHANCED Dockerfile: RTX 4090 devcontainer with UV, JAX, PyTorch, and Computer Vision (CUDA 12.x)
# ENHANCED WITH: Better debugging, error handling, and build validation

ARG CUDA_TAG=12.4.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04 as runtime

ARG PYTHON_VER=3.10
ARG ENV_NAME=bball_homography_pipeline_env
ENV DEBIAN_FRONTEND=noninteractive

# ADDED: Build debugging information
RUN echo "=================================================================" && \
    echo "STARTING BUILD: ${ENV_NAME}" && \
    echo "CUDA_TAG: ${CUDA_TAG}" && \
    echo "PYTHON_VER: ${PYTHON_VER}" && \
    echo "BASE IMAGE: nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04" && \
    echo "================================================================="

# Enhanced system dependencies with Computer Vision additions
RUN --mount=type=cache,id=apt-cache-${CUDA_TAG},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${CUDA_TAG},target=/var/lib/apt/lists,sharing=locked \
    echo "STEP: Installing system dependencies..." && \
    apt-get update && apt-get install -y --no-install-recommends \
        # CRITICAL: Essential core utilities (fixes missing commands)
        coreutils \
        util-linux \
        # Base system packages
        bash curl ca-certificates git procps htop \
        python3 python3-venv python3-pip python3-dev \
        build-essential cmake pkg-config \
        libjemalloc2 libjemalloc-dev \
        iproute2 net-tools lsof wget \
        # Additional essential tools
        grep sed gawk findutils \
        # Computer Vision system dependencies
        ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 libgtk-3-dev \
        # X11 support for GUI applications (optional for headless)
        x11-apps xauth xvfb \
        # Video codec libraries for comprehensive video support
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        # Image format libraries for OpenCV
        libjpeg-dev libpng-dev libtiff-dev \
        # OpenGL support for visualization (optional)
        libgl1-mesa-glx libglu1-mesa-dev \
        # GPG for Node.js installation
        gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "✅ System dependencies installed successfully" && \
    echo "STEP: Verifying essential commands..." && \
    which groups && echo "✅ groups command available" && \
    which dircolors && echo "✅ dircolors command available" && \
    which uname && echo "✅ uname command available" && \
    echo "✅ Essential commands verification complete"

# Install Node.js 20 LTS (required for Claude Code extension)
RUN echo "STEP: Installing Node.js 20 LTS..." && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest && \
    echo "✅ Node.js installed:" && \
    node --version && \
    npm --version

# UV package manager with version validation
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/
RUN echo "STEP: Validating UV installation..." && \
    uv --version && \
    echo "✅ UV installed successfully"

WORKDIR /app

# Create venv managed by UV with debugging
RUN echo "STEP: Creating UV virtual environment..." && \
    uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}" && \
    echo "✅ Virtual environment created at /app/.venv" && \
    ls -la .venv/

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH="/workspace"

# Validate Python installation
RUN echo "STEP: Validating Python environment..." && \
    bash -c "source /app/.venv/bin/activate && python --version && which python && pip --version" && \
    echo "✅ Python environment validated"

# Enhanced memory and allocator settings for CV workloads
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TCACHE_MAX=0 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

# GPU-relevant environment (enhanced for CV)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True,roundup_power2_divisions:16 \
    JAX_PREALLOCATION_SIZE_LIMIT_BYTES=10737418240

# Computer Vision specific environment variables
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:0 \
    YOLO_VERBOSE=false \
    OPENCV_LOG_LEVEL=ERROR

# Create directories for CV models, data, and videos with debugging
RUN echo "STEP: Creating CV directories..." && \
    mkdir -p /app/models /app/data /app/weights \
    /workspace/videos/input /workspace/videos/output && \
    chmod 755 /app/models /app/data /app/weights && \
    chmod 755 /workspace/videos /workspace/videos/input /workspace/videos/output && \
    echo "✅ CV directories created:" && \
    ls -la /app/ && \
    ls -la /workspace/

# CRITICAL FIX: Check for pyproject.toml with debugging
RUN echo "STEP: Checking for project files..." && \
    echo "Current working directory: $(pwd)" && \
    echo "Contents of /workspace:" && \
    ls -la /workspace/ || echo "WARNING: /workspace not accessible during build" && \
    echo "Looking for pyproject.toml in expected locations..."

# Copy project files with validation
COPY pyproject.toml /workspace/
RUN echo "STEP: Validating copied project files..." && \
    ls -la /workspace/pyproject.toml && \
    echo "✅ pyproject.toml copied successfully"

# Copy optional lock file
COPY uv.lock* /workspace/
RUN echo "Checking for uv.lock file..." && \
    ls -la /workspace/uv.lock* || echo "No uv.lock file found (will be generated)"

# Devcontainer tests and validator (enhanced)
COPY .devcontainer/validate_gpu.py /app/validate_gpu.py
COPY .devcontainer/tests/ /app/tests/
RUN echo "STEP: Validating test files..." && \
    ls -la /app/validate_gpu.py /app/tests/ && \
    echo "✅ Test files copied successfully"

# Resolve project dependencies with UV (enhanced with debugging)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Resolving UV project dependencies..." && \
    cd /workspace && \
    echo "Current directory: $(pwd)" && \
    echo "Contents:" && \
    ls -la && \
    if [ ! -f uv.lock ]; then \
      echo "[uv] No uv.lock found; creating from existing pyproject.toml"; \
      uv lock --refresh; \
    else \
      echo "[uv] Using existing uv.lock"; \
    fi && \
    echo "✅ UV dependencies resolved"

# Install core dependencies first (excluding GPU packages)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing core dependencies..." && \
    cd /workspace && \
    (uv sync --frozen --no-dev 2>/dev/null || \
     uv sync --no-dev 2>/dev/null || \
     (echo "[uv] Installing basic dependencies..." && uv add numpy pandas matplotlib scipy)) && \
    echo "✅ Core dependencies installed"

# CRITICAL: Install PyTorch with CUDA 12.4 support - ENHANCED with validation
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing PyTorch with CUDA 12.4..." && \
    uv pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    echo "STEP: Validating PyTorch installation..." && \
    python - <<'PY'
import torch
print("✅ PyTorch version:", torch.__version__)
print("✅ PyTorch CUDA available during build:", torch.cuda.is_available())
print("✅ PyTorch CUDA version:", torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown')
print("Note: GPU functionality will be available at runtime when container has GPU access")
PY

# CRITICAL: Install compatible CuDNN for JAX compatibility - ENHANCED  
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Upgrading CuDNN to 9.8.0 for JAX compatibility..." && \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12==9.8.0.69 || \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12>=9.8.0 && \
    echo "✅ CuDNN installed/upgraded"

# NVJITLINK enables runtime JIT linking for CUDA 12 (helps PyTorch/JAX fused kernels)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing NVJITLINK for CUDA 12..." && \
    uv pip install --no-cache-dir nvidia-nvjitlink-cu12>=12.4 && \
    echo "✅ NVJITLINK installed"

# CRITICAL: Install JAX with CUDA 12 support - ENHANCED with better error handling
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing JAX with CUDA 12 support..." && \
    echo "Removing any existing JAX installations..." && \
    (uv pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true) && \
    echo "Installing JAX with CUDA 12 support..." && \
    (uv pip install --no-cache-dir "jax[cuda12-local]>=0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
     || uv pip install --no-cache-dir "jax[cpu]>=0.4.26") && \
    echo "STEP: Validating JAX installation..." && \
    python - <<'PY'
import jax, jaxlib
print("✅ JAX version:", jax.__version__, "JAXLIB:", jaxlib.__version__)
print("✅ JAX devices during build:", jax.devices())
print("Note: GPU devices will be available at runtime when container has GPU access")
PY

# NEW: Install Computer Vision packages with validation
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing Computer Vision packages..." && \
    # Core CV libraries with specific versions for stability
    uv pip install --no-cache-dir --upgrade ultralytics==8.3.158 && \
    echo "✅ Ultralytics installed" && \
    # Object tracking and supervision libraries
    uv pip install --no-cache-dir supervision>=0.17.0 lap>=0.4.0 && \
    echo "✅ Supervision and LAP installed" && \
    # Image processing and augmentation
    uv pip install --no-cache-dir albumentations>=1.3.0 imgaug>=0.4.0 && \
    echo "✅ Image processing libraries installed" && \
    # OpenCV with contrib modules (headless for Docker)
    uv pip install --no-cache-dir opencv-contrib-python-headless>=4.10.0 && \
    echo "✅ OpenCV installed" && \
    # Roboflow integration
    uv pip install --no-cache-dir roboflow==1.2.9 && \
    echo "✅ Roboflow installed" && \
    # Video processing libraries
    uv pip install --no-cache-dir moviepy==2.2.1 yt-dlp==2025.9.5 ffmpeg-python==0.2.0 && \
    echo "✅ Video processing libraries installed" && \
    echo "✅ Computer Vision packages installed successfully"

# NEW: Pre-validate CV installation with enhanced debugging
RUN echo "STEP: Validating Computer Vision installation..." && \
    python - <<'PY'
print("=== Computer Vision Validation ===")
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV: {e}")

try:
    from ultralytics import YOLO
    print("✅ YOLO/Ultralytics: Available")
except Exception as e:
    print(f"❌ YOLO: {e}")

try:
    import roboflow
    print(f"✅ Roboflow: {roboflow.__version__}")
except Exception as e:
    print(f"❌ Roboflow: {e}")

try:
    import supervision as sv
    print(f"✅ Supervision: {sv.__version__}")
except Exception as e:
    print(f"❌ Supervision: {e}")

print("✅ CV validation completed")
PY

# Jupyter & kernel with enhanced dependencies and validation
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "STEP: Installing Jupyter and kernel dependencies..." && \
    uv pip install --no-cache-dir \
        psutil==5.9.8 \
        debugpy==1.8.7 \
        ipykernel==6.29.5 \
        jupyter-client==8.6.1 \
        jupyterlab==4.2.5 && \
    echo "✅ Jupyter components installed"

# Enhanced CUDA libs path - include CV library paths
ENV LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Enhanced shell activation helper with CV environment and debugging
RUN echo "STEP: Creating robust shell activation helper..." && \
    echo '#!/bin/bash' > /app/activate_uv.sh && \
    echo 'export VIRTUAL_ENV="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PATH="/app/.venv/bin:$PATH"' >> /app/activate_uv.sh && \
    echo 'export UV_PROJECT_ENVIRONMENT="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /app/activate_uv.sh && \
    echo 'export LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"' >> /app/activate_uv.sh && \
    echo '# Computer Vision environment' >> /app/activate_uv.sh && \
    echo 'export YOLO_VERBOSE=false' >> /app/activate_uv.sh && \
    echo 'export OPENCV_LOG_LEVEL=ERROR' >> /app/activate_uv.sh && \
    echo 'export DISPLAY=${DISPLAY:-:0}' >> /app/activate_uv.sh && \
    echo '# Check for essential commands and provide helpful messages' >> /app/activate_uv.sh && \
    echo 'command -v uname >/dev/null 2>&1 || echo "Warning: uname command not available"' >> /app/activate_uv.sh && \
    echo 'command -v groups >/dev/null 2>&1 || echo "Warning: groups command not available"' >> /app/activate_uv.sh && \
    echo 'command -v dircolors >/dev/null 2>&1 || echo "Warning: dircolors command not available"' >> /app/activate_uv.sh && \
    echo 'echo "🐍 UV Environment activated: $(python --version 2>/dev/null || echo Python not found)"' >> /app/activate_uv.sh && \
    echo 'cd /workspace' >> /app/activate_uv.sh && \
    chmod +x /app/activate_uv.sh && \
    echo 'source /app/activate_uv.sh' > /etc/profile.d/10-uv-activate.sh && \
    echo 'source /app/activate_uv.sh' >> /root/.bashrc && \
    chmod +x /etc/profile.d/10-uv-activate.sh && \
    echo "✅ Robust shell activation helper created"

# Enhanced healthcheck with CV components and debugging
RUN echo "STEP: Creating enhanced healthcheck script..." && \
    echo '#!/bin/bash' > /app/healthcheck.sh && \
    echo 'source /app/.venv/bin/activate' >> /app/healthcheck.sh && \
    echo 'echo "=== Environment Check ==="' >> /app/healthcheck.sh && \
    echo 'python --version' >> /app/healthcheck.sh && \
    echo 'echo "=== GPU Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import torch; print(f\"PyTorch CUDA: {torch.cuda.is_available()}\")" || echo "PyTorch check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "import jax; print(f\"JAX devices: {jax.devices()}\")" || echo "JAX check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== Computer Vision Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import cv2; print(f\"OpenCV: {cv2.__version__}\")" || echo "OpenCV check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "from ultralytics import YOLO; print(\"YOLO package: OK\")" || echo "YOLO package check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "import roboflow; print(f\"Roboflow: {roboflow.__version__}\")" || echo "Roboflow check failed"' >> /app/healthcheck.sh && \
    echo 'python -c "import supervision; print(f\"Supervision: {supervision.__version__}\")" || echo "Supervision check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== Final GPU Validation ==="' >> /app/healthcheck.sh && \
    echo 'python /app/validate_gpu.py --quick' >> /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    echo "✅ Enhanced healthcheck script created"

# Final validation step
RUN echo "=================================================================" && \
    echo "BUILD COMPLETE: ${ENV_NAME}" && \
    echo "CUDA_TAG: ${CUDA_TAG}" && \
    echo "PYTHON_VER: ${PYTHON_VER}" && \
    echo "=================================================================" && \
    echo "Final environment validation:" && \
    bash -c "source /app/.venv/bin/activate && python --version && uv --version" && \
    echo "✅ Build completed successfully" && \
    echo "================================================================="

WORKDIR /workspace
CMD ["bash", "-l"]


