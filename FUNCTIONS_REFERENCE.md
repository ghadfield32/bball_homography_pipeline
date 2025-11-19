# Functions and Code Reference - Docker Build Optimization

**Date**: 2025-10-08
**Purpose**: Quick reference for all functions, scripts, and code changes

---

## Table of Contents

1. [Python Functions (fix_docker_build.py)](#python-functions)
2. [Bash Functions (build_optimized.sh)](#bash-functions)
3. [Dockerfile Changes](#dockerfile-changes)
4. [Configuration Changes](#configuration-changes)

---

## Python Functions

### File: [fix_docker_build.py](fix_docker_build.py)

---

#### `check_wsl_environment() -> Dict[str, any]`

**Purpose**: Detect WSL environment and check for Windows mount issues

**Returns**:
```python
{
    "is_wsl": bool,              # True if running in WSL
    "wsl_version": None,         # Reserved for future use
    "on_windows_mount": bool,    # CRITICAL: True if building from /mnt/
    "current_path": str,         # Current working directory
    "docker_available": bool,    # True if Docker in PATH
    "buildx_available": bool     # True if BuildX available
}
```

**Key Logic**:
```python
# Check if on Windows mount - PRIMARY ISSUE DETECTOR
current_path = Path.cwd()
env_info["on_windows_mount"] = str(current_path).startswith("/mnt/")

if env_info["on_windows_mount"]:
    # This is the PRIMARY cause of slow builds and tar errors
    print("WARNING: Building from Windows mount")
```

**Usage**:
```python
env_info = check_wsl_environment()
if env_info["on_windows_mount"]:
    print("ERROR: Must move to WSL filesystem")
    exit(1)
```

---

#### `check_disk_space() -> Dict[str, any]`

**Purpose**: Verify sufficient disk space for Docker builds

**Returns**:
```python
{
    "root_available_gb": float,  # GB available on root filesystem
    "docker_data_gb": float,     # GB used by Docker
    "sufficient_space": bool     # True if >20GB available
}
```

**Key Logic**:
```python
# Parse df output
ok, out, err = run_command(["df", "-BG", "/"])
available_gb = float(parts[3].rstrip('G'))

# Check threshold
if available_gb < 20:
    space_info["sufficient_space"] = False
    print("WARNING: Less than 20GB available")
```

**Usage**:
```python
space_info = check_disk_space()
if not space_info["sufficient_space"]:
    print(f"ERROR: Only {space_info['root_available_gb']:.1f}GB available")
```

---

#### `analyze_buildkit_cache() -> Dict[str, any]`

**Purpose**: Check BuildKit cache state and identify corruption

**Returns**:
```python
{
    "cache_size_gb": float,   # Size of BuildKit cache in GB
    "cache_exists": bool,     # True if cache data found
    "needs_cleanup": bool     # True if cleanup recommended
}
```

**Key Logic**:
```python
# Check BuildKit disk usage
ok, out, err = run_command(["docker", "buildx", "du"])

# Heuristic: if cache output is long, probably needs cleanup
if 'GB' in out and len(out.split('\n')) > 10:
    cache_info["needs_cleanup"] = True
```

**Usage**:
```python
cache_info = analyze_buildkit_cache()
if cache_info["needs_cleanup"]:
    run_command(["docker", "buildx", "prune", "-af"])
```

---

#### `check_dockerignore() -> Dict[str, any]`

**Purpose**: Validate .dockerignore has essential patterns

**Returns**:
```python
{
    "exists": bool,                  # True if .dockerignore found
    "has_essential_patterns": bool,  # True if >60% patterns present
    "missing_patterns": List[str]    # List of missing patterns
}
```

**Key Logic**:
```python
essential_patterns = [
    ".git", "__pycache__", "*.pyc", ".venv",
    "node_modules", "videos", "data", "models", "weights"
]

# Check for at least 60% of essential patterns
ignore_info["has_essential_patterns"] = \
    len(found_patterns) >= len(essential_patterns) * 0.6
```

**Usage**:
```python
ignore_info = check_dockerignore()
if not ignore_info["has_essential_patterns"]:
    create_enhanced_dockerignore()
```

---

#### `check_build_context_size() -> Dict[str, any]`

**Purpose**: Measure build context size and identify large files

**Returns**:
```python
{
    "estimated_size_mb": int,           # Total context size in MB
    "large_files": List[Tuple[int, str]], # [(size_mb, filename), ...]
    "context_too_large": bool           # True if >100MB
}
```

**Key Logic**:
```python
# Get size of current directory
ok, out, err = run_command(["du", "-sm", "."], timeout=120)
size_mb = int(out.split()[0])

# Check threshold
if size_mb > 100:
    context_info["context_too_large"] = True
```

**Usage**:
```python
context_info = check_build_context_size()
if context_info["context_too_large"]:
    print(f"WARNING: Context is {context_info['estimated_size_mb']}MB")
    # Show largest files
    for size, name in context_info["large_files"][:10]:
        print(f"  {size:>6} MB  {name}")
```

---

#### `get_fix_recommendations() -> List[Dict]`

**Purpose**: Generate prioritized fix recommendations based on diagnostics

**Parameters**:
```python
def get_fix_recommendations(
    env_info: Dict,
    space_info: Dict,
    cache_info: Dict,
    ignore_info: Dict,
    context_info: Dict
) -> List[Dict]
```

**Returns**:
```python
[
    {
        "priority": str,      # "🔴 CRITICAL", "🟡 HIGH", "🟢 MEDIUM"
        "issue": str,         # Description of issue
        "impact": str,        # What this causes
        "fix": str,           # How to fix
        "commands": List[str] # Shell commands to run
    },
    ...
]
```

**Priority Logic**:
```python
# Priority 1: Windows mount (CRITICAL)
if env_info["on_windows_mount"]:
    recommendations.append({
        "priority": "🔴 CRITICAL",
        "issue": "Building from Windows mount",
        "impact": "10-50x slower build, tar corruption",
        "fix": "Move project to WSL filesystem"
    })

# Priority 2: Low disk space (CRITICAL)
if not space_info["sufficient_space"]:
    recommendations.append({
        "priority": "🔴 CRITICAL",
        "issue": "Low disk space"
    })

# Priority 3: BuildKit cache (HIGH)
if cache_info["needs_cleanup"]:
    recommendations.append({
        "priority": "🟡 HIGH",
        "issue": "Large BuildKit cache"
    })
```

**Usage**:
```python
recommendations = get_fix_recommendations(
    env_info, space_info, cache_info, ignore_info, context_info
)

for i, rec in enumerate(recommendations, 1):
    print(f"\n{rec['priority']} Recommendation {i}:")
    print(f"  Issue: {rec['issue']}")
    print(f"  Fix: {rec['fix']}")
```

---

#### `apply_automatic_fixes(args) -> bool`

**Purpose**: Apply selected automated fixes

**Parameters**:
```python
args: argparse.Namespace  # Command line arguments
```

**Returns**:
```python
bool  # True if any fixes were applied
```

**Fixes Applied**:
```python
# Fix 1: Create/update .dockerignore
if args.fix_dockerignore or args.fix_all:
    create_enhanced_dockerignore()

# Fix 2: Clean BuildKit cache
if args.clean_cache or args.fix_all:
    run_command(["docker", "buildx", "prune", "-af"])

# Fix 3: Clean Docker system
if args.clean_docker or args.fix_all:
    run_command(["docker", "system", "prune", "-af"])

# Fix 4: Recreate BuildKit builder
if args.recreate_builder or args.fix_all:
    run_command(["docker", "buildx", "rm", "bball-builder"])
    run_command(["docker", "buildx", "create", "--use", "--name", "bball-builder"])
```

**Usage**:
```bash
python3 fix_docker_build.py --fix-all
python3 fix_docker_build.py --clean-cache
python3 fix_docker_build.py --fix-dockerignore
```

---

#### `create_enhanced_dockerignore() -> bool`

**Purpose**: Create optimized .dockerignore file

**Returns**:
```python
bool  # True on success
```

**Content Created**:
```
# Git files (4 patterns)
.git, .gitignore, .gitmodules, .gitattributes

# Python cache (14 patterns)
__pycache__/, *.py[cod], *.egg-info/, build/, dist/, ...

# Large data directories (8 patterns)
data/, datasets/, models/, weights/, videos/, outputs/, results/, experiments/

# ML/CV specific (4 patterns)
mlruns/, wandb/, .roboflow/, yolo*/

# Keep critical files (8 exceptions)
!pyproject.toml, !uv.lock, !.devcontainer/Dockerfile, ...
```

**Usage**:
```python
if create_enhanced_dockerignore():
    print("Enhanced .dockerignore created")
```

---

#### `generate_build_script() -> None`

**Purpose**: Generate optimized bash build script

**Script Generated**: `build_optimized.sh`

**Features**:
- Prerequisite checking
- Interactive cleanup option
- Optimized build command
- Automated testing
- Summary reporting

**Usage**:
```python
generate_build_script()
# Creates: build_optimized.sh
```

---

## Bash Functions

### File: [build_optimized.sh](build_optimized.sh)

---

#### `print_header(title: string) -> void`

**Purpose**: Print formatted section headers

**Parameters**:
```bash
title: string  # Title text to display
```

**Output**:
```
========================================
Title Text Here
========================================
```

**Usage**:
```bash
print_header "Checking Prerequisites"
```

---

#### `check_prerequisites() -> void`

**Purpose**: Verify environment before building

**Checks Performed**:
1. Not on Windows mount (`/mnt/c`)
2. Docker available
3. BuildX available
4. Dockerfile exists
5. Disk space >20GB

**Exits if**:
- Running from `/mnt/c`
- Less than 20GB disk space
- Docker/BuildX not available

**Usage**:
```bash
check_prerequisites
# Exits with error if any check fails
```

---

#### `clean_previous_builds() -> void`

**Purpose**: Optionally clean previous builds

**Interactive**: Prompts user for confirmation

**Actions if confirmed**:
```bash
# Stop compose services
docker compose -f .devcontainer/docker-compose.yml down --volumes

# Remove containers
docker rm -f ${CONTAINER_NAMES}

# Remove images
docker rmi ${IMAGE_NAME}:${IMAGE_TAG}

# Clean caches
docker buildx prune -af
docker system prune -af
```

**Usage**:
```bash
clean_previous_builds
# Prompts: "Clean previous builds? (y/N): "
```

---

#### `start_build() -> void`

**Purpose**: Execute optimized Docker build

**Environment Set**:
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

**Build Command**:
```bash
docker buildx build \
  --progress=plain \           # Show detailed output
  --no-cache \                 # Force fresh build
  --output=type=docker \       # Output to Docker daemon
  --sbom=false \              # Disable SBOM generation
  --provenance=false \        # Disable provenance
  --attest=type=none \        # Disable attestation
  --tag ${IMAGE_NAME}:${IMAGE_TAG} \
  --file ${DOCKERFILE_PATH} \
  ${BUILD_CONTEXT}
```

**Timing**: Records and reports build time

**Usage**:
```bash
start_build
# Builds image and reports time taken
```

---

#### `test_image() -> void`

**Purpose**: Run automated tests on built image

**Tests Performed**:
```bash
# 1. Python availability
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python --version

# 2. UV package manager
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} uv --version

# 3. PyTorch
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
  python -c "import torch; print('PyTorch:', torch.__version__)"

# 4. JAX
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
  python -c "import jax; print('JAX:', jax.__version__)"

# 5. OpenCV
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
  python -c "import cv2; print('OpenCV:', cv2.__version__)"

# 6. YOLO
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
  python -c "from ultralytics import YOLO; print('YOLO: OK')"

# 7. GPU (optional)
docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} \
  python -c "import torch; assert torch.cuda.is_available()"
```

**Exits if**: Any critical test fails

**Usage**:
```bash
test_image
# Runs all tests and reports results
```

---

#### `show_summary() -> void`

**Purpose**: Display build summary and next steps

**Information Shown**:
- Image name and tag
- Image size
- Dockerfile used
- Build time
- Next steps

**Usage**:
```bash
show_summary
# Displays summary at end of build
```

---

## Dockerfile Changes

### File: [.devcontainer/Dockerfile.optimized](.devcontainer/Dockerfile.optimized)

---

### Separate Cache IDs

**Purpose**: Prevent cache corruption by isolating package groups

**Before** (Shared Cache):
```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision torchaudio
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install jax jaxlib
```

**After** (Separate Caches):
```dockerfile
# PyTorch - separate cache
RUN --mount=type=cache,id=uv-pytorch-${CUDA_TAG}-v2,target=/root/.cache/uv,sharing=locked \
    uv pip install torch torchvision torchaudio

# JAX - separate cache
RUN --mount=type=cache,id=uv-jax-${CUDA_TAG}-v2,target=/root/.cache/uv,sharing=locked \
    uv pip install "jax[cuda12-local]>=0.4.26"

# CV - separate cache
RUN --mount=type=cache,id=uv-cv-${CUDA_TAG}-v2,target=/root/.cache/uv,sharing=locked \
    uv pip install ultralytics opencv-contrib-python-headless
```

**Cache IDs**:
| ID | Purpose |
|----|---------|
| `apt-cache-${CUDA_TAG}-v2` | System packages |
| `uv-resolve-${CUDA_TAG}-v2` | Dependency resolution |
| `uv-sync-${CUDA_TAG}-v2` | Core packages |
| `uv-pytorch-${CUDA_TAG}-v2` | PyTorch stack |
| `uv-cudnn-${CUDA_TAG}-v2` | CuDNN |
| `uv-nvjit-${CUDA_TAG}-v2` | NVJITLINK |
| `uv-jax-${CUDA_TAG}-v2` | JAX stack |
| `uv-cv-${CUDA_TAG}-v2` | Computer vision |
| `uv-jupyter-${CUDA_TAG}-v2` | Jupyter |

---

### Timestamp Logging

**Purpose**: Track build progress and identify slow steps

**Pattern**:
```dockerfile
RUN echo "[$(date +%T)] STEP: Description..." && \
    command_here && \
    echo "[$(date +%T)] ✅ Step completed"
```

**Examples**:
```dockerfile
# System dependencies
RUN echo "[$(date +%T)] STEP: Installing system dependencies..." && \
    apt-get update && apt-get install -y packages && \
    echo "[$(date +%T)] ✅ System dependencies installed"

# PyTorch
RUN echo "[$(date +%T)] STEP: Installing PyTorch with CUDA 12.4..." && \
    uv pip install torch torchvision torchaudio && \
    echo "[$(date +%T)] ✅ PyTorch installed"
```

**Output Example**:
```
[14:23:45] STEP: Installing PyTorch with CUDA 12.4...
[14:28:12] ✅ PyTorch installed
```

---

### Validation Checks

**Purpose**: Verify installations and catch errors early

**Pattern**:
```dockerfile
RUN echo "[$(date +%T)] STEP: Validating installation..." && \
    python - <<'PYEOF'
import sys
try:
    import package
    print(f"✅ Package: {package.__version__}")
except Exception as e:
    print(f"❌ Package failed: {e}")
    sys.exit(1)
PYEOF
```

**Validations Added**:

1. **Essential Commands**:
```dockerfile
RUN which groups dircolors uname || echo "WARNING: Some commands missing"
```

2. **Python Environment**:
```dockerfile
RUN bash -c "source /app/.venv/bin/activate && \
             python --version && \
             which python && \
             pip --version"
```

3. **PyTorch**:
```dockerfile
RUN python - <<'PYEOF'
import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
PYEOF
```

4. **JAX**:
```dockerfile
RUN python - <<'PYEOF'
import jax, jaxlib
print(f"✅ JAX version: {jax.__version__}")
print(f"✅ JAX devices: {jax.devices()}")
PYEOF
```

5. **Computer Vision**:
```dockerfile
RUN python - <<'PYEOF'
import cv2
from ultralytics import YOLO
import roboflow
import supervision as sv
print(f"✅ OpenCV: {cv2.__version__}")
print("✅ YOLO: Available")
print(f"✅ Roboflow: {roboflow.__version__}")
print(f"✅ Supervision: {sv.__version__}")
PYEOF
```

---

## Configuration Changes

### File: [.dockerignore](.dockerignore)

**Before**: 24 lines, basic patterns
**After**: 153 lines, comprehensive patterns

**Key Additions**:

1. **Large Data Directories** (CRITICAL):
```
data/
datasets/
models/
weights/
videos/
outputs/
results/
experiments/
```

2. **ML/CV Specific**:
```
mlruns/
wandb/
.roboflow/
yolo*/
```

3. **Python Build Artifacts**:
```
__pycache__/
*.egg-info/
build/
dist/
wheels/
```

4. **Large Binaries**:
```
*.mp4
*.avi
*.mov
*.mkv
*.zip
*.tar
*.tar.gz
```

5. **Keep Critical Files**:
```
!pyproject.toml
!uv.lock
!.devcontainer/Dockerfile
!.devcontainer/Dockerfile.optimized
!.devcontainer/tests/
```

---

## Quick Reference

### Run Diagnostics
```bash
python3 fix_docker_build.py
```

### Apply All Fixes
```bash
python3 fix_docker_build.py --fix-all
```

### Build Optimized Image
```bash
./build_optimized.sh
```

### Manual Build
```bash
docker buildx build \
  --progress=plain \
  --no-cache \
  --output=type=docker \
  --sbom=false \
  --provenance=false \
  -f .devcontainer/Dockerfile.optimized \
  -t bball_homography_pipeline_env_datascience:dev .
```

---

**Last Updated**: 2025-10-08
**Version**: 1.0
**Status**: ✅ Complete Reference
