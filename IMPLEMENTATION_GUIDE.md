# Docker Build Optimization - Implementation Guide

**Last Updated**: 2025-10-08
**Estimated Time**: 1-2 hours
**Difficulty**: Intermediate

---

## Table of Contents

1. [Problem Summary](#problem-summary)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Rollback Plan](#rollback-plan)

---

## Problem Summary

### Current State
- **Build Time**: ~31 minutes (should be 5-10 minutes)
- **Error**: `archive/tar: invalid tar header` during layer export
- **Context Transfer**: 470 seconds for 1MB (should be <5 seconds)

### Root Cause
1. 🔴 **CRITICAL**: Building from Windows mount (`/mnt/c`) → 10-50x slower I/O
2. 🔴 **CRITICAL**: Corrupted BuildKit cache from previous failed builds
3. 🟡 **HIGH**: Large build context (missing/incomplete .dockerignore)
4. 🟡 **HIGH**: Single shared cache causing corruption during long builds

### Solution
- Move project to WSL filesystem
- Implement separate cache IDs per package group
- Enhanced .dockerignore to reduce context size
- Add comprehensive logging and validation

---

## Prerequisites

### Required Tools
- [ ] Windows 11 or Windows 10 with WSL2
- [ ] Docker Desktop (latest stable)
- [ ] Visual Studio Code with Remote-WSL extension
- [ ] At least 20GB free disk space

### Knowledge Required
- Basic Linux command line (cd, ls, cp)
- Basic Docker concepts (images, containers)
- Ability to edit files in VS Code

### Backup Checklist
- [ ] Current project code is committed to Git
- [ ] You have the URL of your remote repository
- [ ] You can access Docker Desktop

---

## Step-by-Step Implementation

### Phase 1: Diagnostic Analysis (15 minutes)

#### Step 1.1: Run Diagnostic Script

Open WSL terminal (in current location):

```bash
cd /mnt/c/Users/ghadf/vscode_projects/docker_projects/bball_homography_pipeline

# Make script executable
chmod +x fix_docker_build.py

# Run diagnostics
python3 fix_docker_build.py
```

**Expected Output:**
```
🔴 CRITICAL: Building from Windows mount (/mnt/c)
🟡 HIGH: Large build context (XXX MB)
🟡 HIGH: BuildKit cache needs cleanup
```

**Action**: Review the recommendations section. Take note of:
- Current disk space available
- Build context size
- Cache status

---

### Phase 2: Project Migration (20 minutes)

#### Step 2.1: Move Project to WSL Filesystem

**Why**: This is the MOST CRITICAL fix. Building from `/mnt/c` causes:
- 10-50x slower I/O
- Primary cause of tar corruption
- Context transfer taking 470s instead of <5s

**Commands**:

```bash
# Create projects directory in WSL
mkdir -p ~/projects

# Copy entire project to WSL
# (This may take 5-10 minutes depending on size)
cp -r /mnt/c/Users/ghadf/vscode_projects/docker_projects/bball_homography_pipeline ~/projects/

# Navigate to new location
cd ~/projects/bball_homography_pipeline

# Verify all files copied
ls -lah

# Check git status (should be clean)
git status
```

**Verification**:
```bash
# Current path should NOT start with /mnt/
pwd
# Should output: /home/<username>/projects/bball_homography_pipeline
```

#### Step 2.2: Open in VS Code from WSL

```bash
# From WSL terminal in new location
code .
```

**Alternative** (if `code .` doesn't work):
1. Open VS Code
2. Press `F1`
3. Type: "Remote-WSL: New WSL Window"
4. Navigate to: `~/projects/bball_homography_pipeline`

**Verification**:
- VS Code title bar should show: `bball_homography_pipeline [WSL: Ubuntu]`
- Terminal in VS Code should show path: `/home/<username>/projects/...`

---

### Phase 3: Clean Build Environment (15 minutes)

#### Step 3.1: Run Automated Cleanup

From the NEW location in WSL:

```bash
cd ~/projects/bball_homography_pipeline

# Run all automatic fixes
python3 fix_docker_build.py --fix-all
```

**This will**:
- ✅ Create enhanced .dockerignore
- ✅ Clean BuildKit cache
- ✅ Clean Docker system
- ✅ Recreate BuildKit builder
- ✅ Generate optimized build script

**Expected Duration**: 3-5 minutes

#### Step 3.2: Manual Cleanup (if automatic fails)

```bash
# Stop all containers
docker compose -f .devcontainer/docker-compose.yml down --volumes

# Remove project-specific containers
docker rm -f bball_homography_pipeline_env_datascience
docker rm -f bball_homography_pipeline_env_mlflow

# Clean BuildKit cache
docker buildx prune -af

# Clean Docker system
docker system prune -af --volumes

# Verify cleanup
docker system df
docker buildx du
```

#### Step 3.3: Restart Docker (IMPORTANT)

**In PowerShell (as Admin)**:
```powershell
# Shutdown WSL completely
wsl --shutdown

# Wait 10 seconds, then restart Docker Desktop
# (Click Start, search for Docker Desktop, click icon)
```

**Verification**:
```bash
# In WSL, verify Docker is running
docker ps
docker buildx ls
```

---

### Phase 4: File Updates (10 minutes)

#### Step 4.1: Verify Enhanced Files

Check that these files were created/updated:

```bash
cd ~/projects/bball_homography_pipeline

# Check .dockerignore
cat .dockerignore | head -20

# Check optimized Dockerfile
ls -lah .devcontainer/Dockerfile.optimized

# Check build script
ls -lah build_optimized.sh

# Check changelog
ls -lah CHANGELOG.md
```

#### Step 4.2: Review Changes

Open in VS Code:
- [.dockerignore](.dockerignore) - Should have comprehensive patterns
- [.devcontainer/Dockerfile.optimized](.devcontainer/Dockerfile.optimized) - Should have timestamp logging
- [CHANGELOG.md](CHANGELOG.md) - Should have today's date

**Key Changes to Verify**:

1. **.dockerignore** should exclude:
   - `data/`, `videos/`, `models/`, `weights/`
   - `.venv/`, `__pycache__/`, `.git/`
   - Large binaries (`.mp4`, `.zip`, etc.)

2. **Dockerfile.optimized** should have:
   - Separate cache IDs: `uv-pytorch-*`, `uv-jax-*`, `uv-cv-*`
   - Timestamp logging: `[$(date +%T)] STEP: ...`
   - Validation after each major installation

---

### Phase 5: First Optimized Build (30-60 minutes)

#### Step 5.1: Prepare for Build

```bash
cd ~/projects/bball_homography_pipeline

# Make build script executable
chmod +x build_optimized.sh

# Quick pre-flight check
python3 fix_docker_build.py
```

**Verify**:
- ✅ Not on Windows mount
- ✅ >20GB disk space
- ✅ BuildKit cache clean
- ✅ .dockerignore present

#### Step 5.2: Start Build

```bash
./build_optimized.sh
```

**What to expect**:

1. **Prerequisites check** (~30 seconds)
   - Verifies WSL filesystem
   - Checks Docker/BuildX
   - Checks disk space

2. **Optional cleanup** (Your choice)
   - Prompt to clean previous builds
   - Recommend: Say "Y" for first build

3. **Build process** (10-15 minutes for first build)
   - Watch for timestamp logging
   - Each step should show `[HH:MM:SS] STEP: ...`
   - Look for validation messages: `✅`

4. **Automated testing** (~2 minutes)
   - Python, UV, PyTorch, JAX, OpenCV, YOLO tests
   - GPU test (if GPU available)

5. **Summary**
   - Build time
   - Image size
   - Next steps

#### Step 5.3: Monitor Build Progress

**Key checkpoints to watch for**:

```
[HH:MM:SS] STEP: Installing system dependencies...
[HH:MM:SS] ✅ System dependencies installed

[HH:MM:SS] STEP: Creating UV virtual environment...
[HH:MM:SS] ✅ Virtual environment created

[HH:MM:SS] STEP: Installing PyTorch with CUDA 12.4...
[HH:MM:SS] ✅ PyTorch version: X.X.X

[HH:MM:SS] STEP: Installing JAX with CUDA 12...
[HH:MM:SS] ✅ JAX version: X.X.X

[HH:MM:SS] STEP: Installing Computer Vision packages...
[HH:MM:SS] ✅ All CV packages installed

BUILD COMPLETE: YYYY-MM-DD HH:MM:SS
```

**If errors occur**: See [Troubleshooting](#troubleshooting) section below

---

### Phase 6: Testing and Validation (15 minutes)

#### Step 6.1: Verify Build Success

After build completes:

```bash
# Check image was created
docker images | grep bball_homography_pipeline_env_datascience

# Should show:
# bball_homography_pipeline_env_datascience  dev     <image-id>  X minutes ago  ~15GB
```

#### Step 6.2: Manual Container Test

```bash
# Start container interactively
docker run --rm -it --gpus all \
  bball_homography_pipeline_env_datascience:dev bash

# Inside container:
python --version              # Should show Python 3.10.x
uv --version                  # Should show uv 0.7.12

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test JAX
python -c "import jax; print(f'JAX: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test YOLO
python -c "from ultralytics import YOLO; print('YOLO: OK')"

# Exit container
exit
```

#### Step 6.3: Test with Docker Compose

```bash
cd ~/projects/bball_homography_pipeline

# Update docker-compose.yml to use new image
# (Should already reference Dockerfile.optimized)

# Start services
docker compose -f .devcontainer/docker-compose.yml up -d

# Check logs
docker compose -f .devcontainer/docker-compose.yml logs -f datascience

# Should see:
# [boot] Starting enhanced container...
# [boot] GPU info: ...
# [boot] Starting Jupyter Lab...
```

**Access Jupyter**:
- Open browser: http://localhost:8895
- Token: `jupyter` (or check .env file)

**Verify in Jupyter**:
1. Open new notebook
2. Run:
   ```python
   import torch, jax, cv2
   from ultralytics import YOLO

   print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
   print(f"JAX: {jax.__version__}, Devices: {jax.devices()}")
   print(f"OpenCV: {cv2.__version__}")
   print("YOLO: Available")
   ```

---

## Verification Checklist

After completing all phases, verify:

### Environment
- [ ] Project located at: `~/projects/bball_homography_pipeline`
- [ ] VS Code shows: `[WSL: Ubuntu]` in title
- [ ] Terminal path starts with `/home/` not `/mnt/`

### Build System
- [ ] BuildKit cache clean (check with `docker buildx du`)
- [ ] .dockerignore present with 150+ lines
- [ ] Dockerfile.optimized exists in .devcontainer/

### Image
- [ ] Image built successfully
- [ ] Build time: 10-15 minutes (first build)
- [ ] Image size: ~15GB
- [ ] All tests passed

### Functionality
- [ ] Container starts without errors
- [ ] PyTorch CUDA available
- [ ] JAX GPU devices detected
- [ ] OpenCV imports successfully
- [ ] YOLO imports successfully
- [ ] Jupyter Lab accessible

---

## Troubleshooting

### Issue: "Still building from /mnt/c"

**Symptoms**:
```
⚠️ WARNING: Building from Windows mount: /mnt/c/...
```

**Solution**:
1. You didn't move to WSL filesystem
2. Re-run Phase 2 Step 2.1
3. Verify with `pwd` - should show `/home/...`

---

### Issue: "tar: invalid tar header" still occurs

**Symptoms**:
```
failed to solve: archive/tar: invalid tar header
```

**Solutions**:

1. **Verify location**:
   ```bash
   pwd  # Must be /home/..., NOT /mnt/c/...
   ```

2. **Clean everything and retry**:
   ```bash
   # Stop all Docker
   docker stop $(docker ps -aq)
   docker system prune -af --volumes
   docker buildx prune -af

   # In PowerShell (Admin):
   wsl --shutdown
   # Restart Docker Desktop

   # Try build again
   ./build_optimized.sh
   ```

3. **Check disk space**:
   ```bash
   df -h ~  # Should have >20GB free
   ```

4. **Try without cache**:
   ```bash
   docker buildx build \
     --progress=plain \
     --no-cache \
     --output=type=docker \
     --sbom=false \
     --provenance=false \
     -t bball_homography_pipeline_env_datascience:dev \
     -f .devcontainer/Dockerfile.optimized .
   ```

---

### Issue: "Context transfer taking forever"

**Symptoms**:
```
[internal] load build context    450.0s
```

**Solution**:

1. **Check .dockerignore**:
   ```bash
   cat .dockerignore | grep -E "(data|videos|models|weights)"
   # Should show all these directories
   ```

2. **Check context size**:
   ```bash
   du -sh .
   # Should be <100MB
   ```

3. **If context is large**, find culprits:
   ```bash
   du -sh * | sort -hr | head -20
   # Add large directories to .dockerignore
   ```

---

### Issue: "Package installation fails"

**Symptoms**:
```
ERROR: Could not install packages: torch, torchvision
```

**Solutions**:

1. **Network timeout** - increase timeout:
   ```dockerfile
   # In Dockerfile, add:
   ENV UV_HTTP_TIMEOUT=600
   ```

2. **PyTorch index issue**:
   ```bash
   # Verify index is accessible
   curl -I https://download.pytorch.org/whl/cu124
   ```

3. **JAX CUDA mismatch**:
   - Check Dockerfile uses correct JAX index
   - Verify CuDNN version: 9.8.0.69

---

### Issue: "GPU not detected in container"

**Symptoms**:
```python
torch.cuda.is_available()  # Returns False
```

**Solutions**:

1. **Check GPU on host**:
   ```bash
   nvidia-smi  # Should show GPU
   ```

2. **Docker GPU test**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Check docker-compose.yml**:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

4. **Restart with GPU**:
   ```bash
   docker compose down
   docker compose up -d
   ```

---

## Rollback Plan

If anything goes wrong, you can rollback:

### Option 1: Use Original Dockerfile

```bash
# Edit docker-compose.yml
# Change:
dockerfile: .devcontainer/Dockerfile.optimized
# To:
dockerfile: .devcontainer/Dockerfile

# Rebuild
docker compose -f .devcontainer/docker-compose.yml build
```

### Option 2: Restore from Git

```bash
cd ~/projects/bball_homography_pipeline

# Check what changed
git status

# Restore specific files
git restore .dockerignore
git restore .devcontainer/Dockerfile

# Or restore everything
git restore .
```

### Option 3: Return to Windows Mount (NOT RECOMMENDED)

```bash
# Only if absolutely necessary
cd /mnt/c/Users/ghadf/vscode_projects/docker_projects/bball_homography_pipeline

# Original files should still be here
code .
```

---

## Performance Benchmarks

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First build | ~31 min | 10-15 min | 2-3x faster |
| Rebuild (cached) | ~15 min | 2-5 min | 3-7x faster |
| Context transfer | 470s (1MB) | <5s | 94x faster |
| Layer export | 325s | <60s | 5x faster |
| Tar corruption | Frequent | None | 100% fix |

### Build Time Breakdown (Expected)

| Phase | Time | Notes |
|-------|------|-------|
| Context load | <5s | Was 470s from /mnt/c |
| Base image | ~30s | First time only |
| System deps | ~60s | Cached after first |
| Python setup | ~30s | Cached |
| PyTorch | ~5min | Large download |
| JAX | ~3min | CUDA dependencies |
| CV packages | ~4min | Ultralytics, OpenCV |
| Final validation | ~30s | Tests |
| **Total** | **10-15min** | First build |

---

## Next Steps After Successful Build

1. **Update CHANGELOG.md**:
   ```bash
   # Add completion entry
   git add CHANGELOG.md
   git commit -m "docs: completed Docker build optimization"
   ```

2. **Test your workflows**:
   - Open existing notebooks
   - Run YOLO detection
   - Test video processing
   - Verify all models load

3. **Document any custom changes**:
   - If you modified Dockerfile for specific needs
   - Add to CHANGELOG.md under "Custom Modifications"

4. **Set up CI/CD** (optional):
   - Use `build_optimized.sh` in GitHub Actions
   - Cache BuildKit layers for faster CI builds

---

## Support and Resources

### Documentation
- Docker BuildKit: https://docs.docker.com/build/buildkit/
- UV Package Manager: https://github.com/astral-sh/uv
- YOLO v8: https://docs.ultralytics.com/

### Files Created
- [CHANGELOG.md](CHANGELOG.md) - Project change history
- [fix_docker_build.py](fix_docker_build.py) - Diagnostic tool
- [.devcontainer/Dockerfile.optimized](.devcontainer/Dockerfile.optimized) - Enhanced Dockerfile
- [.dockerignore](.dockerignore) - Build context optimizer
- [build_optimized.sh](build_optimized.sh) - Automated build script
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - This file

### Getting Help

If you encounter issues not covered in this guide:

1. **Check logs**:
   ```bash
   docker compose logs -f datascience
   ```

2. **Run diagnostics**:
   ```bash
   python3 fix_docker_build.py
   ```

3. **Check CHANGELOG.md** for known issues

---

**Last Updated**: 2025-10-08
**Version**: 1.0
**Status**: Ready for implementation
