#!/bin/bash
# Diagnostic script for unpigz/deflate corruption in Docker builds
# Run this from within WSL2

echo "================================================"
echo "Docker unpigz Corruption Diagnostics"
echo "================================================"
echo ""

# 1. Check if running from WSL or Windows
echo "=== 1. Environment Check ==="
echo "Current path: $(pwd)"
if [[ "$(pwd)" == /mnt/* ]]; then
    echo "⚠️  WARNING: Running from Windows mount (slow I/O)"
    echo "   Path uses /mnt/* which goes through 9P protocol"
else
    echo "✅ Running from native WSL filesystem"
fi
echo ""

# 2. Check Docker storage location
echo "=== 2. Docker Storage Location ==="
DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')
echo "Docker Root: $DOCKER_ROOT"
if [[ "$DOCKER_ROOT" == /mnt/* ]]; then
    echo "⚠️  CRITICAL: Docker storage on Windows mount!"
    echo "   This WILL cause corruption with large images"
else
    echo "✅ Docker storage on native filesystem"
fi
echo ""

# 3. Check available disk space
echo "=== 3. Disk Space Check ==="
df -h "$DOCKER_ROOT" 2>/dev/null || df -h /var/lib/docker
echo ""

# 4. Check Docker daemon status
echo "=== 4. Docker Daemon Health ==="
docker info 2>&1 | grep -E "Server Version|Storage Driver|Logging Driver|Cgroup|Kernel|OS"
echo ""

# 5. Check for zombie buildkit processes
echo "=== 5. BuildKit Process Check ==="
ps aux | grep -E "buildkit|unpigz" | grep -v grep || echo "No buildkit/unpigz processes found"
echo ""

# 6. Check overlay filesystem status
echo "=== 6. Overlay Filesystem Status ==="
if mount | grep -q overlay; then
    mount | grep overlay | head -5
    echo "..."
else
    echo "No overlay mounts found"
fi
echo ""

# 7. Check for corrupted images/layers
echo "=== 7. Docker Image Integrity ==="
echo "Images with potential issues:"
docker images --format "{{.Repository}}:{{.Tag}} - {{.Size}} - {{.CreatedAt}}" | head -10
echo ""

# 8. Check BuildKit builder status
echo "=== 8. BuildKit Builders ==="
docker buildx ls 2>/dev/null || echo "BuildKit not available"
echo ""

# 9. Check for recent Docker errors in logs
echo "=== 9. Recent Docker Errors ==="
if [ -f /var/log/docker.log ]; then
    tail -20 /var/log/docker.log | grep -i error || echo "No recent errors"
else
    journalctl -u docker --no-pager -n 20 2>/dev/null | grep -i error || echo "No recent errors in journal"
fi
echo ""

# 10. Memory and swap status
echo "=== 10. Memory Status ==="
free -h
echo ""

# 11. Check WSL2 memory limits
echo "=== 11. WSL2 Configuration ==="
if [ -f ~/.wslconfig ]; then
    echo "~/.wslconfig contents:"
    cat ~/.wslconfig
elif [ -f /mnt/c/Users/$USER/.wslconfig ]; then
    echo "Windows .wslconfig contents:"
    cat /mnt/c/Users/$USER/.wslconfig
else
    echo "No .wslconfig found (using defaults)"
fi
echo ""

echo "================================================"
echo "RECOMMENDATIONS"
echo "================================================"
echo ""

# Summary and recommendations
if [[ "$(pwd)" == /mnt/* ]] || [[ "$DOCKER_ROOT" == /mnt/* ]]; then
    echo "🚨 CRITICAL: Windows filesystem involvement detected"
    echo ""
    echo "The unpigz corruption is caused by:"
    echo "1. 9P protocol data corruption during long I/O operations"
    echo "2. ~278 seconds of sustained writes through Windows mount"
    echo "3. Bit-level corruption in gzip deflate stream"
    echo ""
    echo "SOLUTIONS (in order of preference):"
    echo ""
    echo "Option A: Move project to WSL filesystem (RECOMMENDED)"
    echo "  mkdir -p ~/projects"
    echo "  cp -r . ~/projects/bball_homography_pipeline"
    echo "  cd ~/projects/bball_homography_pipeline"
    echo ""
    echo "Option B: Nuclear Docker reset (may help temporarily)"
    echo "  docker system prune -a --volumes -f"
    echo "  docker buildx prune -a -f"
    echo "  docker buildx rm --all-inactive"
    echo "  wsl --shutdown  # from PowerShell"
    echo "  # Then restart Docker Desktop"
    echo ""
    echo "Option C: Reduce image size (partial mitigation)"
    echo "  - Split into multi-stage build"
    echo "  - Use smaller base image"
    echo "  - But this only reduces risk, doesn't eliminate it"
else
    echo "✅ Filesystem configuration looks good"
    echo ""
    echo "Try these steps:"
    echo "1. docker system prune -a --volumes -f"
    echo "2. docker buildx prune -a -f"
    echo "3. Restart Docker Desktop"
    echo "4. Rebuild"
fi
echo ""
echo "================================================"
