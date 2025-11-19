#!/bin/bash
# Diagnostic script to analyze Docker build context and identify issues
# Usage: bash diagnose_build_context.sh

set -euo pipefail

echo "================================================================="
echo "Docker Build Context Diagnostic Tool"
echo "================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check current filesystem
echo "1. FILESYSTEM CHECK"
echo "-----------------"
CURRENT_DIR=$(pwd)
FS_TYPE=$(df -T . | tail -1 | awk '{print $2}')
MOUNT_POINT=$(df -h . | tail -1 | awk '{print $6}')

echo "Current directory: $CURRENT_DIR"
echo "Filesystem type: $FS_TYPE"
echo "Mount point: $MOUNT_POINT"

if [[ "$MOUNT_POINT" == /mnt/* ]] || [[ "$MOUNT_POINT" == /c/* ]] || [[ "$MOUNT_POINT" == /d/* ]]; then
    echo -e "${RED}⚠️  WARNING: Building from Windows mount point!${NC}"
    echo -e "${RED}   This causes 10-50x slower I/O and build timeouts${NC}"
    echo -e "${YELLOW}   SOLUTION: Move project to WSL filesystem (~/projects/)${NC}"
else
    echo -e "${GREEN}✓ Building from native WSL/Linux filesystem${NC}"
fi
echo ""

# 2. Check Docker Build Context Size
echo "2. BUILD CONTEXT SIZE"
echo "---------------------"
echo "Analyzing files that will be sent to Docker..."

# Create temporary context tar to see what Docker would send
TEMP_TAR=$(mktemp)
trap "rm -f $TEMP_TAR" EXIT

# Simulate Docker context build (respecting .dockerignore)
tar --exclude-from=<(cat .dockerignore 2>/dev/null || echo "") \
    --exclude='.git' \
    -czf "$TEMP_TAR" . 2>/dev/null || true

CONTEXT_SIZE=$(du -h "$TEMP_TAR" | awk '{print $1}')
CONTEXT_SIZE_BYTES=$(stat -f%z "$TEMP_TAR" 2>/dev/null || stat -c%s "$TEMP_TAR")

echo "Build context size: $CONTEXT_SIZE ($CONTEXT_SIZE_BYTES bytes)"

if (( CONTEXT_SIZE_BYTES > 5000000 )); then
    echo -e "${RED}⚠️  WARNING: Build context is larger than 5MB${NC}"
    echo -e "${YELLOW}   Large contexts cause slow transfers, especially on Windows mounts${NC}"
else
    echo -e "${GREEN}✓ Build context size is acceptable${NC}"
fi
echo ""

# 3. Find largest files being included
echo "3. LARGEST FILES IN BUILD CONTEXT"
echo "----------------------------------"
echo "Top 20 largest files that will be sent to Docker:"

find . -type f \
    ! -path './.git/*' \
    ! -path './node_modules/*' \
    ! -path './data/*' \
    ! -path './models/*' \
    ! -path './weights/*' \
    ! -path './videos/*' \
    -exec du -h {} \; 2>/dev/null | \
    sort -rh | \
    head -20 | \
    awk '{printf "  %8s  %s\n", $1, $2}'

echo ""

# 4. Check .dockerignore effectiveness
echo "4. DOCKERIGNORE ANALYSIS"
echo "------------------------"
if [ -f .dockerignore ]; then
    echo "✓ .dockerignore exists"
    echo "Lines in .dockerignore: $(wc -l < .dockerignore)"
    echo ""
    echo "Current patterns:"
    cat .dockerignore | grep -v '^#' | grep -v '^$' | sed 's/^/  /'
else
    echo -e "${RED}⚠️  WARNING: No .dockerignore file found!${NC}"
fi
echo ""

# 5. Check Docker BuildKit cache
echo "5. DOCKER BUILDKIT CACHE"
echo "------------------------"
docker system df 2>/dev/null || echo "Docker command not available"
echo ""

BUILD_CACHE_SIZE=$(docker system df 2>/dev/null | grep "Build Cache" | awk '{print $4}' || echo "Unknown")
echo "Build cache size: $BUILD_CACHE_SIZE"

if [[ "$BUILD_CACHE_SIZE" =~ "GB" ]]; then
    CACHE_GB=$(echo "$BUILD_CACHE_SIZE" | sed 's/GB.*//')
    if (( $(echo "$CACHE_GB > 10" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${YELLOW}⚠️  Large build cache detected (>10GB)${NC}"
        echo -e "${YELLOW}   Consider pruning: docker builder prune -af${NC}"
    fi
fi
echo ""

# 6. Check for Claude Code notebook that might be overwriting devcontainer.json
echo "6. DEVCONTAINER.JSON CONFLICTS"
echo "-------------------------------"
NOTEBOOK_DEVCONTAINER_CELLS=$(find notebooks -name "*.ipynb" -exec grep -l "%%writefile.*devcontainer.json" {} \; 2>/dev/null || true)

if [ -n "$NOTEBOOK_DEVCONTAINER_CELLS" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Found notebooks with devcontainer.json writefile cells:${NC}"
    echo "$NOTEBOOK_DEVCONTAINER_CELLS" | sed 's/^/  /'
    echo -e "${YELLOW}   These might overwrite your .devcontainer/devcontainer.json${NC}"
else
    echo -e "${GREEN}✓ No conflicting devcontainer.json writefile cells found${NC}"
fi
echo ""

# 7. Recommendations
echo "================================================================="
echo "RECOMMENDATIONS"
echo "================================================================="
echo ""

if [[ "$MOUNT_POINT" == /mnt/* ]] || [[ "$MOUNT_POINT" == /c/* ]]; then
    echo -e "${YELLOW}HIGH PRIORITY: Move project to WSL filesystem${NC}"
    echo "  Current: $CURRENT_DIR"
    echo "  Recommended: ~/projects/bball_homography_pipeline"
    echo "  Command: See move_to_wsl.sh script"
    echo ""
fi

if (( CONTEXT_SIZE_BYTES > 5000000 )); then
    echo -e "${YELLOW}MEDIUM PRIORITY: Reduce build context size${NC}"
    echo "  Add more patterns to .dockerignore"
    echo "  Suggested additions:"
    echo "    notebooks/"
    echo "    *.ipynb"
    echo "    mlruns/"
    echo "    mlflow_db/"
    echo ""
fi

if [[ "$BUILD_CACHE_SIZE" =~ "GB" ]]; then
    echo -e "${YELLOW}MEDIUM PRIORITY: Clean Docker build cache${NC}"
    echo "  Command: docker builder prune -af"
    echo "  Command: docker system prune -a --volumes"
    echo ""
fi

echo "================================================================="
echo "END OF DIAGNOSTIC"
echo "================================================================="
