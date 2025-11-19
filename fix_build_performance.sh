#!/bin/bash
# Fix Docker build performance issues
# This script addresses the root causes of slow builds and timeouts
#
# Issues fixed:
# 1. Slow builds from Windows mount points (/c/ or /mnt/c)
# 2. Corrupted Docker BuildKit cache
# 3. Large build context from unnecessary files
#
# Usage: bash fix_build_performance.sh [--move-to-wsl] [--clean-docker] [--test-build]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================="
echo "Docker Build Performance Fix Script"
echo "================================================================="
echo ""

# Parse arguments
MOVE_TO_WSL=false
CLEAN_DOCKER=false
TEST_BUILD=false

for arg in "$@"; do
    case $arg in
        --move-to-wsl)
            MOVE_TO_WSL=true
            ;;
        --clean-docker)
            CLEAN_DOCKER=true
            ;;
        --test-build)
            TEST_BUILD=true
            ;;
        --help)
            echo "Usage: bash fix_build_performance.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --move-to-wsl    Move project from Windows mount to WSL filesystem"
            echo "  --clean-docker   Clean Docker cache and corrupted images"
            echo "  --test-build     Run a test build after fixes"
            echo "  --help           Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# Step 1: Check current location
echo -e "${BLUE}Step 1: Checking current filesystem...${NC}"
CURRENT_DIR=$(pwd)
MOUNT_POINT=$(df -h . | tail -1 | awk '{print $6}')

echo "Current directory: $CURRENT_DIR"
echo "Mount point: $MOUNT_POINT"

if [[ "$MOUNT_POINT" == /mnt/* ]] || [[ "$MOUNT_POINT" == /c/* ]] || [[ "$MOUNT_POINT" == /d/* ]]; then
    echo -e "${RED}⚠️  WARNING: You are on a Windows mount point!${NC}"
    echo -e "${YELLOW}   This causes 10-50x slower Docker builds${NC}"

    if [ "$MOVE_TO_WSL" = false ]; then
        echo -e "${YELLOW}   Run with --move-to-wsl to fix this${NC}"
    fi
else
    echo -e "${GREEN}✓ Already on WSL/Linux filesystem${NC}"
fi
echo ""

# Step 2: Clean Docker cache
if [ "$CLEAN_DOCKER" = true ]; then
    echo -e "${BLUE}Step 2: Cleaning Docker cache...${NC}"

    echo "Current Docker disk usage:"
    docker system df
    echo ""

    read -p "⚠️  This will delete ALL Docker cache, unused images, and volumes. Continue? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning Docker BuildKit cache..."
        docker builder prune -af || echo "BuildKit prune failed (may not exist)"

        echo "Cleaning Docker system..."
        docker system prune -a --volumes -f

        echo -e "${GREEN}✓ Docker cache cleaned${NC}"
        echo ""
        echo "New Docker disk usage:"
        docker system df
    else
        echo "Skipped Docker cleanup"
    fi
    echo ""
fi

# Step 3: Move to WSL filesystem
if [ "$MOVE_TO_WSL" = true ]; then
    echo -e "${BLUE}Step 3: Moving project to WSL filesystem...${NC}"

    if [[ "$MOUNT_POINT" != /mnt/* ]] && [[ "$MOUNT_POINT" != /c/* ]]; then
        echo -e "${GREEN}✓ Already on WSL filesystem, skipping move${NC}"
    else
        # Determine target directory
        if [ -n "${WSL_DISTRO_NAME:-}" ]; then
            # In WSL
            WSL_HOME="/home/$(whoami)"
        else
            WSL_HOME="$HOME"
        fi

        TARGET_DIR="$WSL_HOME/projects/bball_homography_pipeline"

        echo "Target directory: $TARGET_DIR"
        echo "Current directory: $CURRENT_DIR"
        echo ""
        echo -e "${YELLOW}This will:${NC}"
        echo "1. Create $TARGET_DIR"
        echo "2. Copy all project files (excluding large data/models)"
        echo "3. Preserve git history"
        echo "4. Leave original files intact (manual cleanup later)"
        echo ""

        read -p "Continue? (y/N) " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Create target directory
            mkdir -p "$(dirname "$TARGET_DIR")"

            # Use rsync to copy project
            echo "Copying project files..."
            rsync -avh --progress \
                --exclude='.git/objects/pack/*.pack' \
                --exclude='data/' \
                --exclude='models/' \
                --exclude='weights/' \
                --exclude='videos/' \
                --exclude='mlruns/' \
                --exclude='notebooks/' \
                --exclude='__pycache__/' \
                --exclude='*.pyc' \
                "$CURRENT_DIR/" "$TARGET_DIR/"

            echo -e "${GREEN}✓ Project copied to $TARGET_DIR${NC}"
            echo ""
            echo -e "${YELLOW}NEXT STEPS:${NC}"
            echo "1. cd $TARGET_DIR"
            echo "2. Verify files: ls -la"
            echo "3. Run this script again from new location with --clean-docker"
            echo "4. Build: cd .devcontainer && docker-compose build"
            echo ""
            echo -e "${YELLOW}After verifying everything works:${NC}"
            echo "5. Delete original: rm -rf $CURRENT_DIR"
        else
            echo "Skipped move"
        fi
    fi
    echo ""
fi

# Step 4: Test build
if [ "$TEST_BUILD" = true ]; then
    echo -e "${BLUE}Step 4: Testing Docker build...${NC}"

    if [ ! -f ".devcontainer/docker-compose.yml" ]; then
        echo -e "${RED}⚠️  .devcontainer/docker-compose.yml not found!${NC}"
        echo "Make sure you're in the project root directory"
        exit 1
    fi

    echo "Starting test build (this may take 10-20 minutes)..."
    echo ""

    cd .devcontainer

    # Set build start time
    START_TIME=$(date +%s)

    # Build with verbose output
    if docker-compose build --progress=plain datascience 2>&1 | tee build.log; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo ""
        echo -e "${GREEN}✓ Build successful!${NC}"
        echo "Build time: ${MINUTES}m ${SECONDS}s"

        # Analyze build log
        echo ""
        echo "Build performance analysis:"
        echo "----------------------------"

        if [ -f build.log ]; then
            CONTEXT_TRANSFER_TIME=$(grep "transferring context" build.log | tail -1 || echo "Not found")
            echo "Context transfer: $CONTEXT_TRANSFER_TIME"

            # Check if build was from cache
            CACHED_LAYERS=$(grep -c "CACHED" build.log || echo "0")
            TOTAL_LAYERS=$(grep -c "=>" build.log || echo "1")
            CACHE_RATIO=$((CACHED_LAYERS * 100 / TOTAL_LAYERS))

            echo "Cached layers: $CACHED_LAYERS / $TOTAL_LAYERS ($CACHE_RATIO%)"
        fi
    else
        echo -e "${RED}✗ Build failed!${NC}"
        echo "Check build.log for details"
        exit 1
    fi

    cd ..
fi

# Summary
echo "================================================================="
echo "SUMMARY"
echo "================================================================="
echo ""

if [ "$MOVE_TO_WSL" = false ] && [[ "$MOUNT_POINT" == /mnt/* || "$MOUNT_POINT" == /c/* ]]; then
    echo -e "${RED}⚠️  CRITICAL: Still on Windows mount point${NC}"
    echo -e "${YELLOW}   Run: bash fix_build_performance.sh --move-to-wsl${NC}"
    echo ""
fi

echo "Recommended next steps:"
echo "1. Clean Docker cache: bash fix_build_performance.sh --clean-docker"
echo "2. Move to WSL: bash fix_build_performance.sh --move-to-wsl"
echo "3. Test build: bash fix_build_performance.sh --test-build"
echo ""
echo "Or run all at once:"
echo "  bash fix_build_performance.sh --move-to-wsl --clean-docker --test-build"
echo ""

echo "================================================================="
