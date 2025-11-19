#!/bin/bash
# Optimized Docker build script for Basketball Homography Pipeline
# Generated: 2025-10-08
#
# This script implements all optimizations to fix tar corruption and reduce build time
#
# Usage:
#   chmod +x build_optimized.sh
#   ./build_optimized.sh
#
# Prerequisites:
#   - Run from WSL filesystem (NOT /mnt/c)
#   - BuildKit cache cleaned (run fix_docker_build.py --clean-cache)
#   - Enhanced .dockerignore in place

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="bball_homography_pipeline_env_datascience"
IMAGE_TAG="dev"
DOCKERFILE_PATH=".devcontainer/Dockerfile.optimized"
BUILD_CONTEXT="."

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if running from Windows mount
    if [[ "$(pwd)" == /mnt/* ]]; then
        print_error "CRITICAL: Running from Windows mount (/mnt/c)"
        print_error "This will cause slow builds and tar corruption!"
        echo ""
        echo "Please move project to WSL filesystem:"
        echo "  mkdir -p ~/projects"
        echo "  cp -r $(pwd) ~/projects/"
        echo "  cd ~/projects/bball_homography_pipeline"
        echo ""
        exit 1
    fi
    print_success "Running from WSL filesystem: $(pwd)"

    # Check Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found in PATH"
        exit 1
    fi
    print_success "Docker available: $(docker --version)"

    # Check BuildX is available
    if ! docker buildx version &> /dev/null; then
        print_error "Docker BuildX not available"
        exit 1
    fi
    print_success "BuildX available: $(docker buildx version | head -n1)"

    # Check Dockerfile exists
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        print_warning "Optimized Dockerfile not found at: $DOCKERFILE_PATH"
        echo "Falling back to standard Dockerfile..."
        DOCKERFILE_PATH=".devcontainer/Dockerfile"
        if [ ! -f "$DOCKERFILE_PATH" ]; then
            print_error "No Dockerfile found!"
            exit 1
        fi
    fi
    print_success "Dockerfile found: $DOCKERFILE_PATH"

    # Check .dockerignore exists
    if [ ! -f ".dockerignore" ]; then
        print_warning ".dockerignore not found - build context may be large"
    else
        print_success ".dockerignore found"
    fi

    # Check disk space
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 20 ]; then
        print_error "Low disk space: ${available_space}GB available"
        print_error "Recommended: At least 20GB free"
        exit 1
    fi
    print_success "Disk space available: ${available_space}GB"

    echo ""
}

clean_previous_builds() {
    print_header "Cleaning Previous Builds (Optional)"

    echo "This will remove existing images and cache."
    read -p "Clean previous builds? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Cleaning previous builds..."

        # Stop and remove existing containers
        docker compose -f .devcontainer/docker-compose.yml down --volumes 2>/dev/null || true

        # Remove existing images
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG} 2>/dev/null || true
        docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true

        # Clean BuildKit cache
        docker buildx prune -af

        print_success "Previous builds cleaned"
    else
        print_warning "Skipping cleanup - using existing cache"
    fi

    echo ""
}

start_build() {
    print_header "Starting Optimized Build"

    # Set BuildKit environment variables
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1

    echo "Build configuration:"
    echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  Dockerfile: ${DOCKERFILE_PATH}"
    echo "  Context: ${BUILD_CONTEXT}"
    echo "  BuildKit: Enabled"
    echo ""

    # Record start time
    START_TIME=$(date +%s)

    # Build with optimized settings
    print_warning "Building image (this may take 10-15 minutes)..."
    echo ""

    docker buildx build \
        --progress=plain \
        --no-cache \
        --output=type=docker \
        --sbom=false \
        --provenance=false \
        --attest=type=none \
        --tag ${IMAGE_NAME}:${IMAGE_TAG} \
        --tag ${IMAGE_NAME}:latest \
        --file ${DOCKERFILE_PATH} \
        ${BUILD_CONTEXT}

    # Calculate build time
    END_TIME=$(date +%s)
    BUILD_TIME=$((END_TIME - START_TIME))
    BUILD_MINUTES=$((BUILD_TIME / 60))
    BUILD_SECONDS=$((BUILD_TIME % 60))

    echo ""
    print_success "Build completed in ${BUILD_MINUTES}m ${BUILD_SECONDS}s"
    echo ""
}

test_image() {
    print_header "Testing Built Image"

    print_warning "Running basic tests..."

    # Test Python is available
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python --version; then
        print_success "Python test passed"
    else
        print_error "Python test failed"
        return 1
    fi

    # Test UV is available
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} uv --version; then
        print_success "UV test passed"
    else
        print_error "UV test failed"
        return 1
    fi

    # Test PyTorch (CPU mode, no GPU needed)
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "import torch; print('PyTorch:', torch.__version__)"; then
        print_success "PyTorch test passed"
    else
        print_error "PyTorch test failed"
        return 1
    fi

    # Test JAX
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "import jax; print('JAX:', jax.__version__)"; then
        print_success "JAX test passed"
    else
        print_error "JAX test failed"
        return 1
    fi

    # Test OpenCV
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "import cv2; print('OpenCV:', cv2.__version__)"; then
        print_success "OpenCV test passed"
    else
        print_error "OpenCV test failed"
        return 1
    fi

    # Test YOLO
    if docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "from ultralytics import YOLO; print('YOLO: OK')"; then
        print_success "YOLO test passed"
    else
        print_error "YOLO test failed"
        return 1
    fi

    echo ""
    print_success "All basic tests passed!"

    # GPU test (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_warning "Testing GPU access (requires --gpus flag)..."
        if docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} \
            python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU test: OK')"; then
            print_success "GPU test passed"
        else
            print_warning "GPU test failed (this is OK if no GPU is available)"
        fi
    fi

    echo ""
}

show_summary() {
    print_header "Build Summary"

    # Get image size
    IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "{{.Size}}")

    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Size: ${IMAGE_SIZE}"
    echo "Dockerfile: ${DOCKERFILE_PATH}"
    echo ""

    print_success "Build complete and tested!"
    echo ""
    echo "Next steps:"
    echo "  1. Test manually: docker run --rm -it ${IMAGE_NAME}:${IMAGE_TAG} bash"
    echo "  2. Start with compose: docker compose -f .devcontainer/docker-compose.yml up"
    echo "  3. Or open in VSCode: Remote-Containers: Reopen in Container"
    echo ""
}

# ============================================================================
# Main execution
# ============================================================================

main() {
    print_header "Basketball Homography Pipeline - Optimized Build"

    check_prerequisites
    clean_previous_builds
    start_build
    test_image
    show_summary
}

# Run main function
main "$@"
