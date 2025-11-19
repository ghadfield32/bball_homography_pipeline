#!/bin/bash
# Validation script for Podman VS Code integration

echo "🔍 Validating Podman VS Code Integration"
echo "========================================"

# Function to print section headers
print_section() {
    echo ""
    echo "=== $1 ==="
}

# Function to check if running in WSL
check_wsl() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "✅ Detected WSL environment"
        export WSL_ENVIRONMENT=true
        return 0
    else
        echo "ℹ️ Not running in WSL"
        export WSL_ENVIRONMENT=false
        return 1
    fi
}

# Check Podman installation
print_section "CHECKING PODMAN INSTALLATION"
if command -v podman >/dev/null 2>&1; then
    echo "✅ Podman installed: $(podman --version)"
    
    # Check Podman info
    if podman info >/dev/null 2>&1; then
        echo "✅ Podman is functional"
    else
        echo "❌ Podman is not functional"
        exit 1
    fi
else
    echo "❌ Podman not found"
    echo "💡 Install Podman: sudo apt install podman"
    exit 1
fi

# Check Docker compatibility
print_section "CHECKING DOCKER COMPATIBILITY"
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker compatibility available: $(docker --version)"
    
    # Test Docker info through Podman
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker API working through Podman"
    else
        echo "⚠️ Docker API not working (may be normal if no containers running)"
    fi
else
    echo "❌ Docker compatibility not available"
    echo "💡 Install podman-docker: sudo apt install podman-docker"
    exit 1
fi

# Check Podman socket
print_section "CHECKING PODMAN SOCKET SERVICE"
if systemctl --user is-active podman.socket >/dev/null 2>&1; then
    echo "✅ Podman socket service active"
    
    # Check socket file
    local socket_path="/run/user/$(id -u)/podman/podman.sock"
    if [ -S "$socket_path" ]; then
        echo "✅ Podman socket found at: $socket_path"
    else
        echo "⚠️ Podman socket not found at expected location"
    fi
else
    echo "⚠️ Podman socket service not active"
    echo "💡 Enable with: systemctl --user enable --now podman.socket"
fi

# Check VS Code settings
print_section "CHECKING VS CODE CONFIGURATION"
if [ -f ".vscode/settings.json" ]; then
    echo "✅ VS Code settings file found"
    
    if grep -q "dev.containers.dockerPath.*podman" ".vscode/settings.json"; then
        echo "✅ VS Code configured to use Podman"
    else
        echo "❌ VS Code not configured to use Podman"
        echo "💡 Add 'dev.containers.dockerPath': 'podman' to .vscode/settings.json"
    fi
    
    if grep -q "docker.host.*podman" ".vscode/settings.json"; then
        echo "✅ VS Code configured with Podman socket"
    else
        echo "ℹ️ VS Code not configured with Podman socket (optional)"
    fi
else
    echo "❌ VS Code settings file not found"
    echo "💡 Create .vscode/settings.json with Podman configuration"
fi

# Check devcontainer configuration
print_section "CHECKING DEVCONTAINER CONFIGURATION"
if [ -f ".devcontainer/devcontainer.json" ]; then
    echo "✅ devcontainer.json found"
    
    if grep -q "dockerComposeFile" ".devcontainer/devcontainer.json"; then
        echo "✅ Using docker-compose configuration"
    else
        echo "ℹ️ Not using docker-compose configuration"
    fi
    
    if grep -q "CONTAINER_RUNTIME.*podman" ".devcontainer/devcontainer.json"; then
        echo "✅ DevContainer configured for Podman"
    else
        echo "ℹ️ DevContainer not explicitly configured for Podman"
    fi
else
    echo "❌ devcontainer.json not found"
    exit 1
fi

# Check docker-compose file
print_section "CHECKING DOCKER-COMPOSE CONFIGURATION"
if [ -f ".devcontainer/docker-compose.yml" ]; then
    echo "✅ docker-compose.yml found"
    
    if grep -q "devices.*nvidia.com/gpu" ".devcontainer/docker-compose.yml"; then
        echo "✅ GPU access configured for Podman"
    else
        echo "⚠️ GPU access not configured for Podman"
    fi
    
    if grep -q "security_opt.*label:disable" ".devcontainer/docker-compose.yml"; then
        echo "✅ Security options configured for Podman"
    else
        echo "ℹ️ Security options not configured for Podman"
    fi
else
    echo "❌ docker-compose.yml not found"
fi

# Check cache directories
print_section "CHECKING CACHE DIRECTORIES"
for cache_dir in "cache/uv" "cache/ultralytics" "cache/roboflow"; do
    if [ -d "$cache_dir" ]; then
        echo "✅ Cache directory found: $cache_dir"
    else
        echo "⚠️ Cache directory not found: $cache_dir"
        echo "💡 Create with: mkdir -p $cache_dir"
    fi
done

# Check NVIDIA Container Toolkit
print_section "CHECKING NVIDIA CONTAINER TOOLKIT"
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "✅ NVIDIA Container Toolkit installed"
    
    # Check CDI configuration
    if [ -f "/etc/cdi/nvidia.yaml" ]; then
        echo "✅ NVIDIA CDI configuration found"
        
        # Test CDI device access
        if nvidia-ctk cdi list >/dev/null 2>&1; then
            echo "✅ NVIDIA CDI devices accessible"
        else
            echo "⚠️ NVIDIA CDI devices not accessible"
        fi
    else
        echo "⚠️ NVIDIA CDI configuration not found"
        echo "💡 Generate with: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
    fi
else
    echo "⚠️ NVIDIA Container Toolkit not found"
    echo "💡 Install for GPU support in containers"
fi

# Test basic container operations
print_section "TESTING BASIC CONTAINER OPERATIONS"
echo "Testing container run capability..."

# Test with a simple container
if docker run --rm hello-world >/dev/null 2>&1; then
    echo "✅ Container run test successful"
else
    echo "⚠️ Container run test failed (may be normal if hello-world image not available)"
fi

# Test GPU access if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Testing GPU access in container..."
    if docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU access test successful"
    else
        echo "⚠️ GPU access test failed"
    fi
else
    echo "ℹ️ nvidia-smi not available, skipping GPU test"
fi

# Check environment
print_section "ENVIRONMENT INFORMATION"
check_wsl
echo "User: $(whoami)"
echo "UID: $(id -u)"
echo "GID: $(id -g)"
echo "Working Directory: $(pwd)"

# Final validation summary
print_section "VALIDATION SUMMARY"
echo "🎉 Podman VS Code integration validation complete!"
echo ""
echo "Next steps:"
echo "1. Open this workspace in VS Code"
echo "2. Command Palette: 'Dev Containers: Reopen in Container'"
echo "3. VS Code will use Podman to build and run the container"
echo ""
echo "Troubleshooting:"
echo "- If container fails to start, check Podman logs"
echo "- Ensure NVIDIA Container Toolkit is installed for GPU support"
echo "- Verify Podman socket service is running"
echo "- Check VS Code settings for Podman configuration"
echo ""
echo "PowerShell helper:"
echo "  .\scripts\vscode-podman-helper.ps1 status"
echo ""
echo "WSL helper:"
echo "  bash scripts/setup-podman-docker-shim.sh"
