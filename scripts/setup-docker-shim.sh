#!/bin/bash
# Enhanced Docker Shim Setup for Podman
# Ensures Docker commands work through Podman for VS Code Dev Containers

set -e

echo "🐳 Setting up Docker Shim for Podman"
echo "===================================="

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

# Install podman-docker package
install_podman_docker() {
    print_section "INSTALLING PODMAN-DOCKER PACKAGE"
    
    echo "Updating package list..."
    sudo apt-get update
    
    echo "Installing podman-docker package..."
    if sudo apt-get install -y podman-docker; then
        echo "✅ podman-docker package installed successfully"
    else
        echo "❌ Failed to install podman-docker package"
        echo "💡 Trying alternative installation method..."
        
        # Alternative: Create symlink manually
        if command -v podman >/dev/null 2>&1; then
            echo "Creating Docker symlink to Podman..."
            sudo ln -sf $(which podman) /usr/bin/docker
            echo "✅ Docker symlink created"
        else
            echo "❌ Podman not found. Please install Podman first."
            return 1
        fi
    fi
}

# Verify Docker shim functionality
verify_docker_shim() {
    print_section "VERIFYING DOCKER SHIM FUNCTIONALITY"
    
    echo "Testing Docker command..."
    if docker --version >/dev/null 2>&1; then
        echo "✅ Docker command working: $(docker --version)"
    else
        echo "❌ Docker command failed"
        return 1
    fi
    
    echo "Testing Docker info command..."
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker info command working"
        echo "Container runtime: $(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo 'Unknown')"
    else
        echo "⚠️ Docker info command failed (may be normal if no containers running)"
    fi
    
    echo "Testing Docker version command..."
    if docker version >/dev/null 2>&1; then
        echo "✅ Docker version command working"
    else
        echo "❌ Docker version command failed"
    fi
}

# Setup Podman socket service
setup_podman_socket() {
    print_section "SETTING UP PODMAN SOCKET SERVICE"
    
    echo "Enabling Podman socket service..."
    if systemctl --user enable --now podman.socket; then
        echo "✅ Podman socket service enabled and started"
    else
        echo "⚠️ Could not enable Podman socket service"
        echo "💡 This may be normal in some environments"
    fi
    
    # Check if socket exists
    local socket_path="/run/user/$(id -u)/podman/podman.sock"
    if [ -S "$socket_path" ]; then
        echo "✅ Podman socket found at: $socket_path"
    else
        echo "⚠️ Podman socket not found at expected location"
        echo "💡 Socket may be created when first container is started"
    fi
}

# Create environment configuration
create_environment_config() {
    print_section "CREATING ENVIRONMENT CONFIGURATION"
    
    local env_file="$HOME/.docker_podman_env"
    
    echo "Creating Docker-Podman environment file..."
    cat > "$env_file" << 'EOF'
# Docker-Podman Environment Configuration
# This file configures environment variables for Docker compatibility

# Podman socket configuration
export PODMAN_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"

# Dev Containers specific configuration
export DEVCONTAINERS_DOCKER_PATH="podman"

# Alternative: Use TCP socket if unix socket doesn't work
# export PODMAN_HOST="tcp://localhost:8080"
# export DOCKER_HOST="tcp://localhost:8080"
EOF
    
    echo "✅ Environment file created"
    echo "📁 Location: $env_file"
    
    # Add to bashrc if not already present
    if ! grep -q "docker_podman_env" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Docker-Podman Environment" >> "$HOME/.bashrc"
        echo "if [ -f \"$env_file\" ]; then" >> "$HOME/.bashrc"
        echo "  source \"$env_file\"" >> "$HOME/.bashrc"
        echo "fi" >> "$HOME/.bashrc"
        echo "✅ Added environment setup to .bashrc"
    else
        echo "✅ Environment setup already in .bashrc"
    fi
}

# Test basic container operations
test_container_operations() {
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
}

# Create validation script
create_validation_script() {
    print_section "CREATING VALIDATION SCRIPT"
    
    local validation_script="../scripts/validate-docker-shim.sh"
    
    cat > "$validation_script" << 'EOF'
#!/bin/bash
# Validation script for Docker shim functionality

echo "🔍 Validating Docker Shim Functionality"
echo "======================================="

# Check Podman installation
echo "Checking Podman installation..."
if command -v podman >/dev/null 2>&1; then
    echo "✅ Podman installed: $(podman --version)"
else
    echo "❌ Podman not found"
    exit 1
fi

# Check Docker shim
echo "Checking Docker shim..."
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker shim available: $(docker --version)"
else
    echo "❌ Docker shim not available"
    exit 1
fi

# Test Docker functionality
echo "Testing Docker functionality..."
if docker info >/dev/null 2>&1; then
    echo "✅ Docker info command working"
else
    echo "⚠️ Docker info command failed"
fi

# Test Podman socket
echo "Checking Podman socket..."
if systemctl --user is-active podman.socket >/dev/null 2>&1; then
    echo "✅ Podman socket service active"
else
    echo "⚠️ Podman socket service not active"
fi

echo ""
echo "🎉 Docker shim validation complete!"
EOF
    
    chmod +x "$validation_script"
    echo "✅ Validation script created"
    echo "📁 Location: $validation_script"
}

# Main execution
main() {
    echo "Starting Docker shim setup for Podman..."
    
    # Check environment
    check_wsl
    
    # Install podman-docker package
    install_podman_docker
    
    # Verify Docker shim functionality
    verify_docker_shim
    
    # Setup Podman socket service
    setup_podman_socket
    
    # Create environment configuration
    create_environment_config
    
    # Test basic container operations
    test_container_operations
    
    # Create validation script
    create_validation_script
    
    echo ""
    echo "🎉 Docker shim setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Restart your shell: source ~/.bashrc"
    echo "2. Test the setup: bash scripts/validate-docker-shim.sh"
    echo "3. Open VS Code and try Dev Containers"
    echo ""
    echo "Validation:"
    echo "  bash scripts/validate-docker-shim.sh"
    echo ""
    echo "Environment variables:"
    echo "  PODMAN_HOST: $PODMAN_HOST"
    echo "  DOCKER_HOST: $DOCKER_HOST"
}

# Run main function
main "$@"
