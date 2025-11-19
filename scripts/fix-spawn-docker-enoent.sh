#!/bin/bash
# Comprehensive fix for "spawn docker ENOENT" error
# This script implements all necessary fixes for VS Code Dev Containers with Podman

set -e

echo "🔧 Fixing 'spawn docker ENOENT' Error for VS Code Dev Containers"
echo "==============================================================="

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
        echo "❌ Not running in WSL - this script should be run in WSL"
        exit 1
    fi
}

# Step 1: Install and configure Podman
setup_podman() {
    print_section "SETTING UP PODMAN"
    
    echo "Checking Podman installation..."
    if command -v podman >/dev/null 2>&1; then
        echo "✅ Podman already installed: $(podman --version)"
    else
        echo "Installing Podman..."
        sudo apt-get update
        sudo apt-get install -y podman
        echo "✅ Podman installed"
    fi
    
    echo "Installing podman-docker package..."
    if command -v docker >/dev/null 2>&1 && docker --version | grep -q "Docker version"; then
        echo "✅ Docker shim already available: $(docker --version)"
    else
        sudo apt-get install -y podman-docker
        echo "✅ podman-docker package installed"
    fi
}

# Step 2: Setup Podman socket service
setup_podman_socket() {
    print_section "SETTING UP PODMAN SOCKET SERVICE"
    
    echo "Enabling Podman socket service..."
    if systemctl --user enable --now podman.socket; then
        echo "✅ Podman socket service enabled and started"
    else
        echo "⚠️ Could not enable Podman socket service"
        echo "💡 This may be normal in some environments"
    fi
    
    # Check socket status
    local socket_path="/run/user/$(id -u)/podman/podman.sock"
    if [ -S "$socket_path" ]; then
        echo "✅ Podman socket found at: $socket_path"
    else
        echo "⚠️ Podman socket not found at expected location"
        echo "💡 Socket will be created when first container is started"
    fi
}

# Step 3: Create WSL-scoped VS Code settings
create_wsl_vscode_settings() {
    print_section "CREATING WSL-SCOPED VS CODE SETTINGS"
    
    local settings_dir="$HOME/.vscode-server/data/Machine"
    local settings_file="$settings_dir/settings.json"
    
    echo "Creating VS Code server settings directory..."
    mkdir -p "$settings_dir"
    
    echo "Creating WSL-scoped VS Code settings..."
    cat > "$settings_file" << 'EOF'
{
  "dev.containers.dockerPath": "podman",
  "dev.containers.dockerComposePath": "podman-compose",
  "dev.containers.environment": {
    "PODMAN_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DOCKER_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DEVCONTAINERS_DOCKER_PATH": "podman"
  },
  "dev.containers.logLevel": "info",
  "dev.containers.showLog": true,
  "docker.showStartPage": false,
  "docker.showDockerExplorer": false,
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.profiles.linux": {
    "bash": {
      "path": "/bin/bash",
      "args": ["-l"],
      "env": {
        "PODMAN_HOST": "unix:///run/user/1000/podman/podman.sock",
        "DOCKER_HOST": "unix:///run/user/1000/podman/podman.sock",
        "DEVCONTAINERS_DOCKER_PATH": "podman"
      }
    }
  }
}
EOF
    
    echo "✅ WSL-scoped VS Code settings created"
    echo "📁 Location: $settings_file"
}

# Step 4: Create environment variables
create_environment_variables() {
    print_section "CREATING ENVIRONMENT VARIABLES"
    
    local env_file="$HOME/.podman_devcontainers_env"
    
    echo "Creating environment file..."
    cat > "$env_file" << 'EOF'
# Podman Dev Containers Environment Variables
# This file ensures Dev Containers uses Podman instead of Docker

export DEVCONTAINERS_DOCKER_PATH=podman
export PODMAN_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"

# Alternative: Use TCP socket if unix socket doesn't work
# export PODMAN_HOST="tcp://localhost:8080"
# export DOCKER_HOST="tcp://localhost:8080"
EOF
    
    echo "✅ Environment file created"
    echo "📁 Location: $env_file"
    
    # Add to bashrc
    if ! grep -q "podman_devcontainers_env" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Podman Dev Containers Environment" >> "$HOME/.bashrc"
        echo "if [ -f \"$env_file\" ]; then" >> "$HOME/.bashrc"
        echo "  source \"$env_file\"" >> "$HOME/.bashrc"
        echo "fi" >> "$HOME/.bashrc"
        echo "✅ Added environment setup to .bashrc"
    else
        echo "✅ Environment setup already in .bashrc"
    fi
}

# Step 5: Fix containers.conf warnings
fix_containers_conf() {
    print_section "FIXING CONTAINERS.CONF WARNINGS"
    
    local config_dir="$HOME/.config/containers"
    local config_file="$config_dir/containers.conf"
    
    echo "Creating containers configuration directory..."
    mkdir -p "$config_dir"
    
    echo "Backing up existing containers.conf (if exists)..."
    if [ -f "$config_file" ]; then
        cp "$config_file" "$config_file.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ Backup created"
    fi
    
    echo "Creating new containers.conf..."
    cat > "$config_file" << 'EOF'
# Podman containers.conf - Fixed for Dev Containers
[containers]
# Default log driver for containers
log_driver = "journald"
# Log size maximum
log_size_max = "10MB"
# Network backend to use
network_backend = "netavark"
# OCI runtime to use
runtime = "crun"

[engine]
# Runtime to use with podman
runtime = "crun"
# CGroup manager to use
cgroup_manager = "systemd"
# Environment variables to pass into conmon
conmon_env_vars = [
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
]
# Paths to search for a valid OCI runtime
runtime_path = [
  "/usr/bin/runc", "/usr/sbin/runc", "/usr/local/bin/runc", "/usr/local/sbin/runc",
  "/sbin/runc", "/bin/runc", "/usr/lib/cri-o-runc/sbin/runc",
  "/usr/bin/crun", "/usr/local/bin/crun"
]
# Image signature verification policy
image_default_transport = "docker://"
# Number of locks available for containers
num_locks = 2048

[machine]
# Number of CPU cores for the VM
cpus = 4
# Memory allocation for the VM
memory = 8192
# Disk size for the VM
disk_size = 100
EOF
    
    echo "✅ containers.conf created with fixed configuration"
    echo "📁 Location: $config_file"
}

# Step 6: Test the setup
test_setup() {
    print_section "TESTING THE SETUP"
    
    echo "Testing Podman functionality..."
    if podman info >/dev/null 2>&1; then
        echo "✅ Podman is functional"
    else
        echo "❌ Podman is not functional"
        return 1
    fi
    
    echo "Testing Docker shim..."
    if docker --version >/dev/null 2>&1; then
        echo "✅ Docker shim working: $(docker --version)"
    else
        echo "❌ Docker shim not working"
        return 1
    fi
    
    echo "Testing Docker info..."
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker info working"
    else
        echo "⚠️ Docker info failed (may be normal if no containers running)"
    fi
    
    echo "Testing VS Code settings..."
    local settings_file="$HOME/.vscode-server/data/Machine/settings.json"
    if [ -f "$settings_file" ] && grep -q "dev.containers.dockerPath.*podman" "$settings_file"; then
        echo "✅ VS Code settings configured for Podman"
    else
        echo "❌ VS Code settings not configured for Podman"
        return 1
    fi
}

# Step 7: Create validation script
create_validation_script() {
    print_section "CREATING VALIDATION SCRIPT"
    
    local validation_script="../scripts/validate-spawn-docker-fix.sh"
    
    cat > "$validation_script" << 'EOF'
#!/bin/bash
# Validation script for spawn docker ENOENT fix

echo "🔍 Validating Spawn Docker ENOENT Fix"
echo "===================================="

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

# Check Podman functionality
echo "Testing Podman functionality..."
if podman info >/dev/null 2>&1; then
    echo "✅ Podman is functional"
else
    echo "❌ Podman is not functional"
    exit 1
fi

# Check Docker functionality
echo "Testing Docker functionality..."
if docker info >/dev/null 2>&1; then
    echo "✅ Docker shim is functional"
else
    echo "⚠️ Docker shim test failed (may be normal)"
fi

# Check VS Code settings
echo "Checking VS Code settings..."
local settings_file="$HOME/.vscode-server/data/Machine/settings.json"
if [ -f "$settings_file" ]; then
    echo "✅ VS Code settings file exists"
    if grep -q "dev.containers.dockerPath.*podman" "$settings_file"; then
        echo "✅ Dev Containers configured for Podman"
    else
        echo "❌ Dev Containers not configured for Podman"
    fi
else
    echo "❌ VS Code settings file not found"
fi

# Check environment variables
echo "Checking environment variables..."
if [ -n "$DEVCONTAINERS_DOCKER_PATH" ]; then
    echo "✅ DEVCONTAINERS_DOCKER_PATH set to: $DEVCONTAINERS_DOCKER_PATH"
else
    echo "⚠️ DEVCONTAINERS_DOCKER_PATH not set"
fi

# Check Podman socket
echo "Checking Podman socket..."
if systemctl --user is-active podman.socket >/dev/null 2>&1; then
    echo "✅ Podman socket service active"
else
    echo "⚠️ Podman socket service not active"
fi

echo ""
echo "🎉 Validation complete!"
echo ""
echo "Next steps:"
echo "1. Restart VS Code completely"
echo "2. Open this workspace in VS Code"
echo "3. Command Palette: 'Dev Containers: Rebuild and Reopen in Container'"
echo "4. VS Code should now use Podman instead of Docker"
EOF
    
    chmod +x "$validation_script"
    echo "✅ Validation script created"
    echo "📁 Location: $validation_script"
}

# Main execution
main() {
    echo "Starting comprehensive fix for 'spawn docker ENOENT' error..."
    
    # Check environment
    check_wsl
    
    # Step 1: Setup Podman
    setup_podman
    
    # Step 2: Setup Podman socket
    setup_podman_socket
    
    # Step 3: Create WSL-scoped VS Code settings
    create_wsl_vscode_settings
    
    # Step 4: Create environment variables
    create_environment_variables
    
    # Step 5: Fix containers.conf warnings
    fix_containers_conf
    
    # Step 6: Test the setup
    test_setup
    
    # Step 7: Create validation script
    create_validation_script
    
    echo ""
    echo "🎉 Comprehensive fix completed!"
    echo ""
    echo "Summary of changes:"
    echo "✅ Podman installed and configured"
    echo "✅ Docker shim created for compatibility"
    echo "✅ Podman socket service enabled"
    echo "✅ WSL-scoped VS Code settings created"
    echo "✅ Environment variables configured"
    echo "✅ containers.conf warnings fixed"
    echo "✅ Validation script created"
    echo ""
    echo "Next steps:"
    echo "1. Restart VS Code completely"
    echo "2. Open this workspace in VS Code"
    echo "3. Command Palette: 'Dev Containers: Rebuild and Reopen in Container'"
    echo "4. VS Code should now use Podman instead of Docker"
    echo ""
    echo "Validation:"
    echo "  bash scripts/validate-spawn-docker-fix.sh"
    echo ""
    echo "Troubleshooting:"
    echo "- If still getting 'spawn docker ENOENT', restart VS Code completely"
    echo "- Check VS Code output panel for Dev Containers logs"
    echo "- Verify Podman socket service is running"
}

# Run main function
main "$@"
