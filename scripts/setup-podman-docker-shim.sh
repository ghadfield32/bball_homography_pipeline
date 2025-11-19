#!/bin/bash
# ========================================
# PODMAN DOCKER COMPATIBILITY SHIM SETUP
# ========================================
# This script creates a Docker compatibility layer for Podman
# to ensure seamless integration with VS Code Dev Containers

set -e

echo "🐳 Setting up Podman Docker Compatibility Shim"
echo "=============================================="

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

# Function to install podman-docker package
install_podman_docker() {
    print_section "INSTALLING PODMAN-DOCKER PACKAGE"
    
    echo "Installing podman-docker package for Docker CLI compatibility..."
    
    # Update package list
    sudo apt-get update
    
    # Install podman-docker package
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
    
    # Verify installation
    echo "Verifying Docker compatibility..."
    if docker --version >/dev/null 2>&1; then
        echo "✅ Docker compatibility working"
        docker --version
    else
        echo "❌ Docker compatibility failed"
        return 1
    fi
}

# Function to setup Podman socket service
setup_podman_socket() {
    print_section "SETTING UP PODMAN SOCKET SERVICE"
    
    echo "Enabling Podman socket service for Docker API compatibility..."
    
    # Enable and start Podman socket service
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
    
    # Set environment variables
    echo "Setting up environment variables..."
    local env_file="$HOME/.podman_docker_env"
    cat > "$env_file" << 'EOF'
# Podman Docker API Environment Variables
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export PODMAN_HOST="unix:///run/user/$(id -u)/podman/podman.sock"

# Alternative: Use TCP socket (if unix socket doesn't work)
# export DOCKER_HOST="tcp://localhost:8080"
# export PODMAN_HOST="tcp://localhost:8080"
EOF
    
    echo "✅ Environment file created: $env_file"
    
    # Add to bashrc if not already present
    if ! grep -q "podman_docker_env" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Podman Docker API Environment" >> "$HOME/.bashrc"
        echo "if [ -f \"$env_file\" ]; then" >> "$HOME/.bashrc"
        echo "  source \"$env_file\"" >> "$HOME/.bashrc"
        echo "fi" >> "$HOME/.bashrc"
        echo "✅ Added environment setup to .bashrc"
    else
        echo "✅ Environment setup already in .bashrc"
    fi
}

# Function to test Docker API compatibility
test_docker_api() {
    print_section "TESTING DOCKER API COMPATIBILITY"
    
    echo "Testing Docker API through Podman socket..."
    
    # Source environment variables
    if [ -f "$HOME/.podman_docker_env" ]; then
        source "$HOME/.podman_docker_env"
    fi
    
    # Test Docker info command
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker info command working"
        echo "Container runtime: $(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo 'Unknown')"
    else
        echo "❌ Docker info command failed"
        echo "💡 This may be normal if no containers are running"
    fi
    
    # Test Docker version command
    if docker version >/dev/null 2>&1; then
        echo "✅ Docker version command working"
    else
        echo "❌ Docker version command failed"
    fi
    
    # Test basic container operations
    echo "Testing basic container operations..."
    if docker run --rm hello-world >/dev/null 2>&1; then
        echo "✅ Container run test successful"
    else
        echo "⚠️ Container run test failed (may be normal if hello-world image not available)"
    fi
}

# Function to create VS Code integration helper
create_vscode_helper() {
    print_section "CREATING VS CODE INTEGRATION HELPER"
    
    local helper_script="../scripts/vscode-podman-helper.ps1"
    mkdir -p "$(dirname "$helper_script")"
    
    cat > "$helper_script" << 'EOF'
# PowerShell Helper Script for VS Code Podman Integration
# Usage: .\scripts\vscode-podman-helper.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "status"
)

Write-Host "🔧 VS Code Podman Integration Helper" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Function to run commands in WSL
function Invoke-WSLCommand {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $null
    )
    
    if ($WorkingDirectory) {
        $wslCommand = "cd '$WorkingDirectory' && $Command"
    } else {
        $wslCommand = $Command
    }
    
    Write-Host "🔄 Executing in WSL: $wslCommand" -ForegroundColor Yellow
    wsl bash -c $wslCommand
}

# Check if we're in the right directory
$currentDir = Get-Location
if (-not (Test-Path ".vscode")) {
    Write-Host "❌ Error: .vscode directory not found" -ForegroundColor Red
    Write-Host "💡 Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}

# Convert Windows path to WSL path
$wslPath = wsl wslpath $currentDir.Path
Write-Host "📁 Project directory (WSL): $wslPath" -ForegroundColor Green

switch ($Command.ToLower()) {
    "status" {
        Write-Host "📊 Checking Podman integration status..." -ForegroundColor Green
        Invoke-WSLCommand "podman --version"
        Invoke-WSLCommand "docker --version"
        Invoke-WSLCommand "systemctl --user is-active podman.socket"
    }
    
    "setup" {
        Write-Host "⚙️ Setting up Podman Docker compatibility..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/setup-podman-docker-shim.sh"
    }
    
    "test" {
        Write-Host "🧪 Testing VS Code Dev Container integration..." -ForegroundColor Green
        Write-Host "1. Open VS Code in this directory" -ForegroundColor Yellow
        Write-Host "2. Press Ctrl+Shift+P" -ForegroundColor Yellow
        Write-Host "3. Run: 'Dev Containers: Reopen in Container'" -ForegroundColor Yellow
        Write-Host "4. VS Code should use Podman to build and run the container" -ForegroundColor Yellow
    }
    
    "logs" {
        Write-Host "📋 Showing Podman logs..." -ForegroundColor Green
        Invoke-WSLCommand "journalctl --user -u podman.socket -f"
    }
    
    "restart" {
        Write-Host "🔄 Restarting Podman socket service..." -ForegroundColor Green
        Invoke-WSLCommand "systemctl --user restart podman.socket"
    }
    
    default {
        Write-Host "❓ Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor Yellow
        Write-Host " status - Check Podman integration status" -ForegroundColor White
        Write-Host " setup - Setup Podman Docker compatibility" -ForegroundColor White
        Write-Host " test - Test VS Code Dev Container integration" -ForegroundColor White
        Write-Host " logs - Show Podman logs" -ForegroundColor White
        Write-Host " restart - Restart Podman socket service" -ForegroundColor White
        Write-Host ""
        Write-Host "Example: .\scripts\vscode-podman-helper.ps1 setup" -ForegroundColor Cyan
    }
}
EOF
    
    echo "✅ VS Code helper script created"
    echo "📁 Location: $helper_script"
    echo "💡 Usage from PowerShell: .\scripts\vscode-podman-helper.ps1 setup"
}

# Function to create validation script
create_validation_script() {
    print_section "CREATING VALIDATION SCRIPT"
    
    local validation_script="../scripts/validate-podman-vscode.sh"
    
    cat > "$validation_script" << 'EOF'
#!/bin/bash
# Validation script for Podman VS Code integration

echo "🔍 Validating Podman VS Code Integration"
echo "========================================"

# Check Podman installation
echo "Checking Podman installation..."
if command -v podman >/dev/null 2>&1; then
    echo "✅ Podman installed: $(podman --version)"
else
    echo "❌ Podman not found"
    exit 1
fi

# Check Docker compatibility
echo "Checking Docker compatibility..."
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker compatibility available: $(docker --version)"
else
    echo "❌ Docker compatibility not available"
    exit 1
fi

# Check Podman socket
echo "Checking Podman socket..."
if systemctl --user is-active podman.socket >/dev/null 2>&1; then
    echo "✅ Podman socket service active"
else
    echo "⚠️ Podman socket service not active"
fi

# Check VS Code settings
echo "Checking VS Code settings..."
if [ -f ".vscode/settings.json" ]; then
    if grep -q "dev.containers.dockerPath.*podman" ".vscode/settings.json"; then
        echo "✅ VS Code configured to use Podman"
    else
        echo "❌ VS Code not configured to use Podman"
    fi
else
    echo "❌ VS Code settings file not found"
fi

# Check devcontainer configuration
echo "Checking devcontainer configuration..."
if [ -f ".devcontainer/devcontainer.json" ]; then
    echo "✅ devcontainer.json found"
else
    echo "❌ devcontainer.json not found"
fi

echo ""
echo "🎉 Validation complete!"
echo ""
echo "Next steps:"
echo "1. Open this workspace in VS Code"
echo "2. Command Palette: 'Dev Containers: Reopen in Container'"
echo "3. VS Code will use Podman to build and run the container"
EOF
    
    chmod +x "$validation_script"
    echo "✅ Validation script created"
    echo "📁 Location: $validation_script"
    echo "💡 Usage: bash scripts/validate-podman-vscode.sh"
}

# Main execution
main() {
    echo "Starting Podman Docker compatibility setup..."
    
    # Check environment
    check_wsl
    
    # Install podman-docker package
    install_podman_docker
    
    # Setup Podman socket service
    setup_podman_socket
    
    # Test Docker API compatibility
    test_docker_api
    
    # Create VS Code integration helper
    create_vscode_helper
    
    # Create validation script
    create_validation_script
    
    echo ""
    echo "🎉 Podman Docker compatibility setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Restart your shell: source ~/.bashrc"
    echo "2. Open VS Code in this directory"
    echo "3. Command Palette: 'Dev Containers: Reopen in Container'"
    echo "4. VS Code will use Podman instead of Docker"
    echo ""
    echo "Validation:"
    echo "  bash scripts/validate-podman-vscode.sh"
    echo ""
    echo "PowerShell helper:"
    echo "  .\scripts\vscode-podman-helper.ps1 status"
}

# Run main function
main "$@"
