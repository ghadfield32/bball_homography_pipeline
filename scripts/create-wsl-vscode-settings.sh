#!/bin/bash
# Create WSL-scoped VS Code settings for Podman integration
# This script ensures VS Code Dev Containers uses Podman instead of Docker

set -e

echo "🔧 Creating WSL-scoped VS Code settings for Podman integration"
echo "=============================================================="

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

# Create WSL VS Code server settings directory
create_vscode_server_settings() {
    print_section "CREATING WSL VS CODE SERVER SETTINGS"
    
    local settings_dir="$HOME/.vscode-server/data/Machine"
    local settings_file="$settings_dir/settings.json"
    
    echo "Creating VS Code server settings directory..."
    mkdir -p "$settings_dir"
    
    echo "Creating WSL-scoped VS Code settings for Podman..."
    cat > "$settings_file" << 'EOF'
{
  // ========================================
  // WSL-SCOPED VS CODE SETTINGS FOR PODMAN
  // ========================================
  // These settings ensure VS Code Dev Containers uses Podman instead of Docker
  // when running inside WSL
  
  // CRITICAL: Point Dev Containers to use Podman CLI
  "dev.containers.dockerPath": "podman",
  
  // Use Podman Compose for docker-compose operations
  "dev.containers.dockerComposePath": "podman-compose",
  
  // Podman-specific environment variables
  "dev.containers.environment": {
    "PODMAN_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DOCKER_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DEVCONTAINERS_DOCKER_PATH": "podman"
  },
  
  // Enhanced logging for debugging
  "dev.containers.logLevel": "info",
  "dev.containers.showLog": true,
  
  // Disable Docker extension features that conflict with Podman
  "docker.showStartPage": false,
  "docker.showDockerExplorer": false,
  
  // Podman-specific container settings
  "dev.containers.defaultExtensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-azuretools.vscode-docker"
  ],
  
  // Terminal configuration for Podman environment
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
    
    # Verify the file was created correctly
    if [ -f "$settings_file" ]; then
        echo "✅ Settings file verified"
        echo "📄 Content preview:"
        head -10 "$settings_file"
    else
        echo "❌ Settings file creation failed"
        return 1
    fi
}

# Create User-scoped settings as backup
create_user_settings() {
    print_section "CREATING USER-SCOPED SETTINGS BACKUP"
    
    local user_settings_dir="$HOME/.vscode-server/data/User"
    local user_settings_file="$user_settings_dir/settings.json"
    
    echo "Creating user settings directory..."
    mkdir -p "$user_settings_dir"
    
    # Check if user settings already exist
    if [ -f "$user_settings_file" ]; then
        echo "⚠️ User settings file already exists, backing up..."
        cp "$user_settings_file" "$user_settings_file.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    echo "Creating user-scoped Podman settings..."
    cat > "$user_settings_file" << 'EOF'
{
  // ========================================
  // USER-SCOPED PODMAN SETTINGS
  // ========================================
  // Backup settings for Podman integration
  
  "dev.containers.dockerPath": "podman",
  "dev.containers.dockerComposePath": "podman-compose",
  "dev.containers.environment": {
    "PODMAN_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DOCKER_HOST": "unix:///run/user/1000/podman/podman.sock",
    "DEVCONTAINERS_DOCKER_PATH": "podman"
  },
  "dev.containers.logLevel": "info",
  "docker.showStartPage": false
}
EOF
    
    echo "✅ User-scoped settings created"
    echo "📁 Location: $user_settings_file"
}

# Create environment variables for Dev Containers
create_environment_variables() {
    print_section "CREATING ENVIRONMENT VARIABLES"
    
    local env_file="$HOME/.devcontainers_podman_env"
    
    echo "Creating Dev Containers environment file..."
    cat > "$env_file" << 'EOF'
# Dev Containers Podman Environment Variables
# Source this file to ensure Dev Containers uses Podman

export DEVCONTAINERS_DOCKER_PATH=podman
export PODMAN_HOST="unix:///run/user/1000/podman/podman.sock"
export DOCKER_HOST="unix:///run/user/1000/podman/podman.sock"

# Alternative: Use TCP socket if unix socket doesn't work
# export PODMAN_HOST="tcp://localhost:8080"
# export DOCKER_HOST="tcp://localhost:8080"
EOF
    
    echo "✅ Environment file created"
    echo "📁 Location: $env_file"
    
    # Add to bashrc if not already present
    if ! grep -q "devcontainers_podman_env" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Dev Containers Podman Environment" >> "$HOME/.bashrc"
        echo "if [ -f \"$env_file\" ]; then" >> "$HOME/.bashrc"
        echo "  source \"$env_file\"" >> "$HOME/.bashrc"
        echo "fi" >> "$HOME/.bashrc"
        echo "✅ Added environment setup to .bashrc"
    else
        echo "✅ Environment setup already in .bashrc"
    fi
}

# Verify Podman installation
verify_podman_installation() {
    print_section "VERIFYING PODMAN INSTALLATION"
    
    echo "Checking Podman installation..."
    if command -v podman >/dev/null 2>&1; then
        echo "✅ Podman installed: $(podman --version)"
    else
        echo "❌ Podman not found"
        echo "💡 Install with: sudo apt install podman"
        return 1
    fi
    
    echo "Checking Docker compatibility..."
    if command -v docker >/dev/null 2>&1; then
        echo "✅ Docker compatibility available: $(docker --version)"
    else
        echo "⚠️ Docker compatibility not available"
        echo "💡 Install with: sudo apt install podman-docker"
    fi
    
    echo "Testing Podman functionality..."
    if podman info >/dev/null 2>&1; then
        echo "✅ Podman is functional"
    else
        echo "❌ Podman is not functional"
        return 1
    fi
}

# Test VS Code settings
test_vscode_settings() {
    print_section "TESTING VS CODE SETTINGS"
    
    local settings_file="$HOME/.vscode-server/data/Machine/settings.json"
    
    if [ -f "$settings_file" ]; then
        echo "✅ WSL VS Code settings file exists"
        
        if grep -q "dev.containers.dockerPath.*podman" "$settings_file"; then
            echo "✅ Dev Containers configured to use Podman"
        else
            echo "❌ Dev Containers not configured for Podman"
        fi
        
        if grep -q "DEVCONTAINERS_DOCKER_PATH" "$settings_file"; then
            echo "✅ Environment variable configured"
        else
            echo "⚠️ Environment variable not configured"
        fi
    else
        echo "❌ WSL VS Code settings file not found"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting WSL VS Code settings creation for Podman integration..."
    
    # Check environment
    check_wsl
    
    # Verify Podman installation
    verify_podman_installation
    
    # Create VS Code settings
    create_vscode_server_settings
    create_user_settings
    
    # Create environment variables
    create_environment_variables
    
    # Test settings
    test_vscode_settings
    
    echo ""
    echo "🎉 WSL VS Code settings creation completed!"
    echo ""
    echo "Next steps:"
    echo "1. Restart VS Code or reload the window"
    echo "2. Open this workspace in VS Code"
    echo "3. Command Palette: 'Dev Containers: Rebuild and Reopen in Container'"
    echo "4. VS Code should now use Podman instead of Docker"
    echo ""
    echo "Troubleshooting:"
    echo "- If still getting 'spawn docker ENOENT', restart VS Code completely"
    echo "- Check VS Code output panel for Dev Containers logs"
    echo "- Verify Podman socket service is running"
    echo ""
    echo "Validation:"
    echo "  bash scripts/validate-podman-vscode.sh"
}

# Run main function
main "$@"

