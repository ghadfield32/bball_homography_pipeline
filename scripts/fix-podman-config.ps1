# Fix Podman containers.conf warning by setting a supported log_driver
# This script addresses the "Failed to decode the keys ["engine.log_driver"]" warning

Write-Host "🔧 Fixing Podman containers.conf configuration..." -ForegroundColor Green

# Create the containers config directory if it doesn't exist
$ConfigDir = "$env:USERPROFILE\.config\containers"
if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
    Write-Host "✅ Created containers config directory: $ConfigDir" -ForegroundColor Green
}

$ConfigFile = "$ConfigDir\containers.conf"

# Check if config file exists and what's in it
if (Test-Path $ConfigFile) {
    Write-Host "📄 Current containers.conf content:" -ForegroundColor Yellow
    Get-Content $ConfigFile
    Write-Host ""
    
    # Check for problematic log_driver settings
    $CurrentContent = Get-Content $ConfigFile -Raw
    if ($CurrentContent -match "log_driver\s*=") {
        Write-Host "⚠️ Found existing log_driver setting that may cause warnings" -ForegroundColor Yellow
    }
}

# Create or update the config file with a supported log_driver
Write-Host "📝 Setting up containers.conf with supported log_driver..." -ForegroundColor Green

$ConfigContent = @"
# Podman containers configuration
# This fixes the "Failed to decode the keys ["engine.log_driver"]" warning

[engine]
# Use k8s-file log driver (supported by Podman)
log_driver = "k8s-file"

# Alternative log drivers you can use:
# log_driver = "journald"  # For systems with systemd/journald
# log_driver = "none"      # Disable logging

[containers]
# Additional container settings for better compatibility
netns = "host"
userns = "host"
ipcns = "host"
utsns = "host"
cgroupns = "host"

# GPU and device access settings
[engine.volume_plugins]
# Enable volume plugins for better compatibility

[network]
# Network settings for better WSL2 compatibility
default_rootless_network_cmd = "slirp4netns"
"@

Set-Content -Path $ConfigFile -Value $ConfigContent -Encoding UTF8

Write-Host "✅ Updated containers.conf with supported log_driver" -ForegroundColor Green
Write-Host "📄 New configuration:" -ForegroundColor Yellow
Get-Content $ConfigFile

Write-Host ""
Write-Host "🔍 Testing Podman configuration..." -ForegroundColor Green
if (Get-Command podman -ErrorAction SilentlyContinue) {
    try {
        $PodmanInfo = podman info 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Podman configuration is valid" -ForegroundColor Green
            Write-Host "📊 Podman version info:" -ForegroundColor Cyan
            podman --version
        } else {
            Write-Host "❌ Podman configuration has issues:" -ForegroundColor Red
            Write-Host $PodmanInfo -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Podman configuration has issues: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "ℹ️ Podman not found in PATH" -ForegroundColor Yellow
    Write-Host "   Make sure Podman is installed and added to your PATH" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Next steps for WSL2 + Podman + GPU setup:" -ForegroundColor Cyan
Write-Host "1. Enter WSL2: wsl -d Ubuntu-22.04" -ForegroundColor White
Write-Host "2. Install NVIDIA Container Toolkit in WSL2:" -ForegroundColor White
Write-Host "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" -ForegroundColor Gray
Write-Host "3. Generate NVIDIA CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" -ForegroundColor White
Write-Host "4. Test GPU access: podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi" -ForegroundColor White
Write-Host "5. Build your project: podman compose -f .devcontainer/docker-compose.yml build" -ForegroundColor White