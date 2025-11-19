# Fix Podman containers.conf warning
Write-Host "Fixing Podman containers.conf configuration..." -ForegroundColor Green

# Create config directory
$ConfigDir = "$env:USERPROFILE\.config\containers"
if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
    Write-Host "Created containers config directory" -ForegroundColor Green
}

$ConfigFile = "$ConfigDir\containers.conf"

# Create config content
$ConfigContent = @"
[engine]
log_driver = "k8s-file"

[containers]
netns = "host"
userns = "host"
ipcns = "host"
utsns = "host"
cgroupns = "host"
"@

Set-Content -Path $ConfigFile -Value $ConfigContent -Encoding UTF8
Write-Host "Updated containers.conf" -ForegroundColor Green

# Test Podman
if (Get-Command podman -ErrorAction SilentlyContinue) {
    Write-Host "Podman found in PATH" -ForegroundColor Green
    podman --version
} else {
    Write-Host "Podman not found in PATH" -ForegroundColor Yellow
}

Write-Host "Next: Enter WSL2 and run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" -ForegroundColor Cyan
