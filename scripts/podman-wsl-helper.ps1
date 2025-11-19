# PowerShell Helper Script for Podman in WSL
# Usage: .\scripts\podman-wsl-helper.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "build"
)

Write-Host "🐳 Podman WSL Helper Script" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

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
if (-not (Test-Path ".devcontainer")) {
    Write-Host "❌ Error: .devcontainer directory not found" -ForegroundColor Red
    Write-Host "💡 Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}

# Convert Windows path to WSL path
$wslPath = wsl wslpath $currentDir.Path
Write-Host "📁 Project directory (WSL): $wslPath" -ForegroundColor Green

switch ($Command.ToLower()) {
    "build" {
        Write-Host "🏗️ Building containers with Podman..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose build --no-cache"
    }
    "up" {
        Write-Host "🚀 Starting containers with Podman..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose up -d"
    }
    "down" {
        Write-Host "🛑 Stopping containers..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose down"
    }
    "logs" {
        Write-Host "📋 Showing container logs..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose logs -f"
    }
    "status" {
        Write-Host "📊 Checking container status..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose ps"
    }
    "test" {
        Write-Host "🧪 Running validation tests..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose exec datascience python /app/validate_gpu.py"
    }
    "shell" {
        Write-Host "🐚 Opening shell in container..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose exec datascience bash"
    }
    "clean" {
        Write-Host "🧹 Cleaning up Podman resources..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && cd .devcontainer && podman compose down --volumes"
        Invoke-WSLCommand "podman system prune -af"
    }
    "config" {
        Write-Host "⚙️ Running Podman configuration fix..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash .devcontainer/podman-config-fix.sh"
    }
    "validate" {
        Write-Host "🔍 Running comprehensive validation..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && python .devcontainer/validate_gpu.py --verbose"
    }
    "cv-test" {
        Write-Host "👁️ Running computer vision tests..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && python .devcontainer/tests/test_yolo.py --verbose"
    }
    default {
        Write-Host "❓ Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor Yellow
        Write-Host "  build     - Build containers"
        Write-Host "  up        - Start containers"
        Write-Host "  down      - Stop containers"
        Write-Host "  logs      - Show container logs"
        Write-Host "  status    - Check container status"
        Write-Host "  test      - Run validation tests"
        Write-Host "  shell     - Open shell in container"
        Write-Host "  clean     - Clean up resources"
        Write-Host "  config    - Fix Podman configuration"
        Write-Host "  validate  - Run comprehensive validation"
        Write-Host "  cv-test   - Run computer vision tests"
        Write-Host ""
        Write-Host "Example: .\scripts\podman-wsl-helper.ps1 build" -ForegroundColor Cyan
    }
}
