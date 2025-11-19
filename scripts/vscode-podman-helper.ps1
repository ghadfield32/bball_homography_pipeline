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
        Write-Host ""
        Write-Host "Podman Version:" -ForegroundColor Yellow
        Invoke-WSLCommand "podman --version"
        Write-Host ""
        Write-Host "Docker Compatibility:" -ForegroundColor Yellow
        Invoke-WSLCommand "docker --version"
        Write-Host ""
        Write-Host "Podman Socket Service:" -ForegroundColor Yellow
        Invoke-WSLCommand "systemctl --user is-active podman.socket"
        Write-Host ""
        Write-Host "VS Code Settings:" -ForegroundColor Yellow
        if (Test-Path ".vscode\settings.json") {
            $settings = Get-Content ".vscode\settings.json" -Raw
            if ($settings -match "dev\.containers\.dockerPath.*podman") {
                Write-Host "✅ VS Code configured to use Podman" -ForegroundColor Green
            } else {
                Write-Host "❌ VS Code not configured to use Podman" -ForegroundColor Red
            }
        } else {
            Write-Host "❌ VS Code settings file not found" -ForegroundColor Red
        }
    }
    
    "setup" {
        Write-Host "⚙️ Setting up Podman Docker compatibility..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/setup-podman-docker-shim.sh"
    }
    
    "fix-spawn-error" {
        Write-Host "🔧 Fixing 'spawn docker ENOENT' error..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/fix-spawn-docker-enoent.sh"
    }
    
    "wsl-settings" {
        Write-Host "⚙️ Creating WSL-scoped VS Code settings..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/create-wsl-vscode-settings.sh"
    }
    
    "docker-shim" {
        Write-Host "🐳 Setting up Docker shim for Podman..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/setup-docker-shim.sh"
    }
    
    "test" {
        Write-Host "🧪 Testing VS Code Dev Container integration..." -ForegroundColor Green
        Write-Host ""
        Write-Host "To test VS Code Dev Container integration:" -ForegroundColor Yellow
        Write-Host "1. Open VS Code in this directory" -ForegroundColor White
        Write-Host "2. Press Ctrl+Shift+P" -ForegroundColor White
        Write-Host "3. Run: 'Dev Containers: Reopen in Container'" -ForegroundColor White
        Write-Host "4. VS Code should use Podman to build and run the container" -ForegroundColor White
        Write-Host ""
        Write-Host "Expected behavior:" -ForegroundColor Yellow
        Write-Host "- Container builds using Podman instead of Docker" -ForegroundColor White
        Write-Host "- All VS Code extensions work normally" -ForegroundColor White
        Write-Host "- GPU access works through Podman's CDI system" -ForegroundColor White
        Write-Host "- Port forwarding works correctly" -ForegroundColor White
    }
    
    "logs" {
        Write-Host "📋 Showing Podman logs..." -ForegroundColor Green
        Invoke-WSLCommand "journalctl --user -u podman.socket -f"
    }
    
    "restart" {
        Write-Host "🔄 Restarting Podman socket service..." -ForegroundColor Green
        Invoke-WSLCommand "systemctl --user restart podman.socket"
        Write-Host "✅ Podman socket service restarted" -ForegroundColor Green
    }
    
    "validate" {
        Write-Host "🔍 Running comprehensive validation..." -ForegroundColor Green
        Invoke-WSLCommand "cd '$wslPath' && bash scripts/validate-podman-vscode.sh"
    }
    
    "clean" {
        Write-Host "🧹 Cleaning up Podman resources..." -ForegroundColor Green
        Invoke-WSLCommand "podman system prune -af"
        Write-Host "✅ Podman resources cleaned" -ForegroundColor Green
    }
    
    "devcontainer" {
        Write-Host "🐳 Dev Container specific commands..." -ForegroundColor Green
        Write-Host ""
        Write-Host "Available Dev Container commands:" -ForegroundColor Yellow
        Write-Host "1. Open VS Code: code ." -ForegroundColor White
        Write-Host "2. Command Palette: Ctrl+Shift+P" -ForegroundColor White
        Write-Host "3. 'Dev Containers: Reopen in Container'" -ForegroundColor White
        Write-Host "4. 'Dev Containers: Rebuild Container'" -ForegroundColor White
        Write-Host "5. 'Dev Containers: Rebuild and Reopen in Container'" -ForegroundColor White
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "- If container fails to start, check Podman logs" -ForegroundColor White
        Write-Host "- Ensure NVIDIA Container Toolkit is installed" -ForegroundColor White
        Write-Host "- Verify Podman socket service is running" -ForegroundColor White
    }
    
    default {
        Write-Host "❓ Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor Yellow
        Write-Host " status         - Check Podman integration status" -ForegroundColor White
        Write-Host " setup          - Setup Podman Docker compatibility" -ForegroundColor White
        Write-Host " fix-spawn-error - Fix 'spawn docker ENOENT' error" -ForegroundColor White
        Write-Host " wsl-settings   - Create WSL-scoped VS Code settings" -ForegroundColor White
        Write-Host " docker-shim    - Setup Docker shim for Podman" -ForegroundColor White
        Write-Host " test           - Test VS Code Dev Container integration" -ForegroundColor White
        Write-Host " logs           - Show Podman logs" -ForegroundColor White
        Write-Host " restart        - Restart Podman socket service" -ForegroundColor White
        Write-Host " validate       - Run comprehensive validation" -ForegroundColor White
        Write-Host " clean          - Clean up Podman resources" -ForegroundColor White
        Write-Host " devcontainer   - Show Dev Container specific commands" -ForegroundColor White
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host " .\scripts\vscode-podman-helper.ps1 setup" -ForegroundColor White
        Write-Host " .\scripts\vscode-podman-helper.ps1 status" -ForegroundColor White
        Write-Host " .\scripts\vscode-podman-helper.ps1 validate" -ForegroundColor White
    }
}
