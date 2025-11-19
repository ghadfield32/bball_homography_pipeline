# nuclear_docker_reset.ps1
# Complete Docker reset to fix unpigz/deflate corruption
# Run from PowerShell as Administrator

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Nuclear Docker Reset for unpigz Corruption Fix" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop all containers
Write-Host "[Step 1/7] Stopping all containers..." -ForegroundColor Yellow
docker stop $(docker ps -aq) 2>$null
docker rm $(docker ps -aq) 2>$null
Write-Host "✓ Containers stopped and removed" -ForegroundColor Green

# Step 2: Remove all images
Write-Host "[Step 2/7] Removing all images..." -ForegroundColor Yellow
docker rmi -f $(docker images -aq) 2>$null
Write-Host "✓ Images removed" -ForegroundColor Green

# Step 3: Prune everything
Write-Host "[Step 3/7] Pruning all Docker resources..." -ForegroundColor Yellow
docker system prune -a --volumes -f
Write-Host "✓ System pruned" -ForegroundColor Green

# Step 4: Reset BuildKit
Write-Host "[Step 4/7] Resetting BuildKit builders..." -ForegroundColor Yellow
docker buildx prune -a -f
docker buildx rm --all-inactive 2>$null
Write-Host "✓ BuildKit reset" -ForegroundColor Green

# Step 5: Verify cleanup
Write-Host "[Step 5/7] Verifying cleanup..." -ForegroundColor Yellow
Write-Host "Remaining disk usage:"
docker system df
Write-Host ""

# Step 6: Shutdown WSL
Write-Host "[Step 6/7] Shutting down WSL2 to clear memory..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep -Seconds 3
Write-Host "✓ WSL2 shutdown complete" -ForegroundColor Green

# Step 7: Instructions
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Restart Docker Desktop" -ForegroundColor White
Write-Host "2. Wait for Docker to fully initialize (check system tray)" -ForegroundColor White
Write-Host ""
Write-Host "3. Move project to WSL filesystem (RECOMMENDED):" -ForegroundColor Yellow
Write-Host "   wsl -d Ubuntu" -ForegroundColor Gray
Write-Host "   mkdir -p ~/projects" -ForegroundColor Gray
Write-Host "   cp -r /mnt/c/Users/ghadf/vscode_projects/docker_projects/bball_homography_pipeline ~/projects/" -ForegroundColor Gray
Write-Host "   cd ~/projects/bball_homography_pipeline/.devcontainer" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Rebuild from WSL:" -ForegroundColor Yellow
Write-Host "   cp .env.template .env" -ForegroundColor Gray
Write-Host "   docker-compose build --no-cache" -ForegroundColor Gray
Write-Host "   docker-compose up -d" -ForegroundColor Gray
Write-Host ""
Write-Host "OR if staying on Windows mount (higher risk of corruption):" -ForegroundColor Red
Write-Host "   cd .devcontainer" -ForegroundColor Gray
Write-Host "   $env:HOME = $env:USERPROFILE" -ForegroundColor Gray
Write-Host "   docker-compose build --no-cache" -ForegroundColor Gray
Write-Host "   docker-compose up -d" -ForegroundColor Gray
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
