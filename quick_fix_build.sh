#!/bin/bash
# Quick fix for Docker build failures
# This script provides immediate fixes without moving files

set -e

echo "================================================================="
echo "Quick Docker Build Fix"
echo "================================================================="
echo ""

# 1. Clean corrupted Docker cache
echo "1. Cleaning corrupted Docker BuildKit cache..."
docker builder prune -af 2>/dev/null || echo "BuildKit prune completed"
echo "✓ BuildKit cache cleared"
echo ""

# 2. Clean old images and containers
echo "2. Cleaning old Docker artifacts..."
docker system prune -f 2>/dev/null || echo "System prune completed"
echo "✓ Old artifacts removed"
echo ""

# 3. Verify .dockerignore
echo "3. Verifying .dockerignore..."
if [ -f .dockerignore ]; then
    echo "✓ .dockerignore exists"
    echo "  Lines: $(wc -l < .dockerignore)"
else
    echo "⚠️  WARNING: No .dockerignore file!"
fi
echo ""

# 4. Check filesystem
echo "4. Checking filesystem..."
MOUNT=$(df -h . | tail -1 | awk '{print $6}')
if [[ "$MOUNT" == /mnt/* ]] || [[ "$MOUNT" == /c/* ]]; then
    echo "⚠️  WARNING: Building from Windows mount ($MOUNT)"
    echo "   This causes slow builds. Consider moving to WSL filesystem."
else
    echo "✓ On WSL/Linux filesystem"
fi
echo ""

# 5. Show disk usage
echo "5. Docker disk usage:"
docker system df
echo ""

echo "================================================================="
echo "Ready to build!"
echo "================================================================="
echo ""
echo "Next steps:"
echo "1. cd .devcontainer"
echo "2. docker-compose build --no-cache datascience"
echo ""
echo "If build still fails due to slow I/O:"
echo "  Run: bash fix_build_performance.sh --move-to-wsl"
echo ""
