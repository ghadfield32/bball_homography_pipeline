#!/bin/bash
# Fix Podman containers.conf warning by setting a supported log_driver
# This script addresses the "Failed to decode the keys ["engine.log_driver"]" warning

set -e

echo "🔧 Fixing Podman containers.conf configuration..."

# Create the containers config directory if it doesn't exist
CONFIG_DIR="$HOME/.config/containers"
mkdir -p "$CONFIG_DIR"

CONFIG_FILE="$CONFIG_DIR/containers.conf"

# Check if config file exists and what's in it
if [ -f "$CONFIG_FILE" ]; then
    echo "📄 Current containers.conf content:"
    cat "$CONFIG_FILE"
    echo ""
fi

# Create or update the config file with a supported log_driver
echo "📝 Setting up containers.conf with supported log_driver..."

cat > "$CONFIG_FILE" << 'EOF'
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
EOF

echo "✅ Updated containers.conf with supported log_driver"
echo "📄 New configuration:"
cat "$CONFIG_FILE"

echo ""
echo "🔍 Testing Podman configuration..."
if command -v podman >/dev/null 2>&1; then
    podman info >/dev/null 2>&1 && echo "✅ Podman configuration is valid" || echo "❌ Podman configuration has issues"
else
    echo "ℹ️ Podman not found in PATH"
fi

echo ""
echo "🎯 Next steps:"
echo "1. If you're using WSL2, ensure NVIDIA Container Toolkit is installed"
echo "2. Generate NVIDIA CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
echo "3. Test GPU access: podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi"
