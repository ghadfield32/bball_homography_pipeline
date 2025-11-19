#!/bin/bash
# Fix containers.conf configuration for Podman

echo "🔧 Fixing containers.conf configuration"
echo "======================================"

# Create proper containers.conf
cat > ~/.config/containers/containers.conf << 'EOF'
[containers]
log_driver = "journald"
log_size_max = 10485760
network_backend = "netavark"
runtime = "crun"

[engine]
runtime = "crun"
cgroup_manager = "systemd"
conmon_env_vars = [
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
]
runtime_path = [
  "/usr/bin/runc", "/usr/sbin/runc", "/usr/local/bin/runc", "/usr/local/sbin/runc",
  "/sbin/runc", "/bin/runc", "/usr/lib/cri-o-runc/sbin/runc",
  "/usr/bin/crun", "/usr/local/bin/crun"
]
image_default_transport = "docker://"
num_locks = 2048

[machine]
cpus = 4
memory = 8192
disk_size = 100
EOF

echo "✅ containers.conf fixed"
echo "Testing Podman..."
if podman info >/dev/null 2>&1; then
    echo "✅ Podman is now functional"
else
    echo "❌ Podman still not functional"
    exit 1
fi

echo "Testing Docker shim..."
if docker --version >/dev/null 2>&1; then
    echo "✅ Docker shim working: $(docker --version)"
else
    echo "❌ Docker shim not working"
    exit 1
fi

echo "🎉 containers.conf fix completed!"
