#!/usr/bin/env python3
"""
Docker Build Diagnostic & Fix Script
Systematically diagnoses and fixes the tar header corruption issue

Usage:
    python fix_docker_build.py                  # Run diagnostics only
    python fix_docker_build.py --fix-all        # Run diagnostics + apply all fixes
    python fix_docker_build.py --clean-cache    # Clean BuildKit cache only
    python fix_docker_build.py --generate-script # Generate optimized build script
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import json
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str, level: str = "info") -> None:
    """Print a formatted section header"""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    color = colors.get(level, Colors.BLUE)
    print(f"\n{color}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{Colors.END}\n")


def run_command(cmd: List[str], timeout: int = 60, check: bool = False) -> Tuple[bool, str, str]:
    """Run a shell command and return success status, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_wsl_environment() -> Dict[str, any]:
    """Detect if running in WSL and get environment info"""
    print_section("Step 1: Environment Detection", "info")

    env_info = {
        "is_wsl": False,
        "wsl_version": None,
        "on_windows_mount": False,
        "current_path": str(Path.cwd()),
        "docker_available": shutil.which("docker") is not None,
        "buildx_available": False
    }

    # Check if WSL
    try:
        with open("/proc/version", "r") as f:
            version = f.read().lower()
            env_info["is_wsl"] = "microsoft" in version or "wsl" in version
            print(f"WSL Detected: {env_info['is_wsl']}")
            if env_info["is_wsl"]:
                print(f"  Version info: {' '.join(version.split()[0:3])}")
    except:
        print("Not running on Linux/WSL")

    # Check if on Windows mount
    current_path = Path.cwd()
    env_info["on_windows_mount"] = str(current_path).startswith("/mnt/")

    if env_info["on_windows_mount"]:
        print(f"{Colors.YELLOW}⚠️  WARNING: Building from Windows mount: {current_path}{Colors.END}")
        print(f"{Colors.YELLOW}   This is the PRIMARY cause of slow builds and tar errors{Colors.END}")
    else:
        print(f"{Colors.GREEN}✅ Building from WSL filesystem: {current_path}{Colors.END}")

    # Check Docker
    if env_info["docker_available"]:
        ok, out, err = run_command(["docker", "version", "--format", "{{.Server.Version}}"])
        if ok:
            print(f"{Colors.GREEN}✅ Docker version: {out.strip()}{Colors.END}")

        ok, out, err = run_command(["docker", "buildx", "version"])
        env_info["buildx_available"] = ok
        if ok:
            print(f"{Colors.GREEN}✅ BuildX available: {out.strip().split()[0]}{Colors.END}")
    else:
        print(f"{Colors.RED}❌ Docker not found in PATH{Colors.END}")

    return env_info


def check_disk_space() -> Dict[str, any]:
    """Check available disk space"""
    print_section("Step 2: Disk Space Analysis", "info")

    space_info = {
        "root_available_gb": 0,
        "docker_data_gb": 0,
        "sufficient_space": True
    }

    # Check root filesystem
    ok, out, err = run_command(["df", "-BG", "/"])
    if ok:
        lines = out.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 4:
                available = parts[3].rstrip('G')
                try:
                    space_info["root_available_gb"] = float(available)
                    print(f"Root filesystem available: {available} GB")

                    if space_info["root_available_gb"] < 20:
                        print(f"{Colors.RED}❌ WARNING: Less than 20GB available{Colors.END}")
                        print(f"{Colors.RED}   Low disk space is a common cause of tar corruption{Colors.END}")
                        space_info["sufficient_space"] = False
                    else:
                        print(f"{Colors.GREEN}✅ Sufficient disk space available{Colors.END}")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠️  Could not parse disk space{Colors.END}")

    # Check Docker disk usage
    ok, out, err = run_command(["docker", "system", "df", "--format", "{{json .}}"])
    if ok:
        print("\nDocker disk usage:")
        for line in out.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    size = data.get('Size', '0B')
                    reclaimable = data.get('Reclaimable', '0B')
                    print(f"  {data.get('Type', 'Unknown')}: {size} (reclaimable: {reclaimable})")
                except:
                    pass

    return space_info


def analyze_buildkit_cache() -> Dict[str, any]:
    """Analyze BuildKit cache state"""
    print_section("Step 3: BuildKit Cache Analysis", "info")

    cache_info = {
        "cache_size_gb": 0,
        "cache_exists": False,
        "needs_cleanup": False
    }

    # Check BuildKit disk usage
    ok, out, err = run_command(["docker", "buildx", "du"])
    if ok:
        print("BuildKit cache usage:")
        print(out)

        # Parse for size
        for line in out.split('\n'):
            if 'Total:' in line or 'RECLAIMABLE' in line:
                print(f"  {line}")

        # Check if cleanup recommended
        if 'GB' in out:
            cache_info["cache_exists"] = True
            # Simple heuristic: if cache output is long, probably needs cleanup
            if len(out.split('\n')) > 10:
                cache_info["needs_cleanup"] = True
                print(f"{Colors.YELLOW}⚠️  Large BuildKit cache detected - cleanup recommended{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  Could not analyze BuildKit cache: {err}{Colors.END}")

    return cache_info


def check_dockerignore() -> Dict[str, any]:
    """Verify .dockerignore is properly configured"""
    print_section("Step 4: .dockerignore Validation", "info")

    ignore_info = {
        "exists": False,
        "has_essential_patterns": False,
        "missing_patterns": []
    }

    dockerignore_path = Path(".dockerignore")
    devcontainer_dockerignore = Path(".devcontainer/.dockerignore")

    # Check both locations
    if dockerignore_path.exists():
        ignore_info["exists"] = True
        print(f"{Colors.GREEN}✅ .dockerignore found at project root{Colors.END}")
        content = dockerignore_path.read_text()
    elif devcontainer_dockerignore.exists():
        ignore_info["exists"] = True
        print(f"{Colors.YELLOW}⚠️  .dockerignore found in .devcontainer/ (should be at root){Colors.END}")
        content = devcontainer_dockerignore.read_text()
    else:
        print(f"{Colors.RED}❌ No .dockerignore found{Colors.END}")
        content = ""

    # Check for essential patterns
    essential_patterns = [
        ".git",
        "__pycache__",
        "*.pyc",
        ".venv",
        "node_modules",
        "videos",
        "data",
        "models",
        "weights"
    ]

    if content:
        found_patterns = []
        for pattern in essential_patterns:
            if pattern in content:
                found_patterns.append(pattern)
            else:
                ignore_info["missing_patterns"].append(pattern)

        ignore_info["has_essential_patterns"] = len(found_patterns) >= len(essential_patterns) * 0.6

        print(f"\nFound patterns: {len(found_patterns)}/{len(essential_patterns)}")
        if ignore_info["missing_patterns"]:
            print(f"{Colors.YELLOW}Missing recommended patterns:{Colors.END}")
            for pattern in ignore_info["missing_patterns"]:
                print(f"  - {pattern}")

    return ignore_info


def check_build_context_size() -> Dict[str, any]:
    """Estimate build context size"""
    print_section("Step 5: Build Context Size Analysis", "info")

    context_info = {
        "estimated_size_mb": 0,
        "large_files": [],
        "context_too_large": False
    }

    print("Analyzing build context (this may take a moment)...")

    # Get size of current directory
    ok, out, err = run_command(["du", "-sm", "."], timeout=120)
    if ok:
        try:
            size_mb = int(out.split()[0])
            context_info["estimated_size_mb"] = size_mb
            print(f"Estimated context size: {size_mb} MB")

            if size_mb > 100:
                context_info["context_too_large"] = True
                print(f"{Colors.YELLOW}⚠️  Large build context detected{Colors.END}")
                print(f"{Colors.YELLOW}   This will slow down 'transferring context' phase{Colors.END}")
            else:
                print(f"{Colors.GREEN}✅ Build context size is reasonable{Colors.END}")
        except:
            print(f"{Colors.YELLOW}⚠️  Could not determine context size{Colors.END}")

    # Find large files/directories
    print("\nLooking for large files/directories...")
    ok, out, err = run_command(["du", "-sm", "*"], timeout=120)
    if ok:
        large_items = []
        for line in out.strip().split('\n'):
            parts = line.split('\t', 1)
            if len(parts) == 2:
                try:
                    size = int(parts[0])
                    name = parts[1]
                    if size > 10:  # Larger than 10MB
                        large_items.append((size, name))
                except:
                    continue

        if large_items:
            large_items.sort(reverse=True)
            context_info["large_files"] = large_items[:10]
            print("Largest items in build context:")
            for size, name in large_items[:10]:
                print(f"  {size:>6} MB  {name}")

    return context_info


def create_enhanced_dockerignore() -> bool:
    """Create an enhanced .dockerignore file"""
    print_section("Creating Enhanced .dockerignore", "info")

    dockerignore_content = """# Git files
.git
.gitignore
.gitmodules
.gitattributes

# IDE and editor files
.vscode
.idea
*.swp
*.swo
*~

# Python cache and build files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv
venv/
ENV/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs and databases
*.log
*.sql
*.sqlite
*.db

# Large data directories (CRITICAL FOR BUILD SPEED)
data/
datasets/
models/
weights/
videos/
outputs/
results/
experiments/

# Cache directories
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ML/CV specific
mlruns/
wandb/
.roboflow/
yolo*/

# Node/JavaScript
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Documentation
docs/
*.md
!README.md
LICENSE*

# OS files
.DS_Store
Thumbs.db
desktop.ini

# Temporary files
*.tmp
*.temp
*.bak
*.swp
.*.swp

# Large binaries
*.mp4
*.avi
*.mov
*.mkv
*.zip
*.tar
*.tar.gz
*.tgz
*.rar
"""

    dockerignore_path = Path(".dockerignore")

    # Backup existing if present
    if dockerignore_path.exists():
        backup_path = Path(".dockerignore.backup")
        shutil.copy(dockerignore_path, backup_path)
        print(f"Backed up existing .dockerignore to {backup_path}")

    dockerignore_path.write_text(dockerignore_content)
    print(f"{Colors.GREEN}✅ Created enhanced .dockerignore{Colors.END}")
    print("This should significantly reduce build context size")

    return True


def get_fix_recommendations(
    env_info: Dict,
    space_info: Dict,
    cache_info: Dict,
    ignore_info: Dict,
    context_info: Dict
) -> List[Dict]:
    """Generate prioritized fix recommendations"""
    print_section("Step 6: Fix Recommendations", "warning")

    recommendations = []

    # Priority 1: Windows mount issue
    if env_info["on_windows_mount"]:
        recommendations.append({
            "priority": "🔴 CRITICAL",
            "issue": "Building from Windows mount (/mnt/c)",
            "impact": "10-50x slower build, primary cause of tar corruption",
            "fix": "Move project to WSL filesystem",
            "commands": [
                "# In WSL:",
                "mkdir -p ~/projects",
                f"cp -r {env_info['current_path']} ~/projects/",
                "cd ~/projects/bball_homography_pipeline",
                "# Then open in VS Code with: code ."
            ]
        })

    # Priority 2: Disk space
    if not space_info["sufficient_space"]:
        recommendations.append({
            "priority": "🔴 CRITICAL",
            "issue": f"Low disk space ({space_info['root_available_gb']:.1f} GB available)",
            "impact": "Can cause tar corruption during layer export",
            "fix": "Free up disk space or expand WSL VHDX",
            "commands": [
                "# Clean Docker data:",
                "docker system prune -af --volumes",
                "",
                "# Or expand WSL VHDX (in PowerShell as Admin):",
                "wsl --shutdown",
                "# Then use Disk Management to expand VHDX"
            ]
        })

    # Priority 3: BuildKit cache
    if cache_info["needs_cleanup"]:
        recommendations.append({
            "priority": "🟡 HIGH",
            "issue": "Large or potentially corrupted BuildKit cache",
            "impact": "Can cause tar header corruption during export",
            "fix": "Clean BuildKit cache",
            "commands": [
                "docker buildx prune -af",
                "docker system prune -af",
                "wsl --shutdown  # in PowerShell",
                "# Then restart Docker Desktop"
            ]
        })

    # Priority 4: .dockerignore
    if not ignore_info["has_essential_patterns"] or context_info["context_too_large"]:
        recommendations.append({
            "priority": "🟡 HIGH",
            "issue": f"Large build context ({context_info['estimated_size_mb']} MB)",
            "impact": "Slow 'transferring context' phase, increases corruption risk",
            "fix": "Create/enhance .dockerignore",
            "commands": [
                "# Run this script with --fix-dockerignore",
                "# Or manually add to .dockerignore:",
                "data/",
                "videos/",
                "models/",
                "weights/"
            ]
        })

    # Priority 5: BuildKit builder
    if env_info["buildx_available"]:
        recommendations.append({
            "priority": "🟢 MEDIUM",
            "issue": "Potentially stale BuildKit builder",
            "impact": "May have cached corrupted state",
            "fix": "Recreate BuildKit builder",
            "commands": [
                "docker buildx create --use --name bball-builder --driver docker-container",
                "docker buildx inspect --bootstrap"
            ]
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{rec['priority']} Recommendation {i}:")
        print(f"  Issue: {rec['issue']}")
        print(f"  Impact: {rec['impact']}")
        print(f"  Fix: {rec['fix']}")
        print(f"  Commands:")
        for cmd in rec['commands']:
            if cmd:
                print(f"    {cmd}")

    return recommendations


def apply_automatic_fixes(args) -> bool:
    """Apply fixes that can be automated"""
    print_section("Step 7: Applying Automatic Fixes", "success")

    fixes_applied = []

    # Fix 1: Create .dockerignore
    if args.fix_dockerignore or args.fix_all:
        if create_enhanced_dockerignore():
            fixes_applied.append(".dockerignore created/updated")

    # Fix 2: Clean BuildKit cache
    if args.clean_cache or args.fix_all:
        print("\nCleaning BuildKit cache...")
        ok, out, err = run_command(["docker", "buildx", "prune", "-af"], timeout=120)
        if ok:
            print(f"{Colors.GREEN}✅ BuildKit cache cleaned{Colors.END}")
            fixes_applied.append("BuildKit cache cleaned")
        else:
            print(f"{Colors.RED}❌ Failed to clean BuildKit cache: {err}{Colors.END}")

    # Fix 3: Clean Docker system
    if args.clean_docker or args.fix_all:
        print("\nCleaning Docker system...")
        ok, out, err = run_command(["docker", "system", "prune", "-af"], timeout=180)
        if ok:
            print(f"{Colors.GREEN}✅ Docker system cleaned{Colors.END}")
            fixes_applied.append("Docker system cleaned")
        else:
            print(f"{Colors.RED}❌ Failed to clean Docker system: {err}{Colors.END}")

    # Fix 4: Recreate BuildKit builder
    if args.recreate_builder or args.fix_all:
        print("\nRecreating BuildKit builder...")
        # Remove existing
        run_command(["docker", "buildx", "rm", "bball-builder"])
        # Create new
        ok, out, err = run_command([
            "docker", "buildx", "create", "--use",
            "--name", "bball-builder",
            "--driver", "docker-container"
        ])
        if ok:
            ok2, _, _ = run_command(["docker", "buildx", "inspect", "--bootstrap"])
            if ok2:
                print(f"{Colors.GREEN}✅ BuildKit builder recreated{Colors.END}")
                fixes_applied.append("BuildKit builder recreated")
        else:
            print(f"{Colors.YELLOW}⚠️  Could not recreate builder: {err}{Colors.END}")

    if fixes_applied:
        print(f"\n{Colors.GREEN}Fixes applied:{Colors.END}")
        for fix in fixes_applied:
            print(f"  ✅ {fix}")
        return True
    else:
        print(f"\n{Colors.YELLOW}No automatic fixes applied{Colors.END}")
        return False


def generate_build_script() -> None:
    """Generate an optimized build script"""
    print_section("Generating Optimized Build Script", "info")

    build_script = """#!/bin/bash
# Optimized Docker build script
# Generated by diagnostic script

set -e

echo "🔧 Starting optimized Docker build..."

# Set BuildKit environment
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build with optimized settings
echo "Building with BuildKit optimizations..."
docker buildx build \\
  --progress=plain \\
  --no-cache \\
  --output=type=docker \\
  --sbom=false \\
  --provenance=false \\
  --attest=type=none \\
  -t bball_homography_pipeline_env_datascience:dev \\
  -f .devcontainer/Dockerfile .

echo "✅ Build complete!"

# Optional: Test the image
echo "Testing image..."
docker run --rm --gpus all \\
  bball_homography_pipeline_env_datascience:dev \\
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo "✅ Image test complete!"
"""

    script_path = Path("build_optimized.sh")
    script_path.write_text(build_script)
    try:
        script_path.chmod(0o755)
    except:
        print(f"{Colors.YELLOW}⚠️  Could not make script executable (Windows filesystem){Colors.END}")
        print(f"   Run 'chmod +x {script_path}' in WSL")

    print(f"{Colors.GREEN}✅ Created optimized build script: {script_path}{Colors.END}")
    print(f"\nUsage:")
    print(f"  chmod +x {script_path}")
    print(f"  ./{script_path}")


def main():
    """Main diagnostic routine"""
    parser = argparse.ArgumentParser(
        description="Docker Build Diagnostic & Fix Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_docker_build.py                    # Run diagnostics only
  python fix_docker_build.py --fix-all          # Apply all automatic fixes
  python fix_docker_build.py --clean-cache      # Clean BuildKit cache
  python fix_docker_build.py --generate-script  # Generate optimized build script
        """
    )
    parser.add_argument('--fix-dockerignore', action='store_true',
                       help='Create/update .dockerignore')
    parser.add_argument('--clean-cache', action='store_true',
                       help='Clean BuildKit cache')
    parser.add_argument('--clean-docker', action='store_true',
                       help='Clean Docker system')
    parser.add_argument('--recreate-builder', action='store_true',
                       help='Recreate BuildKit builder')
    parser.add_argument('--fix-all', action='store_true',
                       help='Apply all automatic fixes')
    parser.add_argument('--generate-script', action='store_true',
                       help='Generate optimized build script')

    args = parser.parse_args()

    print(f"{Colors.BOLD}Docker Build Diagnostic & Fix Tool{Colors.END}")
    print(f"Working directory: {Path.cwd()}\n")

    # Run diagnostics
    env_info = check_wsl_environment()
    space_info = check_disk_space()
    cache_info = analyze_buildkit_cache()
    ignore_info = check_dockerignore()
    context_info = check_build_context_size()

    # Generate recommendations
    recommendations = get_fix_recommendations(
        env_info, space_info, cache_info, ignore_info, context_info
    )

    # Apply fixes if requested
    if any([args.fix_dockerignore, args.clean_cache, args.clean_docker,
            args.recreate_builder, args.fix_all]):
        apply_automatic_fixes(args)

    # Generate build script if requested
    if args.generate_script or args.fix_all:
        generate_build_script()

    # Final summary
    print_section("Diagnostic Summary", "info")
    print(f"WSL Environment: {'⚠️ On Windows mount' if env_info['on_windows_mount'] else '✅ On WSL filesystem'}")
    print(f"Disk Space: {'❌ Low' if not space_info['sufficient_space'] else '✅ Sufficient'} ({space_info['root_available_gb']:.1f} GB)")
    print(f"Build Context: {'⚠️ Large' if context_info['context_too_large'] else '✅ Reasonable'} ({context_info['estimated_size_mb']} MB)")
    print(f"Docker: {'✅ Available' if env_info['docker_available'] else '❌ Not found'}")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print(f"1. Review recommendations above")
    print(f"2. Apply fixes: python {sys.argv[0]} --fix-all")
    print(f"3. If on /mnt/c, move project to WSL filesystem")
    print(f"4. Run optimized build: ./build_optimized.sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
