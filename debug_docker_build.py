#!/usr/bin/env python3
"""
Docker Build Debugging Script - INVESTIGATE BEFORE FIXING
Follows methodology: Don't fill in missing values, dissect the problem, add debugs
"""
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, capture=True):
    """Run command and return output."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, timeout=30)
            return result.returncode, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def debug_docker_system():
    """Debug Docker system state."""
    section("DOCKER SYSTEM STATE")

    # Docker version
    rc, out, err = run_cmd(["docker", "version", "--format", "json"])
    if rc == 0:
        version = json.loads(out)
        print(f"Docker Version: {version.get('Client', {}).get('Version', 'unknown')}")
        print(f"BuildKit Enabled: {os.environ.get('DOCKER_BUILDKIT', 'not set')}")
    else:
        print(f"ERROR getting Docker version: {err}")

    # Docker info
    rc, out, err = run_cmd(["docker", "info", "--format", "json"])
    if rc == 0:
        info = json.loads(out)
        print(f"\nDocker Info:")
        print(f"  Driver: {info.get('Driver', 'unknown')}")
        print(f"  Storage Driver: {info.get('Driver', 'unknown')}")
        print(f"  Logging Driver: {info.get('LoggingDriver', 'unknown')}")
        print(f"  Cgroup Version: {info.get('CgroupVersion', 'unknown')}")

    # BuildKit info
    print("\nBuildKit Cache Info:")
    rc, out, err = run_cmd(["docker", "buildx", "du"])
    if rc == 0:
        print(out)
    else:
        print(f"ERROR: {err}")


def debug_disk_space():
    """Check disk space on all relevant mounts."""
    section("DISK SPACE ANALYSIS")

    rc, out, err = run_cmd(["df", "-h"])
    if rc == 0:
        print("Disk Usage:")
        print(out)

        # Parse and check for low space
        lines = out.strip().split('\n')[1:]  # Skip header
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                mount = parts[-1]
                usage = parts[4].rstrip('%')
                try:
                    usage_pct = int(usage)
                    if usage_pct > 90:
                        print(f"\n⚠️  WARNING: {mount} is {usage_pct}% full!")
                except ValueError:
                    pass


def debug_wsl_specific():
    """Debug WSL-specific issues."""
    section("WSL ENVIRONMENT CHECK")

    # Check if we're in WSL
    rc, out, err = run_cmd(["uname", "-r"])
    if rc == 0 and "microsoft" in out.lower():
        print("✓ Running in WSL")
        print(f"  Kernel: {out.strip()}")

        # Check WSL version
        if os.path.exists("/proc/version"):
            with open("/proc/version") as f:
                version = f.read()
                print(f"  Version: {version.strip()}")

        # Check for WSL2 vs WSL1
        if "wsl2" in out.lower() or "microsoft" in out.lower():
            print("  Type: WSL2")
        else:
            print("  Type: WSL1 (or unknown)")

        # Check vmmem process (WSL2 specific)
        rc, out, err = run_cmd(["tasklist.exe"], capture=True)
        if rc == 0 and "vmmem" in out.lower():
            print("  vmmem process: Running")

    else:
        print("Not running in WSL (or unable to detect)")


def debug_layer_cache():
    """Inspect Docker layer cache."""
    section("LAYER CACHE INSPECTION")

    # Get build cache
    rc, out, err = run_cmd(["docker", "buildx", "du", "--verbose"])
    if rc == 0:
        print("Build Cache Details:")
        lines = out.strip().split('\n')

        # Look for the problematic layer
        problem_layer = "273a96fcf18eca16d20aa34bdee418792e9ed7e25956c870cf6967cde465080b"

        for line in lines:
            if problem_layer[:12] in line:
                print(f"\n🔍 FOUND PROBLEMATIC LAYER:")
                print(f"  {line}")

        # Show large caches
        print("\nLarge Cache Items (>500MB):")
        for line in lines[1:]:  # Skip header
            if "MB" in line or "GB" in line:
                parts = line.split()
                if len(parts) >= 2:
                    size_str = parts[1]
                    try:
                        if "GB" in size_str:
                            size = float(size_str.replace("GB", ""))
                            if size > 0.5:
                                print(f"  {line}")
                        elif "MB" in size_str:
                            size = float(size_str.replace("MB", ""))
                            if size > 500:
                                print(f"  {line}")
                    except ValueError:
                        pass


def debug_image_layers():
    """Inspect existing images and their layers."""
    section("EXISTING IMAGE INSPECTION")

    # Check if the image exists
    image_name = "bball_homography_pipeline_env_datascience:latest"
    rc, out, err = run_cmd(["docker", "images", image_name, "--format", "json"])

    if rc == 0 and out.strip():
        print(f"Image {image_name} exists")

        # Get image history
        rc, out, err = run_cmd(["docker", "history", image_name, "--no-trunc", "--format", "json"])
        if rc == 0:
            print("\nImage Layer History:")
            layers = out.strip().split('\n')
            for i, layer_json in enumerate(layers[:10]):  # First 10 layers
                try:
                    layer = json.loads(layer_json)
                    print(f"\nLayer {i}:")
                    print(f"  Created: {layer.get('CreatedAt', 'unknown')}")
                    print(f"  Size: {layer.get('Size', 'unknown')}")
                    print(f"  Comment: {layer.get('Comment', 'none')[:80]}")
                except json.JSONDecodeError:
                    pass
    else:
        print(f"Image {image_name} does not exist yet")


def debug_memory_pressure():
    """Check for memory pressure issues."""
    section("MEMORY ANALYSIS")

    # System memory
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            meminfo = f.read()

        for line in meminfo.split('\n'):
            if any(key in line for key in ['MemTotal', 'MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree']):
                print(line)

    # Docker memory limits
    rc, out, err = run_cmd(["docker", "info", "--format", "{{.MemTotal}}"])
    if rc == 0:
        mem_bytes = int(out.strip())
        mem_gb = mem_bytes / (1024**3)
        print(f"\nDocker Memory Limit: {mem_gb:.2f} GB")


def debug_build_context():
    """Check build context size and contents."""
    section("BUILD CONTEXT ANALYSIS")

    # Check .dockerignore
    dockerignore_path = Path(".devcontainer/.dockerignore")
    if dockerignore_path.exists():
        print("✓ .dockerignore exists")
        with open(dockerignore_path) as f:
            ignore_rules = f.read().strip().split('\n')
            print(f"  Rules: {len(ignore_rules)}")
    else:
        print("⚠️  .dockerignore NOT found")

    # Estimate context size
    print("\nBuild Context Size Estimation:")
    rc, out, err = run_cmd(["du", "-sh", "."])
    if rc == 0:
        print(f"  Current directory: {out.strip()}")

    # Check for large files that shouldn't be in context
    print("\nLarge files in build context (>100MB):")
    rc, out, err = run_cmd(["find", ".", "-type", "f", "-size", "+100M", "-not", "-path", "./.git/*"])
    if rc == 0 and out.strip():
        for line in out.strip().split('\n')[:10]:  # First 10
            print(f"  {line}")
    else:
        print("  None found (or error)")


def create_debug_dockerfile():
    """Create a minimal test Dockerfile to isolate the issue."""
    section("CREATING DEBUG DOCKERFILE")

    debug_dockerfile = """
# Debug Dockerfile - Minimal test
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN echo "Testing layer 1" && \\
    apt-get update && \\
    apt-get install -y python3 && \\
    echo "Layer 1 complete"

RUN echo "Testing layer 2" && \\
    apt-get install -y python3-pip && \\
    echo "Layer 2 complete"

WORKDIR /app
CMD ["bash"]
"""

    debug_path = Path(".devcontainer/Dockerfile.debug")
    debug_path.write_text(debug_dockerfile)
    print(f"✓ Created {debug_path}")
    print("\nTo test: docker build -f .devcontainer/Dockerfile.debug -t debug-test .")


def generate_debug_report():
    """Generate a comprehensive debug report."""
    section("GENERATING DEBUG REPORT")

    report_path = Path("docker_build_debug_report.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w") as f:
        f.write(f"Docker Build Debug Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # Capture all debug output
        for func in [debug_docker_system, debug_disk_space, debug_wsl_specific,
                     debug_layer_cache, debug_image_layers, debug_memory_pressure,
                     debug_build_context]:
            func_name = func.__name__.replace('debug_', '').replace('_', ' ').title()
            f.write(f"\n{'=' * 80}\n{func_name}\n{'=' * 80}\n")

            # Redirect stdout to capture function output
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                func()
                output = sys.stdout.getvalue()
                f.write(output)
            except Exception as e:
                f.write(f"ERROR: {e}\n")
            finally:
                sys.stdout = old_stdout

    print(f"✓ Debug report saved to: {report_path.absolute()}")
    return report_path


def main():
    """Run all debug checks."""
    print("Docker Build Debugging - Systematic Investigation")
    print("Following methodology: Dissect the problem with debugs")
    print(f"Working directory: {os.getcwd()}")
    print(f"Time: {datetime.now()}")

    # Change to project root if we're in .devcontainer
    if Path.cwd().name == ".devcontainer":
        os.chdir("..")
        print(f"Changed to project root: {os.getcwd()}")

    try:
        # Run all debug checks
        debug_docker_system()
        debug_disk_space()
        debug_wsl_specific()
        debug_layer_cache()
        debug_image_layers()
        debug_memory_pressure()
        debug_build_context()
        create_debug_dockerfile()

        # Generate report
        report_path = generate_debug_report()

        section("NEXT STEPS")
        print("\n✓ Debug investigation complete!")
        print(f"\n📋 Full report saved to: {report_path}")
        print("\n🔍 Key findings to review:")
        print("  1. Check disk space - look for >90% usage")
        print("  2. Check layer cache - look for corrupted layers")
        print("  3. Check memory - look for low available memory")
        print("  4. Review build context size - should be <1GB")
        print("\n⚠️  DO NOT apply fixes yet - review findings first!")
        print("Once you've reviewed the debug report, we can apply targeted fixes.")

    except Exception as e:
        print(f"\n❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
