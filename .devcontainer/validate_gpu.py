#!/usr/bin/env python3
"""
Docker Build & GPU Validation Script (container-friendly)
- Adds --quick mode to skip Docker CLI checks
- Skips Docker checks automatically if docker is unavailable
- Keeps strict on GPU/Torch/JAX checks
"""
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple


def print_section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def run_command(cmd: List[str], timeout: int = 60) -> Tuple[bool, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout, r.stderr
    except Exception as e:
        return False, "", str(e)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def ensure_project_structure() -> bool:
    print_section("ENSURING PROJECT STRUCTURE")
    cwd = Path.cwd()
    print(f"Current directory: {cwd}")

    if (cwd / ".devcontainer").exists():
        project_root = cwd
    elif cwd.name == ".devcontainer":
        project_root = cwd.parent
    else:
        project_root = cwd
        (project_root / ".devcontainer").mkdir(exist_ok=True)

    dev_dir = project_root / ".devcontainer"
    print(f"Project root: {project_root}")
    print(f"DevContainer directory: {dev_dir}")

    (dev_dir / "tests").mkdir(exist_ok=True)

    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        print("Creating minimal pyproject.toml...")
        pyproject.write_text(
            """[project]
name = "bball_homography_pipeline_env"
version = "0.1.0"
description = "Docker development environment"
requires-python = ">=3.10,<3.13"

dependencies = [
    "pandas>=2.0",
    "numpy>=1.20,<2",
    "matplotlib>=3.4.0",
    "scipy>=1.7.0",
    "jupyterlab>=3.0.0",
]

[tool.uv]
index-strategy = "unsafe-best-match"
"""
        )
        print("✅ Created pyproject.toml")

    return True


def create_env_file() -> bool:
    print_section("CREATING ENVIRONMENT FILE")
    t = Path(".devcontainer/.env.template")
    f = Path(".devcontainer/.env")
    if t.exists() and not f.exists():
        f.write_bytes(t.read_bytes())
        print("✅ Created .env from template")
        return True
    elif f.exists():
        print("✅ .env file already exists")
        return True
    else:
        f.write_text(
            """ENV_NAME=bball_homography_pipeline_env
CUDA_TAG=12.4.0
PYTHON_VER=3.10
HOST_JUPYTER_PORT=8891
HOST_TENSORBOARD_PORT=6008
HOST_EXPLAINER_PORT=8050
HOST_STREAMLIT_PORT=8501
HOST_MLFLOW_PORT=5000
"""
        )
        print("✅ Created minimal .env file")
        return True


def fix_file_permissions() -> bool:
    print_section("FIXING FILE PERMISSIONS")
    try:
        is_wsl = "microsoft" in os.uname().release.lower()
    except Exception:
        is_wsl = False

    if os.name == "nt" or is_wsl:
        print("Detected Windows/WSL environment")
        for p in [
            ".devcontainer/validate_gpu.py",
            ".devcontainer/tests/test_summary.py",
            ".devcontainer/tests/test_pytorch.py",
            ".devcontainer/tests/test_pytorch_gpu.py",
            ".devcontainer/tests/test_uv.py",
        ]:
            fp = Path(p)
            if fp.exists():
                try:
                    os.chmod(fp, 0o755)
                    print(f"✅ Fixed permissions for {p}")
                except Exception as e:
                    print(f"⚠️ Could not fix permissions for {p}: {e}")
    return True


def validate_docker_environment() -> bool:
    print_section("VALIDATING DOCKER ENVIRONMENT")
    if not docker_available():
        print("ℹ️ Docker CLI not found in this environment; skipping Docker checks.")
        return True  # treat as success inside containers
    ok, out, err = run_command(["docker", "info"])
    if not ok:
        print(f"❌ Docker daemon not accessible: {err}")
        return False
    print("✅ Docker daemon is running")

    ok, out, err = run_command(["docker", "compose", "version"])
    if not ok:
        print(f"❌ Docker Compose not available: {err}")
        return False
    print(f"✅ Docker Compose: {out.strip()}")
    return True


def stop_and_remove_containers() -> bool:
    print_section("CLEANING EXISTING CONTAINERS")
    if not docker_available():
        print("ℹ️ Docker CLI not found; skipping container cleanup.")
        return True
    ok, _, err = run_command(
        ["docker", "compose", "-f", ".devcontainer/docker-compose.yml", "down", "--volumes"]
    )
    if not ok:
        print(f"⚠️ Could not stop containers (may not exist): {err}")
    for name in ["bball_homography_pipeline_env_datascience", "bball_homography_pipeline_env_mlflow"]:
        run_command(["docker", "rm", "-f", name])
    print("✅ Container cleanup complete")
    return True


def clean_docker_cache() -> bool:
    print_section("CLEANING DOCKER CACHE")
    if not docker_available():
        print("ℹ️ Docker CLI not found; skipping cache prune.")
        return True
    ok, out, err = run_command(["docker", "builder", "prune", "--all", "--force"])
    if ok:
        print("✅ Docker build cache cleaned")
        if out:
            print(out)
        return True
    print(f"❌ Failed to clean Docker cache: {err}")
    return False


def test_build() -> bool:
    print_section("TESTING DOCKER BUILD")
    if not docker_available():
        print("ℹ️ Docker CLI not found; skipping compose build test.")
        return True
    if Path.cwd().name == ".devcontainer":
        os.chdir("..")
    compose_file = ".devcontainer/docker-compose.yml"
    print(f"Using compose file: {Path(compose_file).absolute()}")
    print(f"Build context: {Path('.').absolute()}")
    ok, out, err = run_command(
        ["docker", "compose", "-f", compose_file, "build", "--no-cache"], timeout=600
    )
    if ok:
        print("✅ Docker build successful!")
        print("\n".join(out.splitlines()[-10:]))
        return True
    print("❌ Docker build failed")
    print("STDERR:\n", err)
    print("STDOUT (last 20 lines):\n", "\n".join(out.splitlines()[-20:]))
    return False


def section_summary(struct_ok, uv_ok, pt_ok, jax_ok):
    print_section("SUMMARY")
    print(f"structure: {struct_ok} uv: {uv_ok} pytorch: {pt_ok} jax: {jax_ok}")


def test_uv() -> bool:
    print_section("UV")
    ok, out, err = run_command(["uv", "--version"])
    print((out or err).strip() or "uv not in PATH")
    return ok


def test_pytorch() -> bool:
    print_section("PYTORCH")
    try:
        import torch
        print("version:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            d = torch.device("cuda:0")
            import time
            x = torch.randn((512, 512), device=d)
            t0 = time.time()
            y = (x @ x.T).sum()
            torch.cuda.synchronize()
            print("sum:", float(y))
            print(f"gpu op ms: {(time.time() - t0)*1000:.2f}")
            return True
        return False
    except Exception as e:
        print("error:", e)
        return False


def test_jax() -> bool:
    print_section("JAX")
    try:
        import jax
        import jax.numpy as jnp

        devs = jax.devices()
        print("devices:", devs)
        gpus = jax.devices("gpu") or [
            d for d in devs
            if getattr(d, "platform", "").lower() in {"gpu", "cuda"} or "cuda" in str(d).lower()
        ]
        if not gpus:
            print("no gpu devices detected by jax")
            return False
        
        # Enhanced JAX GPU test with CuDNN compatibility check
        print("Testing JAX GPU computation with CuDNN compatibility...")
        try:
            x = jnp.ones((512, 512), dtype=jnp.float32)
            x = jax.device_put(x, gpus[0])
            s = jnp.sum(x).block_until_ready()
            print("sum:", float(s))
            
            # Test more complex operations that might trigger CuDNN
            y = jnp.dot(x, x.T)
            result = jnp.sum(y).block_until_ready()
            print("matrix multiplication result:", float(result))
            return True
        except Exception as cudnn_error:
            print(f"JAX GPU computation failed (likely CuDNN issue): {cudnn_error}")
            print("This is likely due to CuDNN version mismatch between PyTorch and JAX")
            return False
            
    except Exception as e:
        print("error:", e)
        return False


def fix_cudnn_compatibility() -> bool:
    """Attempt to fix CuDNN compatibility issues."""
    print_section("CUDNN COMPATIBILITY FIX")
    
    try:
        import subprocess
        
        # Check current CuDNN versions
        print("Checking current CuDNN versions...")
        
        # Try to upgrade CuDNN to compatible version
        print("Attempting to upgrade CuDNN to compatible version...")
        result = subprocess.run([
            'uv', 'pip', 'install', '--upgrade', 
            'nvidia-cudnn-cu12==9.8.0.69'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ CuDNN upgraded successfully")
            return True
        else:
            print(f"⚠️ CuDNN upgrade failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CuDNN fix failed: {e}")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Skip Docker checks; run only structure/UV/Torch/JAX")
    p.add_argument("--fix-cudnn", action="store_true",
                   help="Attempt to fix CuDNN compatibility issues")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("Docker DevContainer Build & GPU Validation")
    print(f"Working directory: {os.getcwd()}")

    # Always run these
    struct_ok = ensure_project_structure()
    env_ok = create_env_file()
    perm_ok = fix_file_permissions()

    # Optional Docker checks
    docker_ok = True
    build_ok = True
    cache_ok = True
    stop_ok = True

    if not args.quick:
        docker_ok = validate_docker_environment()
        stop_ok = stop_and_remove_containers()
        cache_ok = clean_docker_cache()
        build_ok = test_build()

    uv_ok = test_uv()
    pt_ok = test_pytorch()
    jax_ok = test_jax()
    
    # Attempt CuDNN fix if requested and JAX failed
    if args.fix_cudnn and not jax_ok:
        print("\nJAX GPU test failed, attempting CuDNN compatibility fix...")
        cudnn_fix_ok = fix_cudnn_compatibility()
        if cudnn_fix_ok:
            print("Retesting JAX after CuDNN fix...")
            jax_ok = test_jax()

    section_summary(struct_ok, uv_ok, pt_ok, jax_ok)

    # In quick mode, ignore Docker results entirely.
    if args.quick:
        return 0 if all([struct_ok, uv_ok, pt_ok, jax_ok]) else 1

    # Otherwise include Docker outcomes.
    ok = all([
        struct_ok, env_ok, perm_ok,
        docker_ok, stop_ok, cache_ok, build_ok,
        uv_ok, pt_ok, jax_ok
    ])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
