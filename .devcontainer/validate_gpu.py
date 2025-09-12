#!/usr/bin/env python3
"""
GPU validation and environment diagnostics for RTX 4090 devcontainer.
Focus: verify JAX and PyTorch access to CUDA, report common misconfigurations.
"""
import sys
import os
import subprocess
import warnings
import textwrap
import re
warnings.filterwarnings('ignore')


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def validate_environment_variables() -> bool:
    """Validate JAX‑related environment variables (no inline comments, valid types)."""
    print_section("JAX ENVIRONMENT VARIABLE VALIDATION")

    jax_numeric_vars = {
        'XLA_PYTHON_CLIENT_MEM_FRACTION': {'type': 'float', 'range': (0.0, 1.0)},
        'JAX_PREALLOCATION_SIZE_LIMIT_BYTES': {'type': 'int', 'range': (0, None)},
    }
    jax_string_vars = {
        'XLA_FLAGS', 'JAX_PLATFORM_NAME', 'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_PREALLOCATE'
    }

    ok = True
    problems = []

    for var, cfg in jax_numeric_vars.items():
        value = os.environ.get(var)
        print(f"\nCheck {var} -> {value}")
        if value is None:
            print("  not set; defaults apply")
            continue
        if '#' in value:
            clean = value.split('#')[0].strip()
            print("  contains inline comment; use:", clean)
            problems.append((var, value, clean))
            ok = False
            continue
        try:
            if cfg['type'] == 'float':
                v = float(value)
                low, high = cfg['range']
                if (low is not None and v < low) or (high is not None and v > high):
                    print("  out of recommended range")
                else:
                    print("  ok")
            else:
                v = int(value)
                print("  ok")
        except ValueError as e:
            print("  invalid numeric value:", e)
            ok = False

    for var in jax_string_vars:
        value = os.environ.get(var)
        if value and '#' in value:
            print(f"warn: {var} contains '#', which can break parsing")

    if problems:
        print("\nFix suggestions:")
        for var, bad, clean in problems:
            print(f"export {var}={clean}")
    return ok


def check_environment() -> None:
    print_section("ENVIRONMENT CHECK")
    print("python:", sys.executable)
    print("version:", sys.version)
    print("VIRTUAL_ENV:", os.environ.get('VIRTUAL_ENV'))
    print("PATH contains .venv:", '.venv/bin' in os.environ.get('PATH', ''))

    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES']
    print("\nCUDA variables:")
    for var in cuda_vars:
        print(f"  {var}:", os.environ.get(var, 'not set'))

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("\nGPU:", result.stdout.strip())
        else:
            print("\nwarn: nvidia-smi returned non‑zero")
    except FileNotFoundError:
        print("\nwarn: nvidia-smi not found in path")


def test_pytorch() -> bool:
    print_section("PYTORCH GPU TEST")
    try:
        import torch
        print("version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device count:", torch.cuda.device_count())
            print("device 0:", torch.cuda.get_device_name(0))
            # quick matmul
            import time
            dev = torch.device('cuda')
            x = torch.randn(2000, 2000, device=dev)
            y = torch.randn(2000, 2000, device=dev)
            _ = x @ y
            torch.cuda.synchronize()
            t0 = time.time()
            r = x @ y
            torch.cuda.synchronize()
            print("matmul elapsed s:", round(time.time() - t0, 3))
            _ = r.sum().item()
            return True
        return False
    except Exception as e:
        print("pytorch test error:", e)
        return False


def check_cudnn_compatibility() -> bool:
    """Check CuDNN version compatibility between PyTorch and JAX."""
    print_section("CUDNN COMPATIBILITY CHECK")
    try:
        import torch
        import subprocess
        import glob
        
        # Check PyTorch CuDNN version
        pytorch_cudnn = torch.backends.cudnn.version()
        print(f"PyTorch CuDNN version: {pytorch_cudnn}")
        
        # Check installed nvidia-cudnn-cu12 package version
        try:
            result = subprocess.run(['uv', 'pip', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'nvidia-cudnn-cu12' in line:
                        print(f"Installed CuDNN package: {line.strip()}")
                        break
        except Exception as e:
            print(f"Could not check CuDNN package version: {e}")
        
        # Check CuDNN library files
        cudnn_paths = [
            "/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu"
        ]
        
        print("\nCuDNN library search:")
        for path in cudnn_paths:
            if os.path.exists(path):
                cudnn_libs = glob.glob(f"{path}/libcudnn*")
                if cudnn_libs:
                    print(f"  {path}: {len(cudnn_libs)} CuDNN libraries found")
                    for lib in cudnn_libs[:3]:  # Show first 3
                        print(f"    - {os.path.basename(lib)}")
                else:
                    print(f"  {path}: No CuDNN libraries found")
            else:
                print(f"  {path}: Path does not exist")
        
        # Check LD_LIBRARY_PATH
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        print(f"\nLD_LIBRARY_PATH: {ld_path}")
        
        # Version compatibility check
        if pytorch_cudnn < 9000:  # Assuming version format like 9100 for 9.1.0
            print("WARNING: PyTorch CuDNN version may be too old for JAX")
            return False
        
        print("CuDNN compatibility check passed")
        return True
        
    except Exception as e:
        print(f"CuDNN compatibility check failed: {e}")
        return False


def test_jax_initialization() -> bool:
    print_section("JAX INITIALIZATION TEST")
    try:
        os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
        import jax
        import jaxlib
        from jaxlib import xla_client
        print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__)
        
        # Check for CuDNN version mismatch errors
        try:
            opts = xla_client.generate_pjrt_gpu_plugin_options()
            print("gpu plugin options ok; memory_fraction:", opts.get('memory_fraction', 'not set'))
        except Exception as e:
            print("gpu plugin options error:", e)
            if "could not convert string to float" in str(e):
                print("hint: check XLA_PYTHON_CLIENT_MEM_FRACTION for inline comments")
            elif "CuDNN" in str(e) and "version" in str(e):
                print("hint: CuDNN version mismatch detected - check compatibility")
            return False
        return True
    except Exception as e:
        print("jax init error:", e)
        if "CuDNN" in str(e):
            print("hint: CuDNN-related error detected")
        return False


def test_jax() -> bool:
    print_section("JAX GPU TEST")
    try:
        os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
        import jax, jax.numpy as jnp
        from jax.lib import xla_bridge
        print("backend:", xla_bridge.get_backend().platform)
        devices = jax.devices()
        print("devices:", devices)
        gpus = [d for d in devices if 'gpu' in str(d).lower() or getattr(d, 'platform', '') == 'gpu']
        if not gpus:
            print("no gpu devices detected by jax")
            return False
        # quick compute
        import time
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2000, 2000))
        x = jax.device_put(x, gpus[0])
        t0 = time.time()
        s = jnp.sum(x @ x).block_until_ready()
        print("matmul elapsed s:", round(time.time() - t0, 3), "sum:", float(s))
        return True
    except Exception as e:
        print("jax test error:", e)
        return False


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--fix', action='store_true', help='Run with fix recommendations')
    args = p.parse_args()

    if args.quick:
        env_ok = validate_environment_variables()
        pt_ok = test_pytorch()
        return 0 if (env_ok and pt_ok) else 1

    env_ok = validate_environment_variables()
    check_environment()
    cudnn_ok = check_cudnn_compatibility()
    jax_init_ok = test_jax_initialization()
    jax_ok = test_jax()
    pt_ok = test_pytorch()

    print_section("SUMMARY")
    print("env vars:", "ok" if env_ok else "fail")
    print("cudnn compatibility:", "ok" if cudnn_ok else "fail")
    print("jax init:", "ok" if jax_init_ok else "fail")
    print("jax compute:", "ok" if jax_ok else "fail")
    print("pytorch:", "ok" if pt_ok else "fail")

    # Provide fix recommendations if requested
    if args.fix and not (env_ok and cudnn_ok and jax_init_ok and jax_ok and pt_ok):
        print_section("FIX RECOMMENDATIONS")
        if not cudnn_ok:
            print("1. CuDNN version mismatch detected:")
            print("   - Upgrade nvidia-cudnn-cu12 to version >= 9.8.0")
            print("   - Ensure LD_LIBRARY_PATH includes CuDNN library paths")
        if not jax_init_ok:
            print("2. JAX initialization failed:")
            print("   - Check CuDNN compatibility")
            print("   - Verify XLA environment variables (no inline comments)")
        if not jax_ok:
            print("3. JAX GPU computation failed:")
            print("   - Verify GPU is accessible")
            print("   - Check CUDA driver compatibility")

    return 0 if (env_ok and cudnn_ok and jax_init_ok and jax_ok and pt_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
