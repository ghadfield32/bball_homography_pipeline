#!/usr/bin/env python3
"""Aggregated checks for the devcontainer layout and GPU readiness."""
import os
import sys
import time
import subprocess


def section(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


def test_structure() -> bool:
    section("STRUCTURE")
    expected = [
        '/workspace/docker-compose.yml',
        '/workspace/pyproject.toml',
        '/workspace/.devcontainer/devcontainer.json',
        '/workspace/.devcontainer/Dockerfile',
        '/workspace/.devcontainer/.env.template',
        '/workspace/.devcontainer/.dockerignore',
        '/app/validate_gpu.py',
        '/app/tests/'
    ]
    ok = True
    for p in expected:
        if os.path.exists(p):
            print("ok:", p)
        else:
            print("missing:", p)
            ok = False
    return ok


def test_uv() -> bool:
    section("UV")
    try:
        r = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        return r.returncode == 0
    except FileNotFoundError:
        print('uv not in PATH')
        return False


def test_pytorch() -> bool:
    section("PYTORCH")
    try:
        import torch
        print("version:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            d = torch.device('cuda:0')
            x = torch.ones(512, 512, device=d)
            y = torch.sum(x)
            print("sum:", y.item())
            return True
        return False
    except Exception as e:
        print("error:", e)
        return False


def test_jax() -> bool:
    section("JAX")
    try:
        import jax, jax.numpy as jnp

        # Show all devices for visibility
        devs = jax.devices()
        print("devices:", devs)

        # Prefer the supported filtered query
        gpus = jax.devices("gpu")

        # Fallback for older/newer renderings (e.g., "CudaDevice(id=0)")
        if not gpus:
            gpus = [
                d for d in devs
                if getattr(d, "platform", "").lower() in {"gpu", "cuda"} or "cuda" in str(d).lower()
            ]

        if not gpus:
            print("no gpu devices detected by jax")
            return False

        # Tiny compute on the first GPU to ensure execution
        x = jnp.ones((512, 512), dtype=jnp.float32)
        x = jax.device_put(x, gpus[0])
        s = jnp.sum(x).block_until_ready()
        print("sum:", float(s))
        return True
    except Exception as e:
        print("error:", e)
        return False



def main() -> int:
    s_ok = test_structure()
    uv_ok = test_uv()
    pt_ok = test_pytorch()
    j_ok = test_jax()

    section("SUMMARY")
    print("structure:", s_ok, "uv:", uv_ok, "pytorch:", pt_ok, "jax:", j_ok)
    return 0 if all([s_ok, uv_ok, pt_ok, j_ok]) else 1


if __name__ == '__main__':
    sys.exit(main())
