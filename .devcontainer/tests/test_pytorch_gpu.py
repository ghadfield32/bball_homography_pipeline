#!/usr/bin/env python3
"""Small PyTorch GPU benchmark."""
import time


def test_pytorch(force_cpu: bool = False) -> None:
    import torch
    cuda_ok = torch.cuda.is_available() and not force_cpu
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        print(f"device: {name} (sm_{major}{minor:02d})")
        device = torch.device("cuda:0")
    else:
        print("falling back to cpu")
        device = torch.device("cpu")

    size = (1000, 1000)
    a, b = (torch.randn(size, device=device) for _ in range(2))
    _ = a @ b
    t0 = time.time()
    _ = (a @ b).sum().item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"matmul on {device} took {(time.time()-t0)*1000:.2f} ms")


if __name__ == "__main__":
    test_pytorch()
