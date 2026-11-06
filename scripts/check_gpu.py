"""Report the torch / CUDA / GPU stack and run a bf16 matmul sanity check.

Exit code 0 when CUDA is usable and the bf16 result matches an fp32 reference,
1 otherwise. Usage: ``uv run python scripts/check_gpu.py [--size 4096]``.
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="square matmul size")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    args = parser.parse_args(argv)

    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch CUDA build : {torch.version.cuda}")
    print(f"cuDNN            : {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("CUDA available   : False  -> no usable GPU")
        return 1

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    cap = torch.cuda.get_device_capability(dev)
    arch = f"sm_{cap[0]}{cap[1]}"
    print("CUDA available   : True")
    print(f"GPU              : {props.name}")
    print(f"Compute cap.     : {cap[0]}.{cap[1]} ({arch})")
    print(f"VRAM total       : {props.total_memory / 2**30:.2f} GiB")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"VRAM free        : {free / 2**30:.2f} GiB")
    archs = torch.cuda.get_arch_list()
    print(f"Build arch list  : {' '.join(archs)}")
    if arch not in archs:
        print(f"WARNING: {arch} not in this build's arch list; kernels may JIT or fail.")
    print(f"bf16 supported   : {torch.cuda.is_bf16_supported()}")

    n = args.size
    gen = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(n, n, device=dev, generator=gen)
    b = torch.randn(n, n, device=dev, generator=gen)
    ref = a @ b  # fp32 (TF32 off by default for matmul in torch>=1.12)
    a16, b16 = a.bfloat16(), b.bfloat16()
    out = a16 @ b16
    torch.cuda.synchronize()

    rel_err = ((out.float() - ref).norm() / ref.norm()).item()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        a16 @ b16
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    tflops = 2 * n**3 / dt / 1e12
    ok = bool(torch.isfinite(out).all().item()) and rel_err < 1e-2
    print(f"bf16 matmul {n}x{n}: rel. error vs fp32 = {rel_err:.2e}, {tflops:.1f} TFLOP/s")
    print(f"peak VRAM (test) : {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB")
    print("RESULT           :", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
