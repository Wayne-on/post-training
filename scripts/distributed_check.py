import os
import time

import torch
import torch.distributed as dist


def parse_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported DIST_CHECK_DTYPE: {name}") from exc


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    numel = int(os.environ.get("DIST_CHECK_NUMEL", "1"))
    iterations = int(os.environ.get("DIST_CHECK_ITERS", "1"))
    dtype = parse_dtype(os.environ.get("DIST_CHECK_DTYPE", "float32"))
    value = torch.full((numel,), rank + 1, dtype=dtype, device=device)

    expected = world_size * (world_size + 1) / 2
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        value.fill_(rank + 1)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if rank == 0:
        tensor_mib = (value.numel() * value.element_size()) / (1024**2)
        print(
            f"backend={backend}, world_size={world_size}, dtype={dtype}, "
            f"numel={numel}, tensor_mib={tensor_mib:.2f}, iterations={iterations}, "
            f"elapsed_seconds={elapsed:.4f}, all_reduce={value[0].item()}, expected={expected}"
        )
    assert value[0].item() == expected
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
