import os

import torch
import torch.distributed as dist


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

    value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size + 1) / 2
    if rank == 0:
        print(f"backend={backend}, world_size={world_size}, all_reduce={value.item()}, expected={expected}")
    assert value.item() == expected
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
