import platform

import torch


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"gpu count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        cc = f"{props.major}.{props.minor}"
        mem_gb = props.total_memory / 1024**3
        print(f"gpu {index}: {props.name}, cc={cc}, memory={mem_gb:.1f} GiB")


if __name__ == "__main__":
    main()
