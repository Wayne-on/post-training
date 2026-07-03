from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def adapter_weight_file(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}")


def convert_key(key: str, from_prefix: str, to_prefix: str) -> tuple[str, bool]:
    if from_prefix in key:
        return key.replace(from_prefix, to_prefix, 1), True
    return key, False


def copy_adapter_files(source: Path, output: Path, weight_name: str, force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to replace copied metadata files.")
    else:
        output.mkdir(parents=True)

    for item in source.iterdir():
        destination = output / item.name
        if item.name == weight_name:
            continue
        if item.is_dir():
            if destination.exists() and force:
                shutil.rmtree(destination)
            if not destination.exists():
                shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def convert_safetensors(source_weight: Path, output_weight: Path, from_prefix: str, to_prefix: str) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    converted: dict[str, Any] = {}
    changed = 0
    with safe_open(source_weight, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            new_key, did_change = convert_key(key, from_prefix, to_prefix)
            if new_key in converted:
                raise KeyError(f"Key collision after conversion: {key} -> {new_key}")
            converted[new_key] = handle.get_tensor(key)
            changed += int(did_change)

    save_file(converted, output_weight, metadata=metadata)
    return {"total_keys": len(converted), "changed_keys": changed}


def convert_bin(source_weight: Path, output_weight: Path, from_prefix: str, to_prefix: str) -> dict[str, Any]:
    import torch

    state = torch.load(source_weight, map_location="cpu")
    wrapper_key = None
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        wrapper_key = "state_dict"
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise TypeError(f"Unsupported torch checkpoint object: {type(state)!r}")

    converted: dict[str, Any] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key, did_change = convert_key(str(key), from_prefix, to_prefix)
        if new_key in converted:
            raise KeyError(f"Key collision after conversion: {key} -> {new_key}")
        converted[new_key] = value
        changed += int(did_change)

    if wrapper_key:
        state[wrapper_key] = converted
        torch.save(state, output_weight)
    else:
        torch.save(converted, output_weight)
    return {"total_keys": len(converted), "changed_keys": changed}


def write_conversion_report(
    output: Path,
    source: Path,
    source_weight: Path,
    stats: dict[str, Any],
    from_prefix: str,
    to_prefix: str,
) -> None:
    report = {
        "source_adapter": str(source),
        "source_weight_file": str(source_weight),
        "from_prefix": from_prefix,
        "to_prefix": to_prefix,
        **stats,
    }
    (output / "adapter_key_prefix_conversion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a PEFT adapter and rewrite a key prefix in its weight file.")
    parser.add_argument("--source", required=True, help="Source PEFT adapter directory.")
    parser.add_argument("--output", required=True, help="Output PEFT adapter directory.")
    parser.add_argument(
        "--from-prefix",
        default="base_model.model.model.language_model.",
        help="Existing key prefix segment to replace.",
    )
    parser.add_argument(
        "--to-prefix",
        default="base_model.model.model.",
        help="Replacement key prefix segment.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite copied metadata files in output directory.")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Source adapter does not exist: {source}")
    if not (source / "adapter_config.json").exists():
        raise FileNotFoundError(f"Source is not a PEFT adapter directory: missing {source / 'adapter_config.json'}")

    source_weight = adapter_weight_file(source)
    output_weight = output / source_weight.name
    copy_adapter_files(source, output, source_weight.name, args.force)

    if source_weight.suffix == ".safetensors":
        stats = convert_safetensors(source_weight, output_weight, args.from_prefix, args.to_prefix)
    else:
        stats = convert_bin(source_weight, output_weight, args.from_prefix, args.to_prefix)

    if stats["changed_keys"] == 0:
        raise RuntimeError(
            f"No keys matched from-prefix {args.from_prefix!r}. "
            "The adapter may already be converted or the prefix is wrong."
        )

    write_conversion_report(output, source, source_weight, stats, args.from_prefix, args.to_prefix)
    print("[adapter-convert] written:", output)
    print("[adapter-convert] total keys:", stats["total_keys"])
    print("[adapter-convert] changed keys:", stats["changed_keys"])
    print("[adapter-convert] report:", output / "adapter_key_prefix_conversion.json")


if __name__ == "__main__":
    main()
