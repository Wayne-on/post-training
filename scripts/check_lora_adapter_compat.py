from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


LORA_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_[AB](?:\.[^.]+)?\.weight$")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path).expanduser()


def normalized_path_text(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path).expanduser()
    try:
        return str(candidate.resolve()) if candidate.exists() else str(candidate)
    except Exception:
        return str(candidate)


def compare_path_like(left: str | None, right: str | None) -> tuple[bool, str]:
    left_text = normalized_path_text(left)
    right_text = normalized_path_text(right)
    if not left_text or not right_text:
        return False, "missing path"
    if left_text == right_text:
        return True, "exact match"

    left_path = Path(left_text)
    right_path = Path(right_text)
    if left_path.exists() and right_path.exists():
        try:
            if left_path.samefile(right_path):
                return True, "same resolved file"
        except Exception:
            pass

    if left_path.name == right_path.name:
        return False, "basename matches but full path differs"
    return False, "different path"


def adapter_weight_file(adapter_dir: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    return None


def load_adapter_keys(path: Path) -> list[str]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            return list(handle.keys())

    import torch

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported adapter weight object: {type(state)!r}")
    return list(state.keys())


def lora_module_from_key(key: str) -> str | None:
    match = LORA_KEY_RE.match(key)
    if not match:
        return None
    return match.group("module")


def canonical_module_path(path: str) -> str:
    normalized = path
    if normalized.startswith("base_model.model."):
        normalized = normalized[len("base_model.model.") :]
    return normalized


def leaf_name(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def target_modules_from_model(model: Any, target_modules: set[str]) -> set[str]:
    return {
        name
        for name, _module in model.named_modules()
        if name and leaf_name(name) in target_modules
    }


def list_model_target_modules(model_name_or_path: str, target_modules: set[str]) -> tuple[set[str], str | None]:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:
        return set(), f"model inspection imports failed: {exc!r}"

    errors: list[str] = []
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return target_modules_from_model(model, target_modules), None
    except Exception as exc:
        errors.append(f"from_config failed: {exc!r}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": "meta"},
        )
        return target_modules_from_model(model, target_modules), None
    except Exception as exc:
        errors.append(f"from_pretrained meta failed: {exc!r}")

    return set(), "model architecture inspection failed: " + " | ".join(errors)


def peft_load_probe(model_name_or_path: str, adapter_name_or_path: str) -> tuple[set[str], str | None]:
    try:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
    except Exception as exc:
        return set(), f"PEFT load probe imports failed: {exc!r}"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": "meta"},
        )
        model = PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=True)
        modules = {
            canonical_module_path(name.rsplit(".lora_A.", 1)[0])
            for name, _parameter in model.named_parameters()
            if ".lora_A." in name
        }
        return modules, None
    except Exception as exc:
        return set(), f"PEFT load probe failed: {exc!r}"


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_counter(counter: Counter[str]) -> None:
    for key, value in sorted(counter.items()):
        print(f"- {key}: {value}")


def first_items(values: set[str], limit: int) -> list[str]:
    return sorted(values)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PEFT LoRA adapter compatibility for GRPO/SFT handoff.")
    parser.add_argument("config", help="GRPO yaml config path.")
    parser.add_argument("--model-name-or-path", help="Override model.name_or_path from config.")
    parser.add_argument("--adapter-name-or-path", help="Override model.adapter_name_or_path from config.")
    parser.add_argument("--no-model-inspect", action="store_true", help="Skip empty-weight model module inspection.")
    parser.add_argument("--no-peft-probe", action="store_true", help="Skip PEFT meta-load probe.")
    parser.add_argument("--show-limit", type=int, default=30, help="Number of module names to show in mismatch lists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})

    model_name_or_path = args.model_name_or_path or model_cfg.get("name_or_path")
    adapter_name_or_path = args.adapter_name_or_path or model_cfg.get("adapter_name_or_path")
    yaml_target_modules = set(lora_cfg.get("target_modules") or [])

    print_section("Configured Paths")
    print(f"config: {config_path}")
    print(f"model.name_or_path: {model_name_or_path}")
    print(f"model.adapter_name_or_path: {adapter_name_or_path}")

    adapter_dir = resolve_path(adapter_name_or_path)
    if adapter_dir is None:
        raise SystemExit("FAIL: adapter_name_or_path is empty.")

    print_section("3. Adapter Path")
    print(f"adapter path exists: {adapter_dir.exists()} ({adapter_dir})")
    if not adapter_dir.exists():
        raise SystemExit("FAIL: adapter path does not exist.")

    adapter_config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_weight_file(adapter_dir)
    print(f"adapter_config.json exists: {adapter_config_path.exists()}")
    print(f"adapter weight file: {weight_path}")
    if not adapter_config_path.exists():
        raise SystemExit("FAIL: adapter_config.json is missing; this is not a PEFT LoRA adapter directory.")
    if weight_path is None:
        raise SystemExit("FAIL: adapter_model.safetensors/bin is missing; this may be a merged model or wrong checkpoint.")

    adapter_config = load_json(adapter_config_path)
    adapter_target_modules = set(adapter_config.get("target_modules") or [])
    adapter_base = adapter_config.get("base_model_name_or_path")

    print_section("1. Base Model Consistency")
    same_base, base_reason = compare_path_like(model_name_or_path, adapter_base)
    print(f"adapter base_model_name_or_path: {adapter_base}")
    print(f"base model match: {same_base} ({base_reason})")

    print_section("2. Target Module Consistency")
    print(f"yaml target_modules count: {len(yaml_target_modules)}")
    print(f"adapter_config target_modules count: {len(adapter_target_modules)}")
    print(f"yaml - adapter_config: {sorted(yaml_target_modules - adapter_target_modules)}")
    print(f"adapter_config - yaml: {sorted(adapter_target_modules - yaml_target_modules)}")

    print_section("4. Adapter Checkpoint Type")
    print(f"peft_type: {adapter_config.get('peft_type')}")
    print(f"task_type: {adapter_config.get('task_type')}")
    print(f"r: {adapter_config.get('r')}")
    print(f"lora_alpha: {adapter_config.get('lora_alpha')}")
    print(f"inference_mode: {adapter_config.get('inference_mode')}")
    print(f"weight file: {weight_path.name}")
    if (adapter_dir / "config.json").exists() and not adapter_config_path.exists():
        print("WARNING: config.json without adapter_config.json usually means merged model, not LoRA adapter.")

    keys = load_adapter_keys(weight_path)
    lora_modules = {
        canonical_module_path(module)
        for key in keys
        for module in [lora_module_from_key(key)]
        if module
    }
    adapter_leaf_counts = Counter(leaf_name(module) for module in lora_modules)

    print(f"adapter weight keys: {len(keys)}")
    print(f"adapter LoRA modules: {len(lora_modules)}")
    print("adapter LoRA module leaf counts:")
    print_counter(adapter_leaf_counts)

    actual_target_from_weights = set(adapter_leaf_counts)
    print(f"adapter_config target_modules missing from weights: {sorted(adapter_target_modules - actual_target_from_weights)}")
    print(f"weights target_modules not declared in adapter_config: {sorted(actual_target_from_weights - adapter_target_modules)}")

    print_section("5. Current Model Architecture Coverage")
    if args.no_model_inspect:
        print("skipped by --no-model-inspect")
        return

    target_for_model = adapter_target_modules or yaml_target_modules
    model_modules, error = list_model_target_modules(str(model_name_or_path), target_for_model)
    if error:
        print(f"WARNING: {error}")
        return

    model_canonical = {canonical_module_path(module) for module in model_modules}
    model_leaf_counts = Counter(leaf_name(module) for module in model_canonical)
    print(f"current model target modules: {len(model_canonical)}")
    print("current model target module leaf counts:")
    print_counter(model_leaf_counts)

    missing_in_adapter = model_canonical - lora_modules
    unexpected_in_adapter = lora_modules - model_canonical
    print(f"model target modules missing in adapter weights: {len(missing_in_adapter)}")
    for item in first_items(missing_in_adapter, args.show_limit):
        print(f"- missing: {item}")
    print(f"adapter modules not found in current model: {len(unexpected_in_adapter)}")
    for item in first_items(unexpected_in_adapter, args.show_limit):
        print(f"- unexpected: {item}")

    peft_loaded_modules: set[str] = set()
    peft_missing_from_weights: set[str] = set()
    peft_unloaded_weights: set[str] = set()
    print_section("6. PEFT Meta-load Probe")
    if args.no_peft_probe:
        print("skipped by --no-peft-probe")
    else:
        peft_loaded_modules, peft_error = peft_load_probe(str(model_name_or_path), str(adapter_dir))
        if peft_error:
            print(f"WARNING: {peft_error}")
        else:
            peft_loaded_counts = Counter(leaf_name(module) for module in peft_loaded_modules)
            print(f"PEFT loaded LoRA modules: {len(peft_loaded_modules)}")
            print("PEFT loaded LoRA module leaf counts:")
            print_counter(peft_loaded_counts)
            peft_missing_from_weights = peft_loaded_modules - lora_modules
            peft_unloaded_weights = lora_modules - peft_loaded_modules
            print(f"PEFT-loaded modules missing from adapter weights: {len(peft_missing_from_weights)}")
            for item in first_items(peft_missing_from_weights, args.show_limit):
                print(f"- peft missing weight: {item}")
            print(f"adapter weight modules not loaded by PEFT: {len(peft_unloaded_weights)}")
            for item in first_items(peft_unloaded_weights, args.show_limit):
                print(f"- peft unloaded adapter weight: {item}")

    print_section("Summary")
    if not same_base:
        print("FAIL: configured base model and adapter base model differ.")
    elif yaml_target_modules and adapter_target_modules and yaml_target_modules != adapter_target_modules:
        print("WARN: YAML target_modules differ from adapter_config target_modules.")
    elif missing_in_adapter:
        print("FAIL: current model has target modules that are not covered by adapter weights.")
        print("This matches PEFT missing adapter key warnings and points to target_modules or architecture mismatch.")
    elif unexpected_in_adapter:
        print("FAIL: adapter has LoRA modules that are not present in current model.")
        print("This points to base model / Transformers implementation mismatch.")
    elif peft_missing_from_weights:
        print("FAIL: PEFT is creating LoRA modules that are not present in adapter weights.")
        print("This directly explains missing adapter key warnings.")
    elif peft_unloaded_weights:
        print("FAIL: PEFT did not load some adapter weight modules.")
        print("This points to module naming or base model implementation mismatch.")
    else:
        print("PASS: base path, target modules, adapter files, and architecture coverage look compatible.")


if __name__ == "__main__":
    main()
