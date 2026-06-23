#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYSTEM_PROMPT = (
    "\u4f60\u662f\u7269\u6d41\u5ba2\u670d\u610f\u56fe\u8bc6\u522b\u4e0e\u56de\u590d\u52a9\u624b\u3002"
    "\u8bf7\u6839\u636e\u7528\u6237\u8f93\u5165\u5224\u65ad\u610f\u56fe\u3001"
    "\u62bd\u53d6\u624b\u673a\u53f7\u548c\u8fd0\u5355\u53f7\uff0c"
    "\u5e76\u751f\u6210\u5ba2\u670d\u56de\u590d\u3002"
    "\u53ea\u8f93\u51fa\u5408\u6cd5 JSON\uff0c"
    "\u4e0d\u8981\u8f93\u51fa Markdown \u6216\u989d\u5916\u89e3\u91ca\u3002"
    'JSON schema \u5fc5\u987b\u4e3a\uff1a{"intent":"...","slots":{"phone":null\u6216\u5b57\u7b26\u4e32,'
    '"waybill_no":null\u6216\u5b57\u7b26\u4e32},"reply":"..."}'
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a YAML mapping: {path}")
    return data


def resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def dtype_from_name(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_system_prompt(args: argparse.Namespace) -> str:
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8").strip()
    if args.system:
        return args.system
    return DEFAULT_SYSTEM_PROMPT


def load_model_and_tokenizer(config: dict[str, Any], args: argparse.Namespace, repo_dir: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(config["model_name_or_path"])
    adapter_path = resolve_path(config.get("adapter_name_or_path"), repo_dir)
    trust_remote_code = bool(config.get("trust_remote_code", True))

    tokenizer_path = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype_from_name(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def first_model_device(model):
    return next(model.parameters()).device


def generate_once(model, tokenizer, system_prompt: str, user_text: str, config: dict[str, Any], args: argparse.Namespace) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}

    do_sample = bool(config.get("do_sample", False))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(config.get("max_new_tokens", args.max_new_tokens)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(config.get("temperature", args.temperature))
        generation_kwargs["top_p"] = float(config.get("top_p", args.top_p))

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a LoRA adapter using a fixed system prompt.")
    parser.add_argument("config", help="LLaMA-Factory inference YAML with model_name_or_path and adapter_name_or_path.")
    parser.add_argument("--once", help="Run one prompt and exit.")
    parser.add_argument("--system", help="Override the default system prompt.")
    parser.add_argument("--system-file", help="Read the system prompt from a UTF-8 text file.")
    parser.add_argument("--dtype", default="bf16", help="auto, bf16, fp16, or fp32. Default: bf16.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Default: auto.")
    parser.add_argument("--attn-implementation", default=None, help="Optional attention implementation, e.g. sdpa.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path.cwd()
    config = load_yaml(Path(args.config))
    system_prompt = load_system_prompt(args)
    model, tokenizer = load_model_and_tokenizer(config, args, repo_dir)

    if args.once:
        print(generate_once(model, tokenizer, system_prompt, args.once, config, args))
        return 0

    while True:
        try:
            user_text = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            return 0
        print("Assistant:", generate_once(model, tokenizer, system_prompt, user_text, config, args))


if __name__ == "__main__":
    raise SystemExit(main())
