from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None or name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def build_quantization_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if model_cfg.get("load_in_4bit"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=torch_dtype(model_cfg.get("bnb_4bit_compute_dtype", "float16")),
        )
    if model_cfg.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig | None:
    if not cfg or not cfg.get("enabled", False):
        return None
    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        bias=cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules"),
    )


def load_json_or_hf_dataset(data_cfg: dict[str, Any]) -> Dataset:
    path = data_cfg["path"]
    split = data_cfg.get("split", "train")
    path_obj = Path(path)
    if path_obj.exists() and path_obj.suffix.lower() in {".json", ".jsonl"}:
        loaded = load_dataset("json", data_files=str(path_obj))
        return loaded["train"]
    loaded = load_dataset(path, split=split)
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    return loaded


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_chat_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def to_prompt_text(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
