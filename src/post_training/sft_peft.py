from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from packaging.version import Version
from peft import LoraConfig, PeftConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from post_training.common import load_json_or_hf_dataset, set_seed, torch_dtype


TOKEN_COUNT_KEY = "_benchmark_non_padding_tokens"
PADDED_TOKEN_COUNT_KEY = "_benchmark_padded_tokens"
SUPERVISED_TOKEN_COUNT_KEY = "_benchmark_supervised_tokens"
EXAMPLE_COUNT_KEY = "_benchmark_examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct Transformers LoRA/full SFT with measured throughput and BW/DCU metrics."
    )
    parser.add_argument("config", help="Path to the YAML experiment config.")
    parser.add_argument(
        "--output-dir",
        help="Override training.output_dir. Use a new path for every materially different run.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def resolve_finetuning_method(config: dict[str, Any]) -> str:
    method = str(config.get("finetuning", {}).get("method", "lora")).lower()
    if method not in {"lora", "full"}:
        raise ValueError(f"Unsupported finetuning.method: {method!r}; expected 'lora' or 'full'.")
    if method == "lora" and not isinstance(config.get("lora"), dict):
        raise ValueError("LoRA fine-tuning requires a top-level lora configuration.")
    return method


def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return env_rank() == 0


def distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return env_world_size()


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def ensure_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir; historical runs are never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_output_dir_for_ranks(output_dir: Path, timeout_seconds: float = 60.0) -> None:
    """Let rank zero own the non-overwrite check without a pre-DDP mkdir race."""
    ready_path = output_dir / ".launch_ready"
    launch_token = os.environ.get("TORCHELASTIC_RUN_ID") or ":".join(
        (
            os.environ.get("MASTER_ADDR", "single"),
            os.environ.get("MASTER_PORT", "single"),
            os.environ.get("WORLD_SIZE", "1"),
        )
    )
    if env_rank() == 0:
        ensure_new_output_dir(output_dir)
        temporary_ready_path = output_dir / f".launch_ready.{os.getpid()}.tmp"
        temporary_ready_path.write_text(
            f"{launch_token}\n{datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
        os.replace(temporary_ready_path, ready_path)
        return

    deadline = time.monotonic() + timeout_seconds
    while True:
        if ready_path.is_file():
            try:
                marker_lines = ready_path.read_text(encoding="utf-8").splitlines()
                if marker_lines and marker_lines[0] == launch_token:
                    return
            except OSError:
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for rank zero to prepare {output_dir}.")
        time.sleep(0.1)


def check_versions(config: dict[str, Any]) -> dict[str, str]:
    runtime_cfg = config.get("runtime", {})
    expected_transformers = str(runtime_cfg.get("transformers_version", "5.6.0"))
    minimum_peft = str(runtime_cfg.get("minimum_peft_version", "0.18.0"))
    installed_transformers = transformers.__version__
    installed_peft = package_version("peft")

    if Version(installed_transformers) != Version(expected_transformers):
        raise RuntimeError(
            f"This experiment requires transformers=={expected_transformers}; "
            f"found {installed_transformers}."
        )
    if Version(installed_peft) < Version(minimum_peft):
        raise RuntimeError(
            f"Transformers {installed_transformers} requires peft>={minimum_peft} for the supported Trainer path; "
            f"found {installed_peft}. Upgrade PEFT inside the isolated venv first."
        )

    versions = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_hip": str(getattr(torch.version, "hip", None)),
        "torch_cuda": str(getattr(torch.version, "cuda", None)),
        "transformers": installed_transformers,
        "peft": installed_peft,
        "accelerate": package_version("accelerate"),
        "datasets": package_version("datasets"),
    }
    if config.get("training", {}).get("deepspeed"):
        versions["deepspeed"] = package_version("deepspeed")
    report_to = config.get("training", {}).get("report_to", "none")
    report_targets = {report_to} if isinstance(report_to, str) else set(report_to or [])
    if "tensorboard" in report_targets:
        versions["tensorboard"] = package_version("tensorboard")
    return versions


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tokenized_dataset(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        row = dataset[index]
        payload = [row["input_ids"], row["attention_mask"], row["labels"]]
        digest.update(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_and_validate_checkpoint_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(str(model_cfg["name_or_path"]))
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must contain a JSON object: {config_path}")

    text_config = payload.get("text_config")
    if not isinstance(text_config, dict):
        text_config = payload
    architectures = payload.get("architectures") or []
    if not isinstance(architectures, list):
        raise ValueError(f"Model architectures must be a list in {config_path}.")

    expected_model_type = model_cfg.get("expected_checkpoint_model_type")
    if expected_model_type is not None and payload.get("model_type") != expected_model_type:
        raise RuntimeError(
            f"Checkpoint model_type mismatch: expected {expected_model_type!r}, "
            f"found {payload.get('model_type')!r}."
        )
    expected_architecture = model_cfg.get("expected_checkpoint_architecture")
    if expected_architecture is not None and expected_architecture not in architectures:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: expected {expected_architecture!r}, found {architectures!r}."
        )
    expected_text_model_type = model_cfg.get("expected_text_model_type")
    if expected_text_model_type is not None and text_config.get("model_type") != expected_text_model_type:
        raise RuntimeError(
            f"Checkpoint text model_type mismatch: expected {expected_text_model_type!r}, "
            f"found {text_config.get('model_type')!r}."
        )
    expected_hidden_size = model_cfg.get("expected_text_hidden_size")
    if expected_hidden_size is not None and int(text_config.get("hidden_size", -1)) != int(expected_hidden_size):
        raise RuntimeError(
            f"Checkpoint text hidden_size mismatch: expected {expected_hidden_size}, "
            f"found {text_config.get('hidden_size')!r}."
        )

    quantization_config = payload.get("quantization_config")
    if quantization_config is None:
        quantization_config = text_config.get("quantization_config")
    if bool(model_cfg.get("require_unquantized_checkpoint", False)) and quantization_config not in (None, {}):
        raise RuntimeError(
            "This experiment requires the non-quantized checkpoint, but config.json contains "
            f"quantization_config={quantization_config!r}."
        )

    hidden_size = text_config.get("hidden_size")
    return {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "model_type": payload.get("model_type"),
        "architectures": architectures,
        "text_model_type": text_config.get("model_type"),
        "text_hidden_size": int(hidden_size) if hidden_size is not None else None,
        "text_num_hidden_layers": text_config.get("num_hidden_layers"),
        "text_tie_word_embeddings": text_config.get(
            "tie_word_embeddings", payload.get("tie_word_embeddings")
        ),
        "quantization_config_present": quantization_config not in (None, {}),
        "quantization_config": quantization_config,
    }


def resolve_deepspeed_config(
    training_cfg: dict[str, Any],
    model_hidden_size: int | None = None,
) -> dict[str, Any] | None:
    configured_path = training_cfg.get("deepspeed")
    if not configured_path:
        return None

    path = Path(str(configured_path))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DeepSpeed config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DeepSpeed config must contain a JSON object: {path}")

    zero_cfg = payload.get("zero_optimization")
    if not isinstance(zero_cfg, dict):
        raise ValueError(f"DeepSpeed config has no zero_optimization object: {path}")
    stage = int(zero_cfg.get("stage", -1))
    expected_stage = training_cfg.get("expected_deepspeed_stage")
    if expected_stage is not None and stage != int(expected_stage):
        raise ValueError(f"DeepSpeed stage mismatch: expected {expected_stage}, found {stage} in {path}.")
    if bool(training_cfg.get("bf16", True)) and not bool(payload.get("bf16", {}).get("enabled", False)):
        raise ValueError(f"BF16 training requires bf16.enabled=true in {path}.")

    hidden_size_auto_resolution: dict[str, int] = {}
    if stage == 3:
        hidden_size_values = {
            "reduce_bucket_size": (
                model_hidden_size * model_hidden_size if model_hidden_size is not None else None
            ),
            "stage3_prefetch_bucket_size": (
                int(0.9 * model_hidden_size * model_hidden_size)
                if model_hidden_size is not None
                else None
            ),
            "stage3_param_persistence_threshold": (
                10 * model_hidden_size if model_hidden_size is not None else None
            ),
        }
        for key, resolved_value in hidden_size_values.items():
            if zero_cfg.get(key) == "auto":
                if resolved_value is None:
                    raise RuntimeError(
                        f"DeepSpeed ZeRO-3 key {key!r} is 'auto', but the checkpoint has no text hidden_size."
                    )
                zero_cfg[key] = resolved_value
                hidden_size_auto_resolution[key] = resolved_value

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "zero_stage": stage,
        "zero_optimization": zero_cfg,
        "training_arguments_config": payload,
        "hidden_size_auto_resolution": hidden_size_auto_resolution,
        "gradient_accumulation_steps": payload.get("gradient_accumulation_steps"),
        "train_batch_size": payload.get("train_batch_size"),
        "train_micro_batch_size_per_gpu": payload.get("train_micro_batch_size_per_gpu"),
    }


def build_training_arguments(
    training_cfg: dict[str, Any],
    output_dir: Path,
    seed: int,
    deepspeed_setup: dict[str, Any] | None,
) -> TrainingArguments:
    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "logging_strategy": "steps",
        "logging_steps": int(training_cfg.get("logging_steps", 1)),
        "logging_first_step": bool(training_cfg.get("logging_first_step", True)),
        "logging_nan_inf_filter": bool(training_cfg.get("logging_nan_inf_filter", False)),
        "save_strategy": training_cfg.get("save_strategy", "no"),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
        "bf16": bool(training_cfg.get("bf16", True)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
        "report_to": training_cfg.get("report_to", "none"),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
        "dataloader_drop_last": bool(training_cfg.get("dataloader_drop_last", False)),
        "ddp_find_unused_parameters": bool(training_cfg.get("ddp_find_unused_parameters", False)),
        "seed": seed,
        "data_seed": int(training_cfg.get("data_seed", seed)),
        "include_num_input_tokens_seen": "no",
        "skip_memory_metrics": True,
    }
    if training_cfg.get("ddp_backend"):
        training_args_kwargs["ddp_backend"] = training_cfg["ddp_backend"]
    if training_cfg.get("ddp_timeout") is not None:
        training_args_kwargs["ddp_timeout"] = int(training_cfg["ddp_timeout"])
    if training_cfg.get("save_steps") is not None:
        training_args_kwargs["save_steps"] = int(training_cfg["save_steps"])
    if training_cfg.get("save_total_limit") is not None:
        training_args_kwargs["save_total_limit"] = int(training_cfg["save_total_limit"])
    if training_cfg.get("save_only_model") is not None:
        training_args_kwargs["save_only_model"] = bool(training_cfg["save_only_model"])
    if deepspeed_setup is not None:
        training_args_kwargs["deepspeed"] = copy.deepcopy(
            deepspeed_setup["training_arguments_config"]
        )
    if training_cfg.get("max_steps") is not None:
        training_args_kwargs["max_steps"] = int(training_cfg["max_steps"])
    if training_cfg.get("gradient_checkpointing_kwargs"):
        training_args_kwargs["gradient_checkpointing_kwargs"] = dict(
            training_cfg["gradient_checkpointing_kwargs"]
        )
    return TrainingArguments(**training_args_kwargs)


def logical_parameter_numel(parameter: torch.nn.Parameter) -> int:
    ds_numel = getattr(parameter, "ds_numel", None)
    if ds_numel is not None:
        return int(ds_numel)
    ds_shape = getattr(parameter, "ds_shape", None)
    if ds_shape is not None and parameter.numel() == 0:
        return math.prod(int(dimension) for dimension in ds_shape)
    return int(parameter.numel())


def logical_parameter_shape(parameter: torch.nn.Parameter) -> tuple[int, ...]:
    ds_shape = getattr(parameter, "ds_shape", None)
    if ds_shape is not None:
        shape = tuple(int(dimension) for dimension in ds_shape)
    else:
        shape = tuple(int(dimension) for dimension in parameter.shape)
    if math.prod(shape) != logical_parameter_numel(parameter):
        raise RuntimeError(
            "Logical parameter shape/numel mismatch: "
            f"shape={shape}, numel={logical_parameter_numel(parameter)}."
        )
    return shape


def summarize_zero3_partitioned_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    missing_names = [
        name
        for name, parameter in named_parameters
        if not all(hasattr(parameter, attribute) for attribute in ("ds_id", "ds_numel", "ds_shape"))
    ]
    local_partition_numel = 0
    for _, parameter in named_parameters:
        ds_tensor = getattr(parameter, "ds_tensor", None)
        local_partition_numel += int(ds_tensor.numel()) if ds_tensor is not None else int(parameter.numel())
    return {
        "status": "passed" if not missing_names else "incomplete",
        "parameter_tensor_count": len(named_parameters),
        "partitioned_parameter_tensor_count": len(named_parameters) - len(missing_names),
        "missing_partition_metadata_count": len(missing_names),
        "first_missing_partition_metadata_name": missing_names[0] if missing_names else None,
        "logical_parameter_count": sum(
            logical_parameter_numel(parameter) for _, parameter in named_parameters
        ),
        "local_partition_parameter_count": local_partition_numel,
    }


def validate_peft_zero3_save_support() -> dict[str, Any]:
    from deepspeed.runtime.engine import DeepSpeedEngine

    save_checkpoint_parameters = set(inspect.signature(DeepSpeedEngine.save_checkpoint).parameters)
    consolidated_method = getattr(DeepSpeedEngine, "_zero3_consolidated_16bit_state_dict", None)
    consolidated_parameters = (
        set(inspect.signature(consolidated_method).parameters)
        if callable(consolidated_method)
        else set()
    )
    save_checkpoint_supported = "exclude_frozen_parameters" in save_checkpoint_parameters
    consolidated_supported = "exclude_frozen_parameters" in consolidated_parameters
    if not save_checkpoint_supported or not consolidated_supported:
        raise RuntimeError(
            "The vendor DeepSpeed build cannot guarantee PEFT-only ZeRO-3 saving: "
            f"save_checkpoint support={save_checkpoint_supported}, "
            f"consolidated-state support={consolidated_supported}."
        )
    return {
        "status": "passed",
        "trainer_path": (
            "Transformers 5.6 PEFT ZeRO-3 consolidation with exclude_frozen_parameters=true"
        ),
        "deepspeed_save_checkpoint_exclude_frozen_parameters": save_checkpoint_supported,
        "deepspeed_consolidated_state_exclude_frozen_parameters": consolidated_supported,
        "artifact_scope": "LoRA adapter parameters only; frozen base weights are excluded",
    }


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"Tokenized chat template has no input_ids field: {type(value).__name__}")
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Unsupported tokenized chat-template result: {type(value).__name__}")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one chat-template sequence, received a batch.")
        value = value[0]
    return [int(token_id) for token_id in value]


def build_messages(example: dict[str, Any], data_cfg: dict[str, Any]) -> list[dict[str, str]]:
    messages_field = data_cfg.get("messages_field", "messages")
    raw_messages = example.get(messages_field)
    if raw_messages:
        messages = [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            for message in raw_messages
        ]
    else:
        prompt = str(example.get(data_cfg.get("prompt_field", "prompt"), ""))
        response = str(example.get(data_cfg.get("response_field", "response"), ""))
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Each SFT row must end with a non-empty assistant message.")
    if not messages[-1]["content"]:
        raise ValueError("Assistant content must not be empty.")
    return messages


def tokenize_sft_row(
    example: dict[str, Any],
    index: int,
    data_cfg: dict[str, Any],
    tokenizer,
) -> dict[str, list[int]]:
    messages = build_messages(example, data_cfg)
    template_kwargs = dict(data_cfg.get("chat_template_kwargs", {}))
    full_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
    )
    prefix_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )

    if full_ids[: len(prefix_ids)] != prefix_ids:
        mismatch = next(
            (
                position
                for position, (full_token, prefix_token) in enumerate(zip(full_ids, prefix_ids))
                if full_token != prefix_token
            ),
            min(len(full_ids), len(prefix_ids)),
        )
        raise ValueError(
            f"Chat-template prefix mismatch at dataset row {index}, token {mismatch}. "
            "The prompt and full-message serialization must share an exact prefix."
        )

    max_seq_length = int(data_cfg.get("max_seq_length", 2048))
    input_ids = full_ids[:max_seq_length]
    if bool(data_cfg.get("train_on_prompt", False)):
        labels = input_ids.copy()
    else:
        masked_prefix_length = min(len(prefix_ids), len(input_ids))
        labels = [-100] * masked_prefix_length + input_ids[masked_prefix_length:]

    if not any(label != -100 for label in labels):
        raise ValueError(
            f"Dataset row {index} has zero supervised tokens after truncation to {max_seq_length}."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def select_dataset(dataset: Dataset, data_cfg: dict[str, Any]) -> tuple[Dataset, dict[str, Any]]:
    source_rows = len(dataset)
    max_samples = data_cfg.get("max_samples")
    selection = "all_rows"
    if max_samples is not None:
        max_samples = min(int(max_samples), source_rows)
        seed = int(data_cfg.get("sample_seed", 42))
        strategy = data_cfg.get("sample_strategy", "shuffle")
        if strategy == "first":
            dataset = dataset.select(range(max_samples))
            selection = f"first_{max_samples}"
        elif strategy == "shuffle":
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))
            selection = f"seeded_shuffle_{max_samples}_seed_{seed}"
        else:
            raise ValueError(f"Unsupported data.sample_strategy: {strategy}")
    return dataset, {
        "source_sample_count": source_rows,
        "selected_sample_count": len(dataset),
        "selection": selection,
    }


def percentile(values: list[int] | list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_token_stats(tokenized: Dataset, max_seq_length: int) -> dict[str, Any]:
    lengths = [len(input_ids) for input_ids in tokenized["input_ids"]]
    supervised = [sum(label != -100 for label in labels) for labels in tokenized["labels"]]
    return {
        "configured_max_seq_length": max_seq_length,
        "sample_count": len(lengths),
        "non_padding_tokens_one_epoch": int(sum(lengths)),
        "sequence_length_mean": float(statistics.fmean(lengths)),
        "sequence_length_p50": percentile(lengths, 50),
        "sequence_length_p95": percentile(lengths, 95),
        "sequence_length_max": int(max(lengths)),
        "supervised_tokens_mean": float(statistics.fmean(supervised)),
        "supervised_tokens_p50": percentile(supervised, 50),
        "supervised_tokens_p95": percentile(supervised, 95),
        "supervised_tokens_max": int(max(supervised)),
        "zero_supervised_rows": int(sum(value == 0 for value in supervised)),
        "truncated_rows": int(sum(value == max_seq_length for value in lengths)),
    }


class CountingDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None) -> None:
        self.inner = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=-100,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.inner(features)
        batch[TOKEN_COUNT_KEY] = int(batch["attention_mask"].sum().item())
        batch[PADDED_TOKEN_COUNT_KEY] = int(batch["input_ids"].numel())
        batch[SUPERVISED_TOKEN_COUNT_KEY] = int((batch["labels"] != -100).sum().item())
        batch[EXAMPLE_COUNT_KEY] = len(features)
        return batch


class BenchmarkTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed_non_padding_tokens_local = 0
        self.executed_padded_tokens_local = 0
        self.executed_supervised_tokens_local = 0
        self.executed_examples_local = 0
        self.executed_microbatches_local = 0
        self.gradient_payload_bytes_local: int | None = None
        self.gradient_dtype_elements_local: dict[str, int] = {}

    def training_step(self, model, inputs, *args, **kwargs):
        self.executed_non_padding_tokens_local += int(inputs.pop(TOKEN_COUNT_KEY))
        self.executed_padded_tokens_local += int(inputs.pop(PADDED_TOKEN_COUNT_KEY))
        self.executed_supervised_tokens_local += int(inputs.pop(SUPERVISED_TOKEN_COUNT_KEY))
        self.executed_examples_local += int(inputs.pop(EXAMPLE_COUNT_KEY))
        self.executed_microbatches_local += 1

        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.gradient_payload_bytes_local is None:
            payload = 0
            dtype_elements: Counter[str] = Counter()
            for parameter in model.parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad
                payload += gradient.numel() * gradient.element_size()
                dtype_elements[str(gradient.dtype)] += gradient.numel()
            if payload:
                self.gradient_payload_bytes_local = payload
                self.gradient_dtype_elements_local = dict(dtype_elements)
        return loss


@dataclass
class GpuSample:
    timestamp: str
    device_index: int
    utilization_pct: float
    memory_utilization_pct: float | None = None
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None


def parse_visible_physical_devices() -> set[int] | None:
    value = os.environ.get("HIP_VISIBLE_DEVICES")
    if not value:
        return None
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError:
        return None


def parse_memory_pair(line: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)\s*/\s*(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None, None

    def to_mib(value: str, unit: str) -> float:
        number = float(value)
        return number * 1024.0 if unit.lower().startswith("g") else number

    return to_mib(match.group(1), match.group(2)), to_mib(match.group(3), match.group(4))


def parse_hy_smi(output: str, timestamp: str) -> list[GpuSample]:
    """Parse the hy-smi table by column name, never by percentage position.

    The BW image prints ``VRAM% HCU% Dec% Enc%``.  Treating the final
    percentage as compute utilization therefore records ``Enc%`` (usually
    zero), which is valid-looking but wrong benchmark evidence.
    """

    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def fields(line: str) -> list[str]:
        cleaned = ansi_escape.sub("", line).replace("|", " ").strip()
        return cleaned.split()

    lines = output.splitlines()
    header_fields: list[str] | None = None
    header_line_index: int | None = None
    device_column = utilization_column = None
    memory_utilization_column: int | None = None

    for line_index, line in enumerate(lines):
        tokens = fields(line)
        normalized = [token.upper().rstrip(":") for token in tokens]
        if not normalized:
            continue
        device_candidates = [
            index for index, token in enumerate(normalized) if token in {"HCU", "DCU", "GPU", "CARD"}
        ]
        utilization_candidates = [
            index for index, token in enumerate(normalized) if token in {"HCU%", "DCU%", "GPU%", "UTIL%"}
        ]
        if device_candidates and utilization_candidates:
            header_fields = normalized
            header_line_index = line_index
            device_column = device_candidates[0]
            utilization_column = utilization_candidates[0]
            memory_utilization_column = next(
                (
                    index
                    for index, token in enumerate(normalized)
                    if token in {"VRAM%", "MEM%", "MEMORY%"}
                ),
                None,
            )
            break

    if header_fields is None or header_line_index is None:
        return []

    def parse_percent(value: str) -> float:
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", value)
        if match is None:
            raise ValueError(f"Expected a percentage cell in hy-smi output, found {value!r}.")
        result = float(match.group(1))
        if not math.isfinite(result) or not 0.0 <= result <= 100.0:
            raise ValueError(f"hy-smi percentage is outside [0, 100]: {value!r}.")
        return result

    samples: list[GpuSample] = []
    observed_indices: set[int] = set()
    required_column = max(
        index
        for index in (device_column, utilization_column, memory_utilization_column)
        if index is not None
    )
    for line in lines[header_line_index + 1 :]:
        tokens = fields(line)
        if len(tokens) <= required_column or not tokens[device_column].isdigit():
            continue
        device_index = int(tokens[device_column])
        if device_index in observed_indices:
            raise ValueError(f"Duplicate hy-smi row for device {device_index} in one snapshot.")
        observed_indices.add(device_index)
        utilization = parse_percent(tokens[utilization_column])
        memory_utilization = (
            parse_percent(tokens[memory_utilization_column])
            if memory_utilization_column is not None
            else None
        )
        memory_used, memory_total = parse_memory_pair(line)
        samples.append(
            GpuSample(
                timestamp=timestamp,
                device_index=device_index,
                utilization_pct=utilization,
                memory_utilization_pct=memory_utilization,
                memory_used_mib=memory_used,
                memory_total_mib=memory_total,
            )
        )
    return samples


def parse_rocm_smi(output: str, timestamp: str) -> list[GpuSample]:
    by_index: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        index_match = re.search(r"GPU\[?(\d+)\]?", line, re.IGNORECASE)
        if not index_match:
            continue
        device_index = int(index_match.group(1))
        values = by_index.setdefault(device_index, {})
        utilization_match = re.search(r"(?:GPU|HCU)\s*use.*?([0-9.]+)\s*%?", line, re.IGNORECASE)
        memory_match = re.search(r"memory\s*use.*?([0-9.]+)\s*%?", line, re.IGNORECASE)
        if utilization_match:
            values["utilization"] = float(utilization_match.group(1))
        if memory_match:
            values["memory_utilization"] = float(memory_match.group(1))
    return [
        GpuSample(
            timestamp=timestamp,
            device_index=index,
            utilization_pct=values["utilization"],
            memory_utilization_pct=values.get("memory_utilization"),
        )
        for index, values in sorted(by_index.items())
        if "utilization" in values
    ]


def parse_nvidia_smi(output: str, timestamp: str) -> list[GpuSample]:
    samples: list[GpuSample] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            device_index = int(parts[0])
            utilization = float(parts[1])
            memory_utilization = float(parts[2])
            memory_used = float(parts[3])
            memory_total = float(parts[4])
        except ValueError:
            continue
        samples.append(
            GpuSample(
                timestamp=timestamp,
                device_index=device_index,
                utilization_pct=utilization,
                memory_utilization_pct=memory_utilization,
                memory_used_mib=memory_used,
                memory_total_mib=memory_total,
            )
        )
    return samples


def parse_custom_regex(output: str, timestamp: str, pattern: str) -> list[GpuSample]:
    compiled = re.compile(pattern)
    required = {"index", "utilization_gpu_pct"}
    if not required.issubset(compiled.groupindex):
        raise ValueError(f"gpu_monitor.line_regex requires named groups: {sorted(required)}")
    samples: list[GpuSample] = []
    for line in output.splitlines():
        match = compiled.search(line)
        if not match:
            continue
        groups = match.groupdict()

        def optional_float(name: str) -> float | None:
            value = groups.get(name)
            return float(value) if value not in (None, "") else None

        samples.append(
            GpuSample(
                timestamp=timestamp,
                device_index=int(groups["index"]),
                utilization_pct=float(groups["utilization_gpu_pct"]),
                memory_utilization_pct=optional_float("memory_used_pct"),
                memory_used_mib=optional_float("memory_used_mib"),
                memory_total_mib=optional_float("memory_total_mib"),
            )
        )
    return samples


class GpuMonitor:
    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        self.config = config
        self.interval = float(config.get("interval_seconds", 2.0))
        self.timeout = float(config.get("timeout_seconds", 5.0))
        self.output_dir = output_dir
        self.raw_log_path = output_dir / "gpu_monitor_raw.log"
        self.csv_path = output_dir / "gpu_samples.csv"
        self.command, self.parser = self._resolve_command()
        self.line_regex = config.get("line_regex")
        self.visible_devices = parse_visible_physical_devices()
        self.samples: list[GpuSample] = []
        self.status = "not_started"
        self.first_error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self.started_at: str | None = None
        self.stopped_at: str | None = None
        self.duration_seconds: float | None = None
        self._started_monotonic: float | None = None

    def _resolve_command(self) -> tuple[list[str] | None, str | None]:
        requested = self.config.get("command", "auto")
        parser = self.config.get("parser", "auto")
        if requested not in (None, "auto"):
            command = shlex.split(requested) if isinstance(requested, str) else [str(item) for item in requested]
            if not command:
                return None, None
            if parser == "auto":
                parser = Path(command[0]).name.replace("-", "_")
            return command, parser

        hy_smi = shutil.which("hy-smi")
        if hy_smi:
            return [hy_smi], "hy_smi"
        rocm_smi = shutil.which("rocm-smi")
        if rocm_smi:
            return [rocm_smi, "--showuse", "--showmemuse"], "rocm_smi"
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            return [
                nvidia_smi,
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ], "nvidia_smi"
        return None, None

    def start(self) -> None:
        if self.started_at is not None or self.status != "not_started":
            return
        if not self.command:
            self.status = "command_missing"
            self.first_error = "No hy-smi, rocm-smi, or nvidia-smi executable was found."
            return
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        self._started_monotonic = time.monotonic()
        self.status = "running"
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.stopped_at is not None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1.0)
        if self.status == "running":
            self.status = "ok" if self.samples else "no_samples"
        if self._started_monotonic is not None:
            self.duration_seconds = time.monotonic() - self._started_monotonic
            self.stopped_at = datetime.now().isoformat(timespec="milliseconds")
        self._write_csv()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self._sample_once()
            if self._consecutive_failures >= 3:
                break
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.0, self.interval - elapsed))

    def _append_raw(self, timestamp: str, result: subprocess.CompletedProcess[str] | None, error: str | None) -> None:
        with self.raw_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"===== {timestamp} command={json.dumps(self.command)} =====\n")
            if error:
                handle.write(f"ERROR: {error}\n")
            if result is not None:
                handle.write(f"returncode: {result.returncode}\n")
                handle.write("stdout:\n")
                handle.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    handle.write("\n")
                handle.write("stderr:\n")
                handle.write(result.stderr)
                if result.stderr and not result.stderr.endswith("\n"):
                    handle.write("\n")

    def _sample_once(self) -> None:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        try:
            result = subprocess.run(
                self.command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"GPU monitor timed out after {self.timeout}s: {exc}"
            self._append_raw(timestamp, None, error)
            self._record_failure("timeout", error)
            return
        except OSError as exc:
            error = f"GPU monitor command failed to start: {exc}"
            self._append_raw(timestamp, None, error)
            self._record_failure("command_failed", error)
            return

        self._append_raw(timestamp, result, None)
        if result.returncode != 0:
            error = f"GPU monitor exited with code {result.returncode}."
            self._record_failure("command_failed", error)
            return

        try:
            if self.parser == "hy_smi":
                parsed = parse_hy_smi(result.stdout, timestamp)
            elif self.parser == "rocm_smi":
                parsed = parse_rocm_smi(result.stdout, timestamp)
            elif self.parser == "nvidia_smi":
                parsed = parse_nvidia_smi(result.stdout, timestamp)
            elif self.parser == "regex":
                parsed = parse_custom_regex(result.stdout, timestamp, str(self.line_regex or ""))
            else:
                raise ValueError(f"Unsupported GPU monitor parser: {self.parser}")
        except Exception as exc:
            self._record_failure("parse_failed", repr(exc))
            return

        if self.visible_devices is not None:
            parsed = [sample for sample in parsed if sample.device_index in self.visible_devices]
        if not parsed:
            self._record_failure("parse_failed", "No selected-device utilization rows were parsed.")
            return
        self.samples.extend(parsed)
        self._consecutive_failures = 0

    def _record_failure(self, status: str, error: str) -> None:
        self._consecutive_failures += 1
        if self.first_error is None:
            self.first_error = error
        if self._consecutive_failures >= 3:
            self.status = status

    def _write_csv(self) -> None:
        if not self.samples:
            return
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "device_index",
                    "utilization_pct",
                    "memory_utilization_pct",
                    "memory_used_mib",
                    "memory_total_mib",
                ]
            )
            for sample in self.samples:
                writer.writerow(
                    [
                        sample.timestamp,
                        sample.device_index,
                        sample.utilization_pct,
                        sample.memory_utilization_pct,
                        sample.memory_used_mib,
                        sample.memory_total_mib,
                    ]
                )

    def preflight(self, expected_device_count: int) -> dict[str, Any]:
        """Probe the configured device tool before an expensive training run."""
        if not self.command:
            self.status = "command_missing"
            self.first_error = "No hy-smi, rocm-smi, or nvidia-smi executable was found."
        else:
            self._sample_once()

        probe_summary = self.summary()
        if self.samples and self._consecutive_failures == 0:
            probe_summary["status"] = "ok"
        valid, reason = validate_gpu_utilization_summary(
            probe_summary,
            expected_device_count=expected_device_count,
            minimum_samples_per_device=1,
        )
        result = {
            **probe_summary,
            "valid": valid,
            "validation_reason": reason,
            "note": "Idle preflight samples are excluded from the training-window utilization result.",
        }
        if valid:
            self.samples.clear()
            self._consecutive_failures = 0
            self.first_error = None
            self.status = "not_started"
        else:
            self.status = "preflight_failed"
        return result

    def summary(self) -> dict[str, Any]:
        by_device: dict[int, list[GpuSample]] = {}
        for sample in self.samples:
            by_device.setdefault(sample.device_index, []).append(sample)

        devices: list[dict[str, Any]] = []
        device_means: list[float] = []
        for device_index, samples in sorted(by_device.items()):
            utilization = [sample.utilization_pct for sample in samples]
            memory_used = [sample.memory_used_mib for sample in samples if sample.memory_used_mib is not None]
            mean_utilization = float(statistics.fmean(utilization))
            device_means.append(mean_utilization)
            devices.append(
                {
                    "device_index": device_index,
                    "sample_count": len(samples),
                    "utilization_mean_pct": mean_utilization,
                    "utilization_p50_pct": percentile(utilization, 50),
                    "utilization_p95_pct": percentile(utilization, 95),
                    "utilization_max_pct": max(utilization),
                    "peak_memory_used_mib": max(memory_used) if memory_used else None,
                }
            )

        return {
            "status": self.status,
            "source": self.parser,
            "command": self.command,
            "interval_seconds": self.interval,
            "sampling_window": "Trainer on_train_begin through on_train_end callbacks",
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_seconds": self.duration_seconds,
            "semantics": "Device-tool instantaneous compute utilization; not assumed comparable across vendors.",
            "selected_physical_devices": sorted(self.visible_devices) if self.visible_devices is not None else None,
            "sample_count": len(self.samples),
            "utilization_mean_pct": float(statistics.fmean(device_means)) if device_means else None,
            "utilization_min_card_mean_pct": min(device_means) if device_means else None,
            "utilization_max_card_mean_pct": max(device_means) if device_means else None,
            "devices": devices,
            "first_error": self.first_error,
            "raw_log": str(self.raw_log_path),
            "csv": str(self.csv_path) if self.samples else None,
        }


def validate_gpu_utilization_summary(
    summary: dict[str, Any],
    expected_device_count: int,
    minimum_samples_per_device: int,
    minimum_utilization_mean_pct: float | None = None,
) -> tuple[bool, str]:
    if summary.get("status") != "ok":
        return False, f"monitor status is {summary.get('status')!r}, expected 'ok'"

    devices = summary.get("devices") or []
    observed_indices = {int(device["device_index"]) for device in devices}
    expected_indices_raw = summary.get("selected_physical_devices")
    if expected_indices_raw is not None:
        expected_indices = {int(index) for index in expected_indices_raw}
        if len(expected_indices) != expected_device_count:
            return False, (
                f"HIP_VISIBLE_DEVICES selects {len(expected_indices)} devices {sorted(expected_indices)}, "
                f"but world size is {expected_device_count}"
            )
        if observed_indices != expected_indices:
            return False, (
                f"observed physical devices {sorted(observed_indices)}, "
                f"expected {sorted(expected_indices)}"
            )
    elif len(observed_indices) != expected_device_count:
        return False, (
            f"observed {len(observed_indices)} devices {sorted(observed_indices)}, "
            f"expected {expected_device_count}"
        )

    under_sampled = {
        int(device["device_index"]): int(device.get("sample_count", 0))
        for device in devices
        if int(device.get("sample_count", 0)) < minimum_samples_per_device
    }
    if under_sampled:
        return False, (
            f"per-device sample counts {under_sampled} are below minimum "
            f"{minimum_samples_per_device}"
        )
    device_means = [device.get("utilization_mean_pct") for device in devices]
    if any(
        value is None
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 100.0
        for value in device_means
    ):
        return False, "one or more per-device utilization means are missing or outside [0, 100]"
    utilization_mean = summary.get("utilization_mean_pct")
    if utilization_mean is None:
        return False, "utilization mean is unavailable"
    if not math.isfinite(float(utilization_mean)) or not 0.0 <= float(utilization_mean) <= 100.0:
        return False, f"utilization mean is invalid: {utilization_mean!r}"
    if (
        minimum_utilization_mean_pct is not None
        and any(float(value) < float(minimum_utilization_mean_pct) for value in device_means)
    ):
        inactive_devices = {
            int(device["device_index"]): float(device["utilization_mean_pct"])
            for device in devices
            if float(device["utilization_mean_pct"]) < float(minimum_utilization_mean_pct)
        }
        return False, (
            f"per-device training-window utilization means {inactive_devices} are below required "
            f"{float(minimum_utilization_mean_pct):.4f}%"
        )
    return True, "all selected devices have valid utilization samples"


class GpuMonitorCallback(TrainerCallback):
    """Align device-tool sampling with the Trainer runtime window."""

    def __init__(self, monitor: GpuMonitor) -> None:
        self.monitor = monitor

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        if is_main_process():
            self.monitor.start()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if is_main_process():
            self.monitor.stop()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def bytes_to_gib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024.0**3)


def bytes_to_mib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024.0**2)


def collect_hardware() -> dict[str, Any]:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gib": bytes_to_gib(properties.total_memory),
                "gcn_arch_name": getattr(properties, "gcnArchName", None),
            }
        )
    return {
        "visible_device_count": device_count,
        "devices": devices,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def reduce_sum_counts(trainer: BenchmarkTrainer) -> dict[str, int]:
    device = trainer.args.device
    values = torch.tensor(
        [
            trainer.executed_non_padding_tokens_local,
            trainer.executed_padded_tokens_local,
            trainer.executed_supervised_tokens_local,
            trainer.executed_examples_local,
            trainer.executed_microbatches_local,
        ],
        dtype=torch.int64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values_list = [int(value) for value in values.cpu().tolist()]
    return dict(
        zip(
            (
                "executed_non_padding_tokens",
                "executed_padded_tokens",
                "executed_supervised_tokens",
                "executed_examples",
                "executed_microbatches",
            ),
            values_list,
        )
    )


def reduce_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def gather_memory_stats(device: torch.device) -> dict[str, Any]:
    local = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.int64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [torch.zeros_like(local) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, local)
    else:
        gathered = [local]
    per_rank = [
        {
            "rank": rank,
            "peak_allocated_gib": bytes_to_gib(int(values[0].item())),
            "peak_reserved_gib": bytes_to_gib(int(values[1].item())),
        }
        for rank, values in enumerate(gathered)
    ]
    return {
        "source": "torch.cuda allocator API (HIP compatibility layer on BW)",
        "per_rank": per_rank,
        "peak_allocated_gib_max": max(item["peak_allocated_gib"] for item in per_rank),
        "peak_reserved_gib_max": max(item["peak_reserved_gib"] for item in per_rank),
    }


def build_communication_stats(
    world_size: int,
    gradient_payload_bytes: int | None,
    optimizer_steps: int,
    strategy: str,
    payload_source: str,
    zero_stage: int | None = None,
) -> dict[str, Any]:
    if world_size <= 1:
        return {
            "strategy": "single_process",
            "backend": None,
            "gradient_payload_bytes_per_sync": 0,
            "estimated_ring_bytes_per_rank_total": 0,
            "estimated_ring_bytes_job_total": 0,
            "sync_count_assumption": 0,
            "gradient_payload_source": payload_source,
            "zero_stage": zero_stage,
            "estimate_status": "not_applicable",
            "scope": "No inter-GPU communication in the single-device smoke run.",
        }

    payload = gradient_payload_bytes
    backend = torch.distributed.get_backend() if torch.distributed.is_initialized() else None
    if zero_stage == 3:
        return {
            "strategy": strategy,
            "backend": backend,
            "zero_stage": zero_stage,
            "gradient_payload_bytes_per_sync": None,
            "gradient_payload_mib_per_sync": None,
            "trainable_parameter_payload_reference_bytes": payload,
            "trainable_parameter_payload_reference_mib": bytes_to_mib(payload),
            "estimated_ring_bytes_per_rank_total": None,
            "estimated_ring_gib_per_rank_total": None,
            "estimated_ring_bytes_job_total": None,
            "estimated_ring_gib_job_total": None,
            "sync_count_assumption": None,
            "gradient_payload_source": payload_source,
            "estimate_status": "not_derivable_from_gradient_payload",
            "scope": (
                "The BF16 trainable-parameter payload is reported as a scale reference only. ZeRO-3 also performs "
                "implementation-dependent parameter all-gathers/prefetches and gradient reduce-scatters across "
                "micro-batches, so actual link traffic is not derived from parameter count and optimizer steps."
            ),
        }

    per_rank_total = None
    job_total = None
    if payload is not None:
        per_rank_total = payload * 2.0 * (world_size - 1) / world_size * optimizer_steps
        job_total = payload * 2.0 * (world_size - 1) * optimizer_steps
    return {
        "strategy": strategy,
        "backend": backend,
        "zero_stage": zero_stage,
        "gradient_payload_bytes_per_sync": payload,
        "gradient_payload_mib_per_sync": bytes_to_mib(payload),
        "estimated_ring_bytes_per_rank_total": per_rank_total,
        "estimated_ring_gib_per_rank_total": bytes_to_gib(per_rank_total),
        "estimated_ring_bytes_job_total": job_total,
        "estimated_ring_gib_job_total": bytes_to_gib(job_total),
        "sync_count_assumption": optimizer_steps,
        "gradient_payload_source": payload_source,
        "estimate_status": "theoretical_ring_equivalent",
        "scope": (
            "Theoretical ring-equivalent trainable-gradient/update payload, assuming one synchronized update per "
            "optimizer step. For ZeRO-2 this approximates reduce-scatter plus updated-parameter all-gather; it is not "
            "measured link traffic and excludes initialization, buffers, bucket padding, protocol overhead, and "
            "retransmission."
        ),
    }


def validate_saved_adapter(
    output_dir: Path,
    expected_tensor_count: int | None,
    expected_key_prefix: str | None,
    expected_parameter_count: int | None,
) -> dict[str, Any]:
    config_path = output_dir / "adapter_config.json"
    safetensors_path = output_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise RuntimeError(f"Missing saved PEFT config: {config_path}")
    if not safetensors_path.is_file() or safetensors_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty saved PEFT weights: {safetensors_path}")
    forbidden_full_weights = [
        path
        for path in (
            output_dir / "model.safetensors",
            output_dir / "model.safetensors.index.json",
            output_dir / "pytorch_model.bin",
        )
        if path.exists()
    ]
    if forbidden_full_weights:
        raise RuntimeError(
            f"PEFT adapter output unexpectedly contains full-model weights: {forbidden_full_weights}."
        )

    peft_config = PeftConfig.from_pretrained(output_dir)
    from safetensors import safe_open

    serialized_parameter_count = 0
    tensor_dtypes: Counter[str] = Counter()
    first_nonfinite_key: str | None = None
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            serialized_parameter_count += int(tensor.numel())
            tensor_dtypes[str(tensor.dtype)] += 1
            if (
                first_nonfinite_key is None
                and tensor.is_floating_point()
                and not bool(torch.isfinite(tensor).all().item())
            ):
                first_nonfinite_key = key
    if expected_tensor_count is not None and len(keys) != expected_tensor_count:
        raise RuntimeError(
            f"Saved adapter tensor count mismatch: expected {expected_tensor_count}, found {len(keys)}."
        )
    if expected_key_prefix and any(not key.startswith(expected_key_prefix) for key in keys):
        unexpected = next(key for key in keys if not key.startswith(expected_key_prefix))
        raise RuntimeError(
            f"Saved adapter key prefix mismatch: expected {expected_key_prefix!r}, found {unexpected!r}."
        )
    if expected_parameter_count is not None and serialized_parameter_count != expected_parameter_count:
        raise RuntimeError(
            "Saved adapter parameter-count mismatch: "
            f"expected {expected_parameter_count}, found {serialized_parameter_count}."
        )
    if first_nonfinite_key is not None:
        raise RuntimeError(f"Saved adapter contains a non-finite tensor: {first_nonfinite_key!r}.")
    return {
        "status": "passed",
        "validation_scope": (
            "PEFT config plus adapter-only safetensors keys, tensor count, numel, dtype, and finiteness"
        ),
        "config": str(config_path),
        "weights": str(safetensors_path),
        "weight_bytes": safetensors_path.stat().st_size,
        "tensor_count": len(keys),
        "serialized_parameter_count": serialized_parameter_count,
        "tensor_dtypes": dict(sorted(tensor_dtypes.items())),
        "all_tensors_finite": first_nonfinite_key is None,
        "key_prefix": expected_key_prefix,
        "base_model_name_or_path": peft_config.base_model_name_or_path,
    }


def validate_saved_full_checkpoint(
    output_dir: Path,
    expected_model_class: str | None,
    expected_parameter_shapes: dict[str, tuple[int, ...]],
    expected_parameter_count: int,
) -> dict[str, Any]:
    """Validate a gathered Full checkpoint without loading its tensors into RAM."""

    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing saved model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    architectures = model_config.get("architectures") or []
    if expected_model_class and expected_model_class not in architectures:
        raise RuntimeError(
            f"Saved model architecture mismatch: expected {expected_model_class!r}, found {architectures!r}."
        )

    forbidden = [
        path
        for path in (output_dir / "adapter_config.json", output_dir / "adapter_model.safetensors")
        if path.exists()
    ]
    if forbidden:
        raise RuntimeError(f"Full checkpoint unexpectedly contains PEFT artifacts: {forbidden}")

    index_path = output_dir / "model.safetensors.index.json"
    single_path = output_dir / "model.safetensors"
    expected_weight_map: dict[str, str] | None = None
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index_payload = json.load(handle)
        raw_weight_map = index_payload.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise RuntimeError(f"Invalid or empty safetensors weight_map: {index_path}")
        expected_weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
        shard_paths = [output_dir / name for name in sorted(set(expected_weight_map.values()))]
    elif single_path.is_file():
        shard_paths = [single_path]
    else:
        raise RuntimeError(
            f"No Full safetensors checkpoint found in {output_dir}; expected {single_path.name} "
            f"or {index_path.name}."
        )

    from safetensors import safe_open

    actual_key_to_shard: dict[str, str] = {}
    actual_key_to_shape: dict[str, tuple[int, ...]] = {}
    total_numel = 0
    total_weight_bytes = 0
    for shard_path in shard_paths:
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty Full checkpoint shard: {shard_path}")
        total_weight_bytes += shard_path.stat().st_size
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual_key_to_shard:
                    raise RuntimeError(
                        f"Duplicate tensor key {key!r} in {actual_key_to_shard[key]!r} and {shard_path.name!r}."
                    )
                actual_key_to_shard[key] = shard_path.name
                shape = tuple(int(dimension) for dimension in handle.get_slice(key).get_shape())
                actual_key_to_shape[key] = shape
                total_numel += math.prod(shape)

    if not actual_key_to_shard:
        raise RuntimeError("Full checkpoint contains no tensor keys.")
    lora_keys = [key for key in actual_key_to_shard if ".lora_" in key.lower()]
    if lora_keys:
        raise RuntimeError(f"Full checkpoint unexpectedly contains LoRA tensor key: {lora_keys[0]!r}")
    expected_shape_counts = Counter(expected_parameter_shapes.values())
    actual_shape_counts = Counter(actual_key_to_shape.values())
    missing_shape_counts = expected_shape_counts - actual_shape_counts
    extra_shape_counts = actual_shape_counts - expected_shape_counts
    if missing_shape_counts or extra_shape_counts:
        missing_example = next(iter(missing_shape_counts.items()), None)
        extra_example = next(iter(extra_shape_counts.items()), None)
        raise RuntimeError(
            "Full checkpoint parameter-shape multiset mismatch: "
            f"missing_example={missing_example}, extra_example={extra_example}. "
            "The saved key namespace may differ, but every trained tensor shape and multiplicity must match."
        )
    expected_parameter_numel_from_shapes = sum(
        math.prod(shape) for shape in expected_parameter_shapes.values()
    )
    if expected_parameter_numel_from_shapes != expected_parameter_count:
        raise RuntimeError(
            f"Pre-save parameter manifest is internally inconsistent: expected {expected_parameter_count}, "
            f"shape manifest contains {expected_parameter_numel_from_shapes}."
        )
    if total_numel != expected_parameter_count:
        raise RuntimeError(
            f"Serialized Full checkpoint parameter count mismatch: expected {expected_parameter_count}, "
            f"found {total_numel}."
        )
    if expected_weight_map is not None:
        expected_keys = set(expected_weight_map)
        actual_keys = set(actual_key_to_shard)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:3]
            extra = sorted(actual_keys - expected_keys)[:3]
            raise RuntimeError(f"Safetensors index/header key mismatch: missing={missing}, extra={extra}.")
        wrong_shards = [
            key
            for key, expected_shard in expected_weight_map.items()
            if actual_key_to_shard[key] != expected_shard
        ]
        if wrong_shards:
            key = wrong_shards[0]
            raise RuntimeError(
                f"Safetensors shard mismatch for {key!r}: index={expected_weight_map[key]!r}, "
                f"header={actual_key_to_shard[key]!r}."
            )

    return {
        "status": "passed",
        "validation_scope": "structural_safetensors_headers_only_no_reload_or_forward",
        "config": str(config_path),
        "architectures": architectures,
        "index": str(index_path) if index_path.is_file() else None,
        "shards": [str(path) for path in shard_paths],
        "shard_count": len(shard_paths),
        "weight_bytes": total_weight_bytes,
        "tensor_count": len(actual_key_to_shard),
        "serialized_tensor_numel": total_numel,
        "expected_parameter_tensor_count": len(expected_parameter_shapes),
        "expected_parameter_numel": expected_parameter_count,
        "parameter_shape_multiset_status": "passed",
        "contains_lora_keys": False,
    }


def validate_full_checkpoint_reload(
    output_dir: Path,
    expected_model_class: str | None,
    expected_parameter_shapes: dict[str, tuple[int, ...]],
    expected_parameter_count: int,
    device: torch.device,
) -> dict[str, Any]:
    """Reload the gathered checkpoint as a fresh model instance and run a finite forward."""

    # Rank zero performs this validation while the other ranks wait for the result.
    # Clear Transformers' global ZeRO-3 weakref first; otherwise from_pretrained()
    # enters deepspeed.zero.Init and waits for collectives from all ranks.
    from transformers.integrations.deepspeed import (
        is_deepspeed_zero3_enabled,
        unset_hf_deepspeed_config,
    )

    unset_hf_deepspeed_config()
    if is_deepspeed_zero3_enabled():
        raise RuntimeError("Could not disable the global Transformers ZeRO-3 load context.")

    reload_model, loading_info = AutoModelForCausalLM.from_pretrained(
        output_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    try:
        loading_failures = {
            key: loading_info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
            if loading_info.get(key)
        }
        if loading_failures:
            raise RuntimeError(f"Reloaded Full checkpoint has loading errors: {loading_failures}.")
        if expected_model_class and reload_model.__class__.__name__ != expected_model_class:
            raise RuntimeError(
                f"Reloaded model class mismatch: expected {expected_model_class}, "
                f"found {reload_model.__class__.__name__}."
            )
        reloaded_shapes = {
            name: tuple(int(dimension) for dimension in parameter.shape)
            for name, parameter in reload_model.named_parameters()
        }
        missing_keys = sorted(set(expected_parameter_shapes) - set(reloaded_shapes))
        extra_keys = sorted(set(reloaded_shapes) - set(expected_parameter_shapes))
        if missing_keys or extra_keys:
            raise RuntimeError(
                f"Reloaded canonical parameter keys differ: missing={missing_keys[:3]}, extra={extra_keys[:3]}."
            )
        shape_mismatches = [
            key
            for key, expected_shape in expected_parameter_shapes.items()
            if reloaded_shapes[key] != expected_shape
        ]
        if shape_mismatches:
            key = shape_mismatches[0]
            raise RuntimeError(
                f"Reloaded parameter shape mismatch for {key!r}: expected {expected_parameter_shapes[key]}, "
                f"found {reloaded_shapes[key]}."
            )
        reloaded_parameter_count = sum(parameter.numel() for parameter in reload_model.parameters())
        if reloaded_parameter_count != expected_parameter_count:
            raise RuntimeError(
                f"Reloaded parameter count mismatch: expected {expected_parameter_count}, "
                f"found {reloaded_parameter_count}."
            )

        reload_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True, use_fast=True)
        probe = reload_tokenizer("你好", return_tensors="pt")
        probe = {
            key: value
            for key, value in probe.items()
            if key in {"input_ids", "attention_mask"}
        }
        reload_model.to(device)
        reload_model.eval()
        probe = {key: value.to(device) for key, value in probe.items()}
        with torch.inference_mode():
            logits = reload_model(**probe).logits
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("Reloaded Full checkpoint produced non-finite logits.")
        logits_shape = list(logits.shape)
        logits_dtype = str(logits.dtype)
        del logits, probe, reload_tokenizer
        return {
            "status": "passed",
            "validation_scope": "fresh_instance_from_pretrained_parameter_manifest_and_bf16_forward",
            "model_class": reload_model.__class__.__name__,
            "parameter_tensor_count": len(reloaded_shapes),
            "parameter_count": reloaded_parameter_count,
            "logits_shape": logits_shape,
            "logits_dtype": logits_dtype,
            "logits_finite": True,
            "loading_info_status": "clean",
            "loading_info": {
                key: loading_info.get(key, [])
                for key in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            },
            "transformers_zero3_load_context_disabled": True,
        }
    finally:
        del reload_model
        torch.cuda.empty_cache()


def classify_exception(exc: BaseException) -> tuple[str, bool | None]:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, KeyboardInterrupt):
        return "user_interrupt", False
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "hiperroroutofmemory" in text or "out of memory" in text:
        return "hip_oom", True
    if "collective" in text and "timeout" in text or "allreduce" in text and "timeout" in text:
        return "collective_timeout", False
    if "nonfinite" in text or "non-finite" in text or "nan loss" in text:
        return "nonfinite_loss", False
    if "required gpu utilization is unavailable" in text:
        return "gpu_utilization_unavailable", False
    return "runtime_error", False


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def write_report(report: dict[str, Any], benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_path = benchmark_dir / "benchmark_metrics.json"
    json_content = json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(json_path, json_content)

    token_stats = report.get("effective_sequence_length", {})
    batch = report.get("batch", {})
    parameters = report.get("parameters", {})
    memory = report.get("memory", {})
    utilization = report.get("gpu_utilization", {})
    throughput = report.get("throughput", {})
    timing = report.get("timing", {})
    communication = report.get("communication", {})
    progress = report.get("validation_progress", {})
    utilization_validation = report.get("gpu_utilization_validation", {})
    comparison = report.get("comparison") or {}

    effective_length = (
        f"cap={token_stats.get('configured_max_seq_length')}; "
        f"mean/p50/p95/max="
        f"{format_number(token_stats.get('sequence_length_mean'), 1)}/"
        f"{format_number(token_stats.get('sequence_length_p50'), 1)}/"
        f"{format_number(token_stats.get('sequence_length_p95'), 1)}/"
        f"{format_number(token_stats.get('sequence_length_max'), 0)}"
    )
    training_window_memory = memory.get("training_window") or memory
    end_to_end_memory = memory.get("end_to_end") or {}
    memory_text = (
        f"training-window max allocated {format_number(training_window_memory.get('peak_allocated_gib_max'))} GiB; "
        f"max reserved {format_number(training_window_memory.get('peak_reserved_gib_max'))} GiB"
    )
    end_to_end_memory_text = (
        f"max allocated {format_number(end_to_end_memory.get('peak_allocated_gib_max'))} GiB; "
        f"max reserved {format_number(end_to_end_memory.get('peak_reserved_gib_max'))} GiB; "
        f"{end_to_end_memory.get('scope')}"
        if end_to_end_memory
        else "not available"
    )
    utilization_text = (
        f"{format_number(utilization.get('utilization_mean_pct'))}% mean "
        f"({utilization.get('source')}, collection={utilization.get('status')}, "
        f"validation={utilization_validation.get('valid', 'not_run')})"
    )
    if report.get("gpu_count", 1) == 1:
        communication_text = "0 (single GPU)"
    elif communication.get("estimate_status") == "not_derivable_from_gradient_payload":
        communication_text = (
            f"{format_number(communication.get('trainable_parameter_payload_reference_mib'))} MiB BF16 "
            "trainable-parameter payload "
            "reference; ZeRO-3 link traffic not measured or derived"
        )
    else:
        communication_text = (
            f"{format_number(communication.get('gradient_payload_mib_per_sync'))} MiB gradients/sync; "
            f"estimated {format_number(communication.get('estimated_ring_gib_per_rank_total'))} GiB/rank/run"
        )
    artifact_type = report.get("artifact_type", "unknown")
    artifact_status = progress.get("artifact_validation_status")
    validation_text = (
        f"train {progress.get('completed_optimizer_steps')}/{progress.get('expected_optimizer_steps')} steps; "
        f"{artifact_type}={artifact_status}; held-out eval={progress.get('heldout_eval')}"
    )
    rows: list[tuple[str, Any]] = [
        ("GPU", report.get("gpu")),
        ("模型", report.get("model")),
        (
            "模型参数范围",
            f"{report.get('model_scope')}; direct total/trainable/frozen="
            f"{parameters.get('total_parameters')}/{parameters.get('trainable_parameters')}/"
            f"{parameters.get('frozen_parameters')}",
        ),
        ("训练方式", report.get("training_method")),
        ("有效序列长度", effective_length),
        ("样本数", report.get("samples")),
        ("Epoch", report.get("epochs")),
        ("Per-device batch size", batch.get("per_device_batch_size")),
        ("单次全局 micro batch", batch.get("global_micro_batch_size")),
        ("Gradient accumulation", batch.get("gradient_accumulation_steps")),
        ("Global batch size", batch.get("global_batch_size")),
        ("每 Epoch 尾部有效 Global batch", batch.get("tail_effective_global_batch_size_per_epoch")),
        (
            "可训练参数量",
            f"{parameters.get('trainable_parameters')} ({format_number(parameters.get('trainable_percent'), 4)}%)",
        ),
        ("冻结参数量", parameters.get("frozen_parameters")),
        ("单卡显存", memory_text),
        ("端到端保存/验证峰值", end_to_end_memory_text),
        ("GPU 利用率", utilization_text),
        ("Tokens/s/GPU", format_number(throughput.get("measured_non_padding_tokens_per_second_per_gpu"))),
    ]
    has_historical_baseline = comparison.get("baseline_tokens_per_second_per_gpu") is not None
    if has_historical_baseline:
        baseline_label = str(comparison.get("baseline_label", "A800 历史"))
        rows.extend(
            [
                (
                    "BW 按历史公式估算 Tokens/s/GPU",
                    format_number(throughput.get("a800_compatible_estimated_tokens_per_second_per_gpu")),
                ),
                (
                    f"{baseline_label}估算 Tokens/s/GPU",
                    format_number(comparison.get("baseline_tokens_per_second_per_gpu")),
                ),
                (
                    f"{baseline_label}参数范围",
                    f"total={comparison.get('baseline_total_parameters_approx')}; "
                    f"trainable={comparison.get('baseline_trainable_parameters_approx')}; "
                    f"{comparison.get('parameter_scope_note')}",
                ),
            ]
        )
    rows.extend(
        [
        ("单步耗时", f"{format_number(timing.get('seconds_per_optimizer_step'))} s/optimizer step"),
        ("验证进度", validation_text),
        ("直接原因", report.get("direct_reason")),
        ("通信规模", communication_text),
        ("是否 OOM", report.get("oom")),
        ("对比定位", comparison.get("classification")),
        ("结论", report.get("conclusion")),
        ]
    )

    markdown_lines = [
        "# Direct Transformers SFT Benchmark",
        "",
        "| 字段 | 结果 |",
        "| --- | --- |",
    ]
    for key, value in rows:
        safe_value = str(value).replace("|", "\\|").replace("\n", " ")
        markdown_lines.append(f"| {key} | {safe_value} |")
    markdown_lines.extend(
        [
            "",
            "`Tokens/s/GPU` uses non-padding input tokens from every micro-batch actually executed, "
            "summed across ranks, divided by Trainer runtime and world size.",
        ]
    )
    if has_historical_baseline:
        markdown_lines.extend(
            [
                "The BW historical-formula value and historical baseline value are estimates. Formula alignment "
                "does not make tokenizer/template versions, framework scope, or hardware attribution equivalent.",
                "BW memory uses the PyTorch/HIP allocator while the historical baseline used a platform SMI; "
                "their measurement boundaries are not directly equivalent.",
            ]
        )
    else:
        markdown_lines.append(
            "Memory uses the PyTorch/HIP allocator; no historical throughput or memory baseline is claimed."
        )
    markdown_lines.extend(
        [
            "This report validates training feasibility and efficiency only; it does not establish held-out task quality.",
            "",
        ]
    )
    markdown_path = benchmark_dir / "benchmark_metrics.md"
    atomic_write_text(markdown_path, "\n".join(markdown_lines))


def build_base_report(
    config_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    versions: dict[str, str],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    finetuning_method = resolve_finetuning_method(config)
    world_size = distributed_world_size()
    per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    names = sorted({item["name"] for item in hardware.get("devices", [])})
    memory_sizes = sorted({round(item["total_memory_gib"], 1) for item in hardware.get("devices", [])})
    gpu_text = f"{world_size} x {'/'.join(names) or 'unknown'}"
    if len(memory_sizes) == 1:
        gpu_text += f", {memory_sizes[0]} GiB/card"

    model_name = model_cfg["name_or_path"]
    distributed_method = (
        f"DeepSpeed ZeRO-{training_cfg.get('expected_deepspeed_stage')}"
        if training_cfg.get("deepspeed") and training_cfg.get("expected_deepspeed_stage") is not None
        else "DeepSpeed" if training_cfg.get("deepspeed") else (
        "DDP" if world_size > 1 else "single process"
        )
    )
    method_label = "PEFT LoRA" if finetuning_method == "lora" else "Full-parameter SFT"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "initializing",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "gpu": gpu_text,
        "gpu_count": world_size,
        "model": model_name,
        "model_scope": config.get("finetuning", {}).get(
            "model_scope", "the model returned by Transformers AutoModelForCausalLM"
        ),
        "finetuning_method": finetuning_method,
        "finetuning_type": finetuning_method,
        "artifact_type": "peft_adapter" if finetuning_method == "lora" else "full_checkpoint",
        "training_method": f"Transformers 5.6 + {method_label} + BF16 + {distributed_method}",
        "samples": data_cfg.get("max_samples") or "all selected rows",
        "epochs": training_cfg.get("num_train_epochs", 1),
        "batch": {
            "per_device_batch_size": per_device_batch,
            "global_micro_batch_size": per_device_batch * world_size,
            "gradient_accumulation_steps": grad_accum,
            "global_batch_size": per_device_batch * world_size * grad_accum,
        },
        "versions": versions,
        "hardware": hardware,
        "comparison": config.get("comparison"),
        "direct_reason": "initializing",
        "oom": None,
        "conclusion": "Run has not completed.",
    }


def run(
    config_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    expected_world_size = int(config["training"].get("expected_world_size", env_world_size()))
    if env_world_size() != expected_world_size:
        raise RuntimeError(
            f"World-size mismatch: config expects {expected_world_size}, launcher provided {env_world_size()}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("No accelerator is visible through torch.cuda/HIP compatibility APIs.")

    seed = int(config["training"].get("seed", 42))
    model_cfg = config["model"]
    data_cfg = config["data"]
    finetuning_method = resolve_finetuning_method(config)
    lora_cfg = config.get("lora", {})
    full_cfg = {**config.get("finetuning", {}), **config.get("full", {})}
    training_cfg = config["training"]
    benchmark_cfg = config.get("benchmark", {})
    checkpoint_config = load_and_validate_checkpoint_config(model_cfg)
    report["checkpoint_config"] = checkpoint_config
    if bool(training_cfg.get("bf16", True)) is not True or bool(training_cfg.get("fp16", False)):
        raise RuntimeError("This BW benchmark requires training.bf16=true and training.fp16=false.")
    if str(model_cfg.get("torch_dtype", "bfloat16")).lower() not in {"bf16", "bfloat16"}:
        raise RuntimeError("This BW benchmark requires model.torch_dtype=bfloat16.")

    deepspeed_setup = resolve_deepspeed_config(
        training_cfg,
        model_hidden_size=checkpoint_config.get("text_hidden_size"),
    )
    if deepspeed_setup is not None:
        report["deepspeed"] = {
            key: copy.deepcopy(value)
            for key, value in deepspeed_setup.items()
            if key != "training_arguments_config"
        }
    if finetuning_method == "full":
        if deepspeed_setup is None or int(deepspeed_setup["zero_stage"]) != 3:
            raise RuntimeError("This BW Full benchmark requires DeepSpeed ZeRO-3.")
        if not bool(
            deepspeed_setup["zero_optimization"].get(
                "stage3_gather_16bit_weights_on_model_save", False
            )
        ):
            raise RuntimeError(
                "Full ZeRO-3 saving requires stage3_gather_16bit_weights_on_model_save=true."
            )
        if not bool(training_cfg.get("save_only_model", False)):
            raise RuntimeError("Full SFT requires training.save_only_model=true to avoid optimizer-state checkpoints.")
        if not bool(training_cfg.get("require_safetensors", False)):
            raise RuntimeError("Full SFT requires training.require_safetensors=true for structural validation.")
        if not bool(full_cfg.get("fresh_reload_validation", False)):
            raise RuntimeError("Full SFT requires finetuning.fresh_reload_validation=true.")
    elif deepspeed_setup is not None and int(deepspeed_setup["zero_stage"]) == 3:
        if not bool(training_cfg.get("zero3_adapter_only_save", False)):
            raise RuntimeError(
                "PEFT ZeRO-3 requires training.zero3_adapter_only_save=true so the optimized "
                "exclude-frozen-parameters save path is treated as a hard requirement."
            )
        if not bool(
            deepspeed_setup["zero_optimization"].get(
                "stage3_gather_16bit_weights_on_model_save", False
            )
        ):
            raise RuntimeError(
                "PEFT ZeRO-3 requires stage3_gather_16bit_weights_on_model_save=true as a safe fallback."
            )

    # TrainingArguments owns the Transformers DeepSpeed configuration and must stay alive before
    # from_pretrained() so Transformers enters deepspeed.zero.Init for ZeRO-3 model loading.
    train_args = build_training_arguments(training_cfg, output_dir, seed, deepspeed_setup)
    # TrainingArguments initializes the local distributed device first. Seed afterward so each HIP
    # rank does not touch every visible device before local-rank device selection is established.
    set_seed(seed)
    if deepspeed_setup is not None:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        requested_zero3 = int(deepspeed_setup["zero_stage"]) == 3
        zero3_model_load_context_active = bool(is_deepspeed_zero3_enabled())
        if zero3_model_load_context_active != requested_zero3:
            raise RuntimeError(
                "Transformers DeepSpeed model-load context mismatch: "
                f"requested_zero3={requested_zero3}, active={zero3_model_load_context_active}."
            )
        hf_deepspeed_config = getattr(train_args, "hf_deepspeed_config", None)
        effective_model_load_config = getattr(hf_deepspeed_config, "config", None)
        if not isinstance(effective_model_load_config, dict):
            raise RuntimeError("TrainingArguments did not retain an effective Hugging Face DeepSpeed config.")
        if requested_zero3:
            invalid_hidden_size_keys = [
                key
                for key in (
                    "reduce_bucket_size",
                    "stage3_prefetch_bucket_size",
                    "stage3_param_persistence_threshold",
                )
                if not isinstance(
                    effective_model_load_config.get("zero_optimization", {}).get(key), int
                )
                or effective_model_load_config.get("zero_optimization", {}).get(key) <= 0
            ]
            if invalid_hidden_size_keys:
                raise RuntimeError(
                    "DeepSpeed model-load config has unresolved or invalid hidden-size values: "
                    f"{invalid_hidden_size_keys}."
                )
        report["deepspeed"]["zero3_model_load_context_active"] = zero3_model_load_context_active
        report["deepspeed"]["model_load_effective_config"] = copy.deepcopy(
            effective_model_load_config
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    source_path = Path(str(data_cfg["path"]))
    dataset = load_json_or_hf_dataset(data_cfg)
    expected_source_sample_count = data_cfg.get("expected_source_sample_count")
    if expected_source_sample_count is not None and len(dataset) != int(expected_source_sample_count):
        raise RuntimeError(
            f"Source dataset row-count mismatch: expected {expected_source_sample_count}, found {len(dataset)}."
        )
    dataset_sha256 = sha256_file(source_path)
    expected_sha256 = data_cfg.get("expected_sha256")
    if expected_sha256 and dataset_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"Source dataset SHA256 mismatch: expected {expected_sha256}, found {dataset_sha256}."
        )
    dataset, dataset_selection = select_dataset(dataset, data_cfg)
    tokenized = dataset.map(
        lambda row, index: tokenize_sft_row(row, index, data_cfg, tokenizer),
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=bool(data_cfg.get("load_from_cache_file", False)),
        desc="Tokenizing direct SFT dataset",
    )
    token_stats = build_token_stats(tokenized, int(data_cfg.get("max_seq_length", 2048)))
    if token_stats["zero_supervised_rows"]:
        raise RuntimeError(f"Found {token_stats['zero_supervised_rows']} rows with zero supervised tokens.")
    report["dataset"] = {
        "path": str(source_path),
        "sha256": dataset_sha256,
        **dataset_selection,
    }
    report["samples"] = dataset_selection["selected_sample_count"]
    report["effective_sequence_length"] = token_stats
    report["tokenized_dataset_sha256"] = sha256_tokenized_dataset(tokenized)
    model_path = Path(str(model_cfg["name_or_path"]))
    report["model_artifact_hashes"] = {
        name: sha256_file(model_path / name)
        for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
        if (model_path / name).is_file()
    }

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "local_files_only": bool(model_cfg.get("local_files_only", True)),
        "dtype": torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if model_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    base_model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **load_kwargs)
    expected_model_class = model_cfg.get("expected_model_class")
    if expected_model_class and base_model.__class__.__name__ != expected_model_class:
        raise RuntimeError(
            f"Unexpected model class: expected {expected_model_class}, found {base_model.__class__.__name__}."
        )
    original_use_cache = bool(getattr(base_model.config, "use_cache", True))
    base_model.config.use_cache = False

    base_named_parameters = list(base_model.named_parameters())
    base_parameter_count = sum(
        logical_parameter_numel(parameter) for _, parameter in base_named_parameters
    )
    if deepspeed_setup is not None and int(deepspeed_setup["zero_stage"]) == 3:
        partitioning = summarize_zero3_partitioned_parameters(base_named_parameters)
        report["deepspeed"]["model_load_partitioning"] = partitioning
        if partitioning["status"] != "passed":
            raise RuntimeError(
                "ZeRO-3 partitioned from_pretrained did not cover every base-model parameter; "
                f"first missing parameter: {partitioning['first_missing_partition_metadata_name']!r}."
            )
        if int(partitioning["logical_parameter_count"]) != base_parameter_count:
            raise RuntimeError("ZeRO-3 partition report disagrees with the logical base-parameter count.")
    expected_base_parameters = lora_cfg.get(
        "expected_base_parameters", model_cfg.get("expected_base_parameters")
    )
    if expected_base_parameters is not None and base_parameter_count != int(expected_base_parameters):
        raise RuntimeError(
            f"Base parameter mismatch: expected {expected_base_parameters}, found {base_parameter_count}."
        )

    lora_module_names: list[str] = []
    lora_target_module_counts: dict[str, int] = {}
    if finetuning_method == "lora":
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg["target_modules"]),
        )
        model = get_peft_model(base_model, peft_config)
        metadata_model = model.get_base_model()
        lora_module_names = sorted(
            name
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") and len(getattr(module, "lora_A")) > 0
        )
        expected_lora_module_count = lora_cfg.get("expected_lora_module_count")
        if expected_lora_module_count is not None and len(lora_module_names) != int(expected_lora_module_count):
            raise RuntimeError(
                f"LoRA module-count mismatch: expected {expected_lora_module_count}, found {len(lora_module_names)}."
            )
        lora_target_module_counts = dict(
            sorted(Counter(name.rsplit(".", 1)[-1] for name in lora_module_names).items())
        )
        expected_target_module_counts = lora_cfg.get("expected_target_module_counts")
        if expected_target_module_counts is not None:
            normalized_expected_counts = {
                str(name): int(value) for name, value in expected_target_module_counts.items()
            }
            if lora_target_module_counts != dict(sorted(normalized_expected_counts.items())):
                raise RuntimeError(
                    "LoRA per-target module counts mismatch: "
                    f"expected {normalized_expected_counts}, found {lora_target_module_counts}."
                )
        forbidden_substrings = [
            str(value).lower() for value in lora_cfg.get("forbidden_module_name_substrings", [])
        ]
        forbidden_lora_names = [
            name
            for name in lora_module_names
            if any(value in name.lower() for value in forbidden_substrings)
        ]
        if forbidden_lora_names:
            raise RuntimeError(
                f"LoRA unexpectedly targeted a forbidden module: {forbidden_lora_names[0]!r}."
            )
    else:
        model = base_model
        metadata_model = model
        unexpected_lora_names = [name for name, _ in model.named_parameters() if ".lora_" in name.lower()]
        if unexpected_lora_names:
            raise RuntimeError(
                f"Full fine-tuning model unexpectedly contains a LoRA parameter: {unexpected_lora_names[0]!r}."
            )

    input_require_grads_enabled = False
    if bool(training_cfg.get("gradient_checkpointing", False)) and bool(
        training_cfg.get("gradient_checkpointing_input_require_grads", False)
    ):
        enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
        if not callable(enable_input_require_grads):
            enable_input_require_grads = getattr(metadata_model, "enable_input_require_grads", None)
        if not callable(enable_input_require_grads):
            raise RuntimeError("Gradient checkpointing requested input gradients, but the model cannot enable them.")
        enable_input_require_grads()
        input_require_grads_enabled = True
    named_parameters = list(model.named_parameters())
    parameter_shapes = {
        name: logical_parameter_shape(parameter)
        for name, parameter in named_parameters
    }
    trainable_parameter_names = sorted(name for name, parameter in named_parameters if parameter.requires_grad)
    frozen_parameter_names = sorted(name for name, parameter in named_parameters if not parameter.requires_grad)
    trainable_parameter_count = sum(
        logical_parameter_numel(parameter)
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    frozen_parameter_count = sum(
        logical_parameter_numel(parameter)
        for _, parameter in named_parameters
        if not parameter.requires_grad
    )
    total_parameter_count = trainable_parameter_count + frozen_parameter_count
    trainable_percent = trainable_parameter_count / total_parameter_count * 100.0

    if finetuning_method == "lora":
        expected_trainable = lora_cfg.get("expected_trainable_parameters")
        expected_total = lora_cfg.get(
            "expected_total_parameters", lora_cfg.get("expected_total_parameters_with_adapter")
        )
        expected_frozen = lora_cfg.get("expected_frozen_parameters")
        if expected_trainable is not None and trainable_parameter_count != int(expected_trainable):
            raise RuntimeError(
                f"LoRA trainable-parameter mismatch: expected {expected_trainable}, found {trainable_parameter_count}."
            )
        if expected_total is not None and total_parameter_count != int(expected_total):
            raise RuntimeError(
                f"LoRA total-parameter mismatch: expected {expected_total}, found {total_parameter_count}."
            )
        if expected_frozen is not None and frozen_parameter_count != int(expected_frozen):
            raise RuntimeError(
                f"LoRA frozen-parameter mismatch: expected {expected_frozen}, found {frozen_parameter_count}."
            )
        print(
            f"LoRA trainable parameters: {trainable_parameter_count}; total parameters: {total_parameter_count}; "
            f"trainable percent: {trainable_percent:.6f}"
        )
    else:
        expected_total = full_cfg.get("expected_total_parameters")
        expected_trainable = full_cfg.get("expected_trainable_parameters")
        expected_frozen = full_cfg.get("expected_frozen_parameters")
        minimum_trainable_percent = float(full_cfg.get("minimum_trainable_percent", 99.99))
        if expected_total is not None and total_parameter_count != int(expected_total):
            raise RuntimeError(
                f"Full total-parameter mismatch: expected {expected_total}, found {total_parameter_count}."
            )
        if expected_trainable is not None and trainable_parameter_count != int(expected_trainable):
            raise RuntimeError(
                f"Full trainable-parameter mismatch: expected {expected_trainable}, found {trainable_parameter_count}."
            )
        if expected_frozen is not None and frozen_parameter_count != int(expected_frozen):
            raise RuntimeError(
                f"Full frozen-parameter mismatch: expected {expected_frozen}, found {frozen_parameter_count}."
            )
        if trainable_parameter_count <= 0 or trainable_percent < minimum_trainable_percent:
            raise RuntimeError(
                f"Full parameter coverage is too low: {trainable_parameter_count}/{total_parameter_count} "
                f"({trainable_percent:.6f}%), required >= {minimum_trainable_percent:.6f}%."
            )
        print(
            f"full trainable parameters: {trainable_parameter_count}; total parameters: {total_parameter_count}; "
            f"trainable percent: {trainable_percent:.6f}"
        )

    report["model_class"] = metadata_model.__class__.__name__
    parameter_report = {
        "base_parameters": base_parameter_count,
        "total_parameters": total_parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "frozen_parameters": frozen_parameter_count,
        "trainable_percent": trainable_percent,
        "frozen_percent": frozen_parameter_count / total_parameter_count * 100.0,
        "parameter_tensor_count": len(named_parameters),
        "trainable_parameter_tensor_count": len(trainable_parameter_names),
        "frozen_parameter_tensor_count": len(frozen_parameter_names),
        "trainable_parameter_names_sha256": sha256_lines(trainable_parameter_names),
        "frozen_parameter_names_sha256": sha256_lines(frozen_parameter_names),
        "parameter_shape_manifest_sha256": sha256_lines(
            [f"{name}:{','.join(str(value) for value in parameter_shapes[name])}" for name in sorted(parameter_shapes)]
        ),
    }
    if finetuning_method == "lora":
        parameter_report.update(
            {
                "total_parameters_with_adapter": total_parameter_count,
                "configured_target_modules": list(lora_cfg["target_modules"]),
                "lora_module_count": len(lora_module_names),
                "lora_target_module_counts": lora_target_module_counts,
                "lora_module_names_sha256": sha256_lines(lora_module_names),
                "logical_parameter_counting": (
                    "ds_numel/ds_shape when present, otherwise torch parameter numel/shape"
                ),
            }
        )
    report["parameters"] = parameter_report
    actual_attention = getattr(metadata_model.config, "_attn_implementation", None)
    report["attention_implementation"] = actual_attention
    qwen35_linear_attention: dict[str, Any] | None = None
    if metadata_model.__class__.__name__.startswith("Qwen3_5"):
        modeling_module = __import__(metadata_model.__class__.__module__, fromlist=["is_fast_path_available"])
        fast_path_available = bool(getattr(modeling_module, "is_fast_path_available", False))
        qwen35_linear_attention = {
            "fast_path_available": fast_path_available,
            "backend": (
                "flash-linear-attention plus causal-conv1d fast path"
                if fast_path_available
                else "Transformers torch fallback"
            ),
        }
        report["qwen35_linear_attention"] = qwen35_linear_attention
    distributed_label = (
        f"DeepSpeed ZeRO-{deepspeed_setup['zero_stage']}"
        if deepspeed_setup is not None
        else ("DDP" if expected_world_size > 1 else "single process")
    )
    method_label = "PEFT LoRA" if finetuning_method == "lora" else "Full-parameter SFT"
    attention_label = f"{actual_attention or 'model default'} self-attention"
    if qwen35_linear_attention is not None:
        attention_label += f" + {qwen35_linear_attention['backend']} for linear-attention layers"
    report["training_method"] = (
        f"Transformers {transformers.__version__} + {method_label} + BF16 + "
        f"{attention_label} + {distributed_label}"
    )
    report["gradient_checkpointing"] = {
        "enabled": bool(training_cfg.get("gradient_checkpointing", False)),
        "kwargs": training_cfg.get("gradient_checkpointing_kwargs"),
        "input_require_grads_enabled": input_require_grads_enabled,
    }

    collator = CountingDataCollator(
        tokenizer,
        pad_to_multiple_of=data_cfg.get("pad_to_multiple_of"),
    )
    trainer = BenchmarkTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if (
        finetuning_method == "lora"
        and deepspeed_setup is not None
        and int(deepspeed_setup["zero_stage"]) == 3
    ):
        report["artifact_save"] = validate_peft_zero3_save_support()
        report["artifact_save"].update(
            {
                "intermediate_save_strategy": str(train_args.save_strategy),
                "save_only_model": bool(train_args.save_only_model),
                "stage3_gather_16bit_weights_on_model_save": bool(
                    deepspeed_setup["zero_optimization"].get(
                        "stage3_gather_16bit_weights_on_model_save", False
                    )
                ),
            }
        )
    if deepspeed_setup is not None:
        deepspeed_plugin = getattr(trainer.accelerator.state, "deepspeed_plugin", None)
        effective_deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", None)
        if not isinstance(effective_deepspeed_config, dict):
            raise RuntimeError("Trainer did not expose an effective DeepSpeed configuration before training.")
        effective_zero_stage = int(
            effective_deepspeed_config.get("zero_optimization", {}).get("stage", -1)
        )
        if effective_zero_stage != int(deepspeed_setup["zero_stage"]):
            raise RuntimeError(
                f"Effective DeepSpeed stage mismatch: requested {deepspeed_setup['zero_stage']}, "
                f"resolved {effective_zero_stage}."
            )
        expected_deepspeed_batch_values = {
            "train_micro_batch_size_per_gpu": int(training_cfg.get("per_device_train_batch_size", 1)),
            "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
            "train_batch_size": (
                int(training_cfg.get("per_device_train_batch_size", 1))
                * int(training_cfg.get("gradient_accumulation_steps", 1))
                * expected_world_size
            ),
        }
        for key, expected_value in expected_deepspeed_batch_values.items():
            actual_value = effective_deepspeed_config.get(key)
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Effective DeepSpeed {key} mismatch: expected {expected_value}, found {actual_value!r}."
                )
        if bool(training_cfg.get("bf16", True)) and not bool(
            effective_deepspeed_config.get("bf16", {}).get("enabled", False)
        ):
            raise RuntimeError("Effective DeepSpeed configuration does not have BF16 enabled.")
        report["deepspeed"]["effective_config"] = json_safe(effective_deepspeed_config)
    monitor = GpuMonitor(benchmark_cfg.get("gpu_monitor", {}), benchmark_dir)
    trainer.add_callback(GpuMonitorCallback(monitor))
    world_size = distributed_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(f"Initialized distributed world size {world_size}, expected {expected_world_size}.")

    local_microbatches_per_epoch = len(trainer.get_train_dataloader())
    updates_per_epoch = max(
        math.ceil(local_microbatches_per_epoch / train_args.gradient_accumulation_steps),
        1,
    )
    expected_optimizer_steps = (
        int(training_cfg["max_steps"])
        if training_cfg.get("max_steps") is not None and int(training_cfg["max_steps"]) > 0
        else math.ceil(float(training_cfg.get("num_train_epochs", 1)) * updates_per_epoch)
    )
    tail_microbatches_per_rank = local_microbatches_per_epoch % train_args.gradient_accumulation_steps
    if tail_microbatches_per_rank == 0:
        tail_microbatches_per_rank = train_args.gradient_accumulation_steps
    report["batch"].update(
        {
            "local_microbatches_per_epoch": local_microbatches_per_epoch,
            "optimizer_steps_per_epoch": updates_per_epoch,
            "tail_microbatches_per_rank_per_epoch": tail_microbatches_per_rank,
            "tail_effective_global_batch_size_per_epoch": (
                tail_microbatches_per_rank
                * int(training_cfg.get("per_device_train_batch_size", 1))
                * world_size
            ),
            "global_batch_size_semantics": (
                "nominal full accumulation window; an epoch-tail update can be smaller"
            ),
        }
    )

    device = trainer.args.device
    preflight_valid = True
    if bool(benchmark_cfg.get("require_gpu_utilization", False)):
        if is_main_process():
            preflight_summary = monitor.preflight(expected_device_count=world_size)
            report["gpu_monitor_preflight"] = preflight_summary
            report["gpu_utilization"] = monitor.summary()
            preflight_valid = bool(preflight_summary["valid"])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            preflight_tensor = torch.tensor(int(preflight_valid), dtype=torch.int32, device=device)
            torch.distributed.broadcast(preflight_tensor, src=0)
            preflight_valid = bool(preflight_tensor.item())
        if not preflight_valid:
            reason = (
                report.get("gpu_monitor_preflight", {}).get("validation_reason")
                if is_main_process()
                else "rank-zero GPU monitor preflight failed"
            )
            raise RuntimeError(
                f"Required GPU utilization is unavailable during preflight: {reason}. "
                "Fix the monitor command/parser before starting this benchmark."
            )

    torch.cuda.reset_peak_memory_stats(device)
    distributed_barrier()
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    try:
        train_result = trainer.train()
        torch.cuda.synchronize(device)
        wall_runtime_local = time.perf_counter() - wall_start
    except BaseException:
        if is_main_process():
            monitor.stop()
            report["gpu_utilization"] = monitor.summary()
            report["throughput"] = {
                "status": "partial_local_rank_only",
                "executed_non_padding_tokens_local_rank": trainer.executed_non_padding_tokens_local,
                "executed_padded_tokens_local_rank": trainer.executed_padded_tokens_local,
                "executed_supervised_tokens_local_rank": trainer.executed_supervised_tokens_local,
                "executed_examples_local_rank": trainer.executed_examples_local,
                "executed_microbatches_local_rank": trainer.executed_microbatches_local,
                "measured_non_padding_tokens_per_second_per_gpu": None,
            }
            report["timing"] = {
                "partial_wall_runtime_seconds_local_rank": time.perf_counter() - wall_start,
                "seconds_per_optimizer_step": None,
            }
            report["memory"] = {
                "source": "torch.cuda allocator API (partial rank-zero failure evidence)",
                "peak_allocated_gib_max": bytes_to_gib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_gib_max": bytes_to_gib(torch.cuda.max_memory_reserved(device)),
            }
            report["validation_progress"] = {
                "completed_optimizer_steps": int(trainer.state.global_step),
                "expected_optimizer_steps": expected_optimizer_steps,
                "completed_microbatches_local_rank": trainer.executed_microbatches_local,
                "artifact_validation_status": "not_reached",
                "heldout_eval": "not_configured",
            }
            if finetuning_method == "lora":
                report["validation_progress"]["adapter_validation_status"] = "not_reached"
        raise
    wall_runtime = reduce_max_float(wall_runtime_local, device)
    trainer_runtime = reduce_max_float(float(train_result.metrics.get("train_runtime", wall_runtime)), device)
    runtime_counts = reduce_sum_counts(trainer)
    training_window_memory_stats = gather_memory_stats(device)
    captured_gradient_payload = trainer.gradient_payload_bytes_local
    if captured_gradient_payload is not None:
        captured_gradient_payload = int(reduce_max_float(float(captured_gradient_payload), device))
    if deepspeed_setup is not None:
        gradient_payload = trainable_parameter_count * torch.tensor([], dtype=torch.bfloat16).element_size()
        gradient_payload_source = (
            f"theoretical BF16 bytes for all trainable {finetuning_method} parameters"
        )
    else:
        gradient_payload = captured_gradient_payload
        gradient_payload_source = "captured from the first available trainable gradients"
    distributed_barrier()

    completed_steps = int(trainer.state.global_step)
    measured_tokens_per_second_per_gpu = (
        runtime_counts["executed_non_padding_tokens"] / trainer_runtime / world_size
        if trainer_runtime > 0 and world_size > 0
        else None
    )
    padded_tokens_per_second_per_gpu = (
        runtime_counts["executed_padded_tokens"] / trainer_runtime / world_size
        if trainer_runtime > 0 and world_size > 0
        else None
    )
    supervised_tokens_per_second_per_gpu = (
        runtime_counts["executed_supervised_tokens"] / trainer_runtime / world_size
        if trainer_runtime > 0 and world_size > 0
        else None
    )

    report["throughput"] = {
        **runtime_counts,
        "definition": (
            "sum(non-padding attention_mask tokens from every executed micro-batch across all ranks) "
            "/ max Trainer runtime across ranks / world size"
        ),
        "trainer_runtime_seconds_max_rank": trainer_runtime,
        "measured_non_padding_tokens_per_second_per_gpu": measured_tokens_per_second_per_gpu,
        "measured_padded_tokens_per_second_per_gpu": padded_tokens_per_second_per_gpu,
        "measured_supervised_tokens_per_second_per_gpu": supervised_tokens_per_second_per_gpu,
        "padding_ratio": (
            1.0 - runtime_counts["executed_non_padding_tokens"] / runtime_counts["executed_padded_tokens"]
            if runtime_counts["executed_padded_tokens"]
            else None
        ),
    }
    if training_cfg.get("max_steps") is None:
        a800_compatible_total_tokens = int(
            round(token_stats["non_padding_tokens_one_epoch"] * float(training_cfg.get("num_train_epochs", 1)))
        )
        a800_compatible_tokens_per_second_per_gpu = (
            a800_compatible_total_tokens / trainer_runtime / world_size
            if trainer_runtime > 0 and world_size > 0
            else None
        )
        report["throughput"]["a800_compatible_estimated_total_tokens"] = a800_compatible_total_tokens
        report["throughput"][
            "a800_compatible_estimated_tokens_per_second_per_gpu"
        ] = a800_compatible_tokens_per_second_per_gpu
        report["throughput"]["a800_compatible_definition"] = (
            "tokenizer-counted non-padding tokens for one selected-data epoch * configured epochs "
            "/ Trainer runtime / world size; retained only to reconcile with the historical A800 report"
        )
    report["timing"] = {
        "trainer_runtime_seconds": trainer_runtime,
        "synchronized_wall_runtime_seconds_max_rank": wall_runtime,
        "seconds_per_optimizer_step": trainer_runtime / completed_steps if completed_steps else None,
    }
    report["memory"] = {
        **training_window_memory_stats,
        "scope": "Trainer train window only; final artifact save and validation have not run yet",
    }
    utilization_summary = monitor.summary() if is_main_process() else {}
    report["gpu_utilization"] = utilization_summary
    report["trainer_metrics"] = json_safe(train_result.metrics)
    if deepspeed_setup is not None:
        finalized_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.deepspeed_config
        unresolved_hidden_size_keys = [
            key
            for key in (
                "reduce_bucket_size",
                "stage3_prefetch_bucket_size",
                "stage3_param_persistence_threshold",
            )
            if finalized_deepspeed_config.get("zero_optimization", {}).get(key) == "auto"
        ]
        if unresolved_hidden_size_keys:
            raise RuntimeError(
                f"DeepSpeed left hidden-size-dependent values unresolved: {unresolved_hidden_size_keys}."
            )
        report["deepspeed"]["effective_config"] = json_safe(finalized_deepspeed_config)
    report["gradient_dtype_elements"] = trainer.gradient_dtype_elements_local
    report["captured_gradient_payload_bytes_max_rank"] = captured_gradient_payload
    report["communication"] = build_communication_stats(
        world_size,
        gradient_payload,
        completed_steps,
        strategy=distributed_label,
        payload_source=gradient_payload_source,
        zero_stage=deepspeed_setup["zero_stage"] if deepspeed_setup is not None else None,
    )

    metadata_model.config.use_cache = original_use_cache
    trainer.save_model(str(output_dir))
    trainer.save_state()
    if is_main_process():
        tokenizer.save_pretrained(output_dir)
    distributed_barrier()

    artifact_validation: dict[str, Any] = {"status": "not_run_on_nonzero_rank"}
    artifact_validation_error: BaseException | None = None
    artifact_validation_ok = True
    if is_main_process():
        try:
            if finetuning_method == "lora":
                artifact_validation = validate_saved_adapter(
                    output_dir,
                    expected_tensor_count=(
                        int(lora_cfg["expected_adapter_tensor_count"])
                        if lora_cfg.get("expected_adapter_tensor_count") is not None
                        else None
                    ),
                    expected_key_prefix=lora_cfg.get("expected_adapter_key_prefix"),
                    expected_parameter_count=(
                        int(lora_cfg["expected_trainable_parameters"])
                        if lora_cfg.get("expected_trainable_parameters") is not None
                        else None
                    ),
                )
            else:
                structural_validation = validate_saved_full_checkpoint(
                    output_dir,
                    expected_model_class=expected_model_class,
                    expected_parameter_shapes=parameter_shapes,
                    expected_parameter_count=total_parameter_count,
                )
                reload_validation = validate_full_checkpoint_reload(
                    output_dir,
                    expected_model_class=expected_model_class,
                    expected_parameter_shapes=parameter_shapes,
                    expected_parameter_count=total_parameter_count,
                    device=device,
                )
                artifact_validation = {
                    "status": "passed",
                    "validation_scope": "structural_headers_plus_fresh_instance_reload_and_forward",
                    "structural": structural_validation,
                    "reload": reload_validation,
                }
        except BaseException as exc:
            artifact_validation_error = exc
            artifact_validation_ok = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        artifact_validation_tensor = torch.tensor(int(artifact_validation_ok), dtype=torch.int32, device=device)
        torch.distributed.broadcast(artifact_validation_tensor, src=0)
        artifact_validation_ok = bool(artifact_validation_tensor.item())
    end_to_end_memory_stats = gather_memory_stats(device)
    if finetuning_method == "lora":
        end_to_end_scope = (
            "peak from the reset immediately before Trainer.train through final PEFT adapter "
            "consolidation/save and tensor validation; no fresh base-model reload is performed"
        )
    else:
        end_to_end_scope = (
            "peak from the reset immediately before Trainer.train through final full-model gather/save and "
            "artifact validation, including the rank-zero fresh reload"
        )
    report["memory"] = {
        **training_window_memory_stats,
        "scope": "Trainer train window (primary training-memory metric)",
        "training_window": training_window_memory_stats,
        "end_to_end": {
            **end_to_end_memory_stats,
            "scope": end_to_end_scope,
        },
        "artifact_validation_included_in_end_to_end": True,
    }
    if not artifact_validation_ok:
        if artifact_validation_error is not None:
            raise RuntimeError(
                f"Saved {report['artifact_type']} validation failed: {artifact_validation_error}"
            ) from artifact_validation_error
        raise RuntimeError(
            f"Saved {report['artifact_type']} validation failed on rank zero; see rank-zero failure evidence."
        )

    train_loss = train_result.metrics.get("train_loss")
    if train_loss is not None and not math.isfinite(float(train_loss)):
        raise RuntimeError(f"Non-finite train loss: {train_loss}")
    if completed_steps != expected_optimizer_steps:
        raise RuntimeError(
            f"Training stopped before the expected progress: {completed_steps}/{expected_optimizer_steps} steps."
        )

    validation_progress = {
        "completed_optimizer_steps": completed_steps,
        "expected_optimizer_steps": expected_optimizer_steps,
        "completed_microbatches_global": runtime_counts["executed_microbatches"],
        "trainer_epoch": trainer.state.epoch,
        "artifact_validation_status": artifact_validation.get("status"),
        "artifact_validation": artifact_validation,
        "heldout_eval": "not_configured",
    }
    if finetuning_method == "lora":
        validation_progress["adapter_validation_status"] = artifact_validation.get("status")
        validation_progress["adapter_validation"] = artifact_validation
    report["validation_progress"] = validation_progress

    utilization_valid = True
    if bool(benchmark_cfg.get("require_gpu_utilization", False)):
        minimum_samples = int(benchmark_cfg.get("minimum_gpu_samples_per_device", 2))
        minimum_utilization_mean_pct = float(
            benchmark_cfg.get("minimum_gpu_utilization_mean_pct", 0.1)
        )
        if is_main_process():
            utilization_valid, utilization_reason = validate_gpu_utilization_summary(
                utilization_summary,
                expected_device_count=world_size,
                minimum_samples_per_device=minimum_samples,
                minimum_utilization_mean_pct=minimum_utilization_mean_pct,
            )
            report["gpu_utilization_validation"] = {
                "valid": utilization_valid,
                "reason": utilization_reason,
                "minimum_samples_per_device": minimum_samples,
                "minimum_utilization_mean_pct": minimum_utilization_mean_pct,
                "artifact_preserved_if_invalid": True,
            }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            utilization_tensor = torch.tensor(int(utilization_valid), dtype=torch.int32, device=device)
            torch.distributed.broadcast(utilization_tensor, src=0)
            utilization_valid = bool(utilization_tensor.item())
        if not utilization_valid:
            reason = (
                report.get("gpu_utilization_validation", {}).get("reason")
                if is_main_process()
                else "rank-zero training-window GPU utilization validation failed"
            )
            raise RuntimeError(
                f"Required GPU utilization is unavailable after training: {reason}. "
                f"The validated {report['artifact_type']} was preserved, but this is not a valid completed benchmark."
            )

    report["status"] = "completed"
    report["direct_reason"] = "completed_expected_steps"
    report["oom"] = False
    comparison_note = ""
    comparison_cfg = config.get("comparison") or {}
    if comparison_cfg.get("baseline_tokens_per_second_per_gpu") is not None:
        comparison_note = (
            " This is a cross-framework platform-stack reproduction against the historical A800 result; "
            "the difference must not be attributed to hardware alone."
        )
    elif comparison_cfg.get("classification"):
        comparison_note = (
            f" Comparison scope: {comparison_cfg['classification']}. No historical throughput baseline is claimed."
        )
    if finetuning_method == "lora":
        completion_text = (
            "Direct Transformers + PEFT LoRA SFT completed without OOM; "
            "the adapter-only safetensors artifact passed structural, parameter-count, and finiteness validation."
        )
    else:
        completion_text = (
            "Direct Transformers Full-parameter SFT completed without OOM; the gathered Full checkpoint "
            "passed safetensors completeness checks, a fresh-instance reload, and a finite BF16 forward."
        )
    report["conclusion"] = (
        completion_text
        + " This establishes BW training feasibility and efficiency only, not customer-service model quality."
        + comparison_note
    )
    return report


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output_dir = Path(args.output_dir or config["training"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    config["training"]["output_dir"] = str(output_dir)

    prepare_output_dir_for_ranks(output_dir)
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if env_rank() == 0:
        with (benchmark_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    report: dict[str, Any] | None = None
    try:
        versions = check_versions(config)
        hardware = collect_hardware()
        report = build_base_report(config_path, config, output_dir, versions, hardware)
        report = run(config_path, config, output_dir, report)
    except BaseException as exc:
        direct_reason, oom = classify_exception(exc)
        trace = traceback.format_exc()
        trace_path = benchmark_dir / f"traceback_rank{env_rank()}.log"
        failure_evidence_path = benchmark_dir / f"failure_rank{env_rank()}.json"
        failure_evidence = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rank": env_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": env_world_size(),
            "direct_reason": direct_reason,
            "direct_error": f"{type(exc).__name__}: {exc}",
            "oom": oom,
            "traceback": str(trace_path),
        }
        try:
            atomic_write_text(trace_path, trace)
            atomic_write_text(
                failure_evidence_path,
                json.dumps(json_safe(failure_evidence), ensure_ascii=False, indent=2) + "\n",
            )
        except Exception as evidence_exc:
            print(f"[benchmark] failed to write rank failure evidence: {evidence_exc!r}", file=os.sys.stderr)
        if env_rank() == 0:
            try:
                versions = {
                    "python": os.sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": package_version("peft"),
                }
                hardware = collect_hardware()
                report = report or build_base_report(config_path, config, output_dir, versions, hardware)
                failed_method = report.get("finetuning_method", "unknown")
                report.update(
                    {
                        "status": "failed",
                        "direct_reason": direct_reason,
                        "direct_error": f"{type(exc).__name__}: {exc}",
                        "oom": oom,
                        "traceback": str(trace_path),
                        "failure_evidence": str(failure_evidence_path),
                        "conclusion": (
                            f"The direct {failed_method} run did not complete. Use direct_reason and the traceback "
                            "as the failure evidence; no throughput conclusion is valid for this run."
                        ),
                    }
                )
                write_report(report, benchmark_dir)
            except Exception as report_exc:
                print(f"[benchmark] failed to write aggregate failure report: {report_exc!r}", file=os.sys.stderr)
        print(trace, file=os.sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    if env_rank() == 0 and report is not None:
        write_report(report, benchmark_dir)
        key_metrics = {
            "status": report["status"],
            "output_dir": str(output_dir),
            "trainable_parameters": report["parameters"]["trainable_parameters"],
            "tokens_per_second_per_gpu": report["throughput"][
                "measured_non_padding_tokens_per_second_per_gpu"
            ],
            "seconds_per_optimizer_step": report["timing"]["seconds_per_optimizer_step"],
            "peak_allocated_gib_max": report["memory"]["peak_allocated_gib_max"],
            "gpu_utilization_mean_pct": report["gpu_utilization"].get("utilization_mean_pct"),
            "gpu_utilization_status": report["gpu_utilization"].get("status"),
            "oom": report["oom"],
        }
        print("[benchmark] completed")
        print(json.dumps(key_metrics, ensure_ascii=False, indent=2))
        print(f"[benchmark] JSON: {benchmark_dir / 'benchmark_metrics.json'}")
        print(f"[benchmark] Markdown: {benchmark_dir / 'benchmark_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
