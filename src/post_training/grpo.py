from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from peft import prepare_model_for_kbit_training
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from post_training.common import (
    build_lora_config,
    build_quantization_config,
    load_config,
    load_json_or_hf_dataset,
    load_tokenizer,
    set_seed,
    torch_dtype,
)


def normalize_prompt(example: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, str]:
    normalized = {"prompt": str(example[data_cfg.get("prompt_field", "prompt")])}
    for field in ("answer", "intent", "phone", "waybill_no", "style_prefix"):
        source_field = data_cfg.get(f"{field}_field", field)
        value = example.get(source_field, "")
        normalized[field] = "" if value is None else str(value)
    return normalized


def as_list(value: list[Any] | Any, length: int) -> list[Any]:
    return value if isinstance(value, list) else [value] * length


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(completion).strip()


def parse_strict_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def has_markdown_or_extra_explanation(text: str) -> bool:
    stripped = text.strip()
    if "```" in stripped or stripped.startswith("#") or stripped.startswith("- "):
        return True
    return not (stripped.startswith("{") and stripped.endswith("}"))


def slot_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def has_hallucinated_identifier(obj: dict[str, Any], expected_phone: str, expected_waybill: str) -> bool:
    slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    phone = slot_text(slots.get("phone"))
    waybill = slot_text(slots.get("waybill_no"))
    if not expected_phone and re.fullmatch(r"1[3-9]\d{9}", phone):
        return True
    if expected_phone and phone and phone != expected_phone:
        return True
    if not expected_waybill and re.fullmatch(r"[A-Z]{1,4}\d{8,20}", waybill):
        return True
    if expected_waybill and waybill and waybill != expected_waybill:
        return True
    return False


def customer_service_json_reward(
    completions: list[Any],
    intent: list[str] | str | None = None,
    phone: list[str] | str | None = None,
    waybill_no: list[str] | str | None = None,
    style_prefix: list[str] | str | None = None,
    **_: Any,
) -> list[float]:
    intents = as_list(intent or "", len(completions))
    phones = as_list(phone or "", len(completions))
    waybills = as_list(waybill_no or "", len(completions))
    style_prefixes = as_list(style_prefix or "", len(completions))
    rewards: list[float] = []

    for completion, expected_intent, expected_phone, expected_waybill, expected_prefix in zip(
        completions, intents, phones, waybills, style_prefixes
    ):
        text = completion_to_text(completion)
        obj = parse_strict_json_object(text)
        reward = 0.0

        if obj is None:
            rewards.append(-2.0)
            continue

        reward += 1.0  # legal JSON object
        slots = obj.get("slots")
        has_schema = isinstance(slots, dict) and all(key in obj for key in ("intent", "slots", "reply"))
        if has_schema and "phone" in slots and "waybill_no" in slots:
            reward += 1.0

        actual_intent = slot_text(obj.get("intent"))
        actual_phone = slot_text(slots.get("phone")) if isinstance(slots, dict) else ""
        actual_waybill = slot_text(slots.get("waybill_no")) if isinstance(slots, dict) else ""
        reply = slot_text(obj.get("reply"))

        if expected_intent and actual_intent == expected_intent:
            reward += 1.0
        if actual_phone == slot_text(expected_phone):
            reward += 1.0
        if actual_waybill == slot_text(expected_waybill):
            reward += 1.0
        if reply and not has_markdown_or_extra_explanation(text):
            reward += 1.0
        if expected_prefix and reply.startswith(slot_text(expected_prefix)):
            reward += 2.0
        if has_markdown_or_extra_explanation(text):
            reward -= 1.0
        if has_hallucinated_identifier(obj, slot_text(expected_phone), slot_text(expected_waybill)):
            reward -= 2.0

        rewards.append(reward)
    return rewards


def exact_or_contains_reward(completions: list[str], answer: list[str] | str | None = None, **_: Any) -> list[float]:
    answers = answer if isinstance(answer, list) else [answer] * len(completions)
    rewards: list[float] = []
    for completion, expected in zip(completions, answers):
        if not expected:
            rewards.append(0.0)
            continue
        normalized_completion = re.sub(r"\s+", " ", completion).strip().lower()
        normalized_expected = re.sub(r"\s+", " ", str(expected)).strip().lower()
        rewards.append(1.0 if normalized_expected in normalized_completion else 0.0)
    return rewards


def select_reward_function(name: str):
    if name == "customer_service_json":
        return customer_service_json_reward
    if name == "exact_or_contains":
        return exact_or_contains_reward
    raise ValueError(f"Unsupported reward function: {name}")


def ensure_transformers_warning_state(model: Any) -> None:
    if not isinstance(getattr(model, "warnings_issued", None), dict):
        setattr(model, "warnings_issued", {})


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def count_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        value = parameter.numel()
        total += value
        if parameter.requires_grad:
            trainable += value
    return {
        "model_parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
    }


def percentile_nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_lengths(values: list[int], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"avg_{prefix}_tokens_estimated": None,
            f"min_{prefix}_tokens_estimated": None,
            f"p50_{prefix}_tokens_estimated": None,
            f"p95_{prefix}_tokens_estimated": None,
            f"max_{prefix}_tokens_estimated": None,
            f"total_{prefix}_tokens_estimated": 0,
        }

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: float | int = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2

    return {
        f"avg_{prefix}_tokens_estimated": sum(values) / len(values),
        f"min_{prefix}_tokens_estimated": min(values),
        f"p50_{prefix}_tokens_estimated": median,
        f"p95_{prefix}_tokens_estimated": percentile_nearest_rank(values, 0.95),
        f"max_{prefix}_tokens_estimated": max(values),
        f"total_{prefix}_tokens_estimated": sum(values),
    }


def estimate_prompt_token_stats(dataset: Any, tokenizer: Any, max_prompt_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    for prompt in dataset["prompt"]:
        tokenized = tokenizer(
            str(prompt),
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        lengths.append(len(tokenized.get("input_ids", [])))
    stats = summarize_lengths(lengths, "prompt")
    stats["max_prompt_length"] = max_prompt_length
    return stats


def collect_cuda_peak_stats() -> dict[str, Any]:
    local_stats: dict[str, Any] = {"rank": get_rank()}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        local_stats.update(
            {
                "device": int(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "cuda_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024**2)),
                "cuda_peak_memory_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024**2)),
            }
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_stats)
        per_rank = [item for item in gathered if isinstance(item, dict)]
    else:
        per_rank = [local_stats]

    allocated = [item.get("cuda_peak_memory_allocated_mib") for item in per_rank]
    reserved = [item.get("cuda_peak_memory_reserved_mib") for item in per_rank]
    gpu_names = sorted({str(item["gpu_name"]) for item in per_rank if item.get("gpu_name")})
    return {
        "gpu_names": gpu_names,
        "gpu_count": len(per_rank),
        "cuda_peak_memory_allocated_mib_per_rank": per_rank,
        "cuda_peak_memory_allocated_mib_max": max((int(value) for value in allocated if value is not None), default=None),
        "cuda_peak_memory_reserved_mib_max": max((int(value) for value in reserved if value is not None), default=None),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_benchmark_report(
    cfg: dict[str, Any],
    config_path: str,
    dataset: Any,
    tokenizer: Any,
    parameter_stats: dict[str, int],
    train_metrics: dict[str, Any],
    cuda_stats: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    reward_cfg = cfg.get("reward", {})

    gpu_count = int(cuda_stats.get("gpu_count") or get_world_size())
    epochs = float(training_cfg.get("num_train_epochs", 1))
    per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    global_batch = per_device_batch * grad_accum * gpu_count
    num_generations = int(training_cfg.get("num_generations", 1))
    max_prompt_length = int(training_cfg.get("max_prompt_length", 1024))
    max_completion_length = int(training_cfg.get("max_completion_length", 256))
    sample_count = int(len(dataset))

    prompt_stats = estimate_prompt_token_stats(dataset, tokenizer, max_prompt_length)
    prompt_tokens_per_epoch = int(prompt_stats["total_prompt_tokens_estimated"])
    prompt_tokens_with_generations_per_epoch = prompt_tokens_per_epoch * num_generations
    completion_tokens_upper_bound_per_epoch = sample_count * num_generations * max_completion_length
    total_tokens_upper_bound = int(
        (prompt_tokens_with_generations_per_epoch + completion_tokens_upper_bound_per_epoch) * epochs
    )
    prompt_only_total_tokens = int(prompt_tokens_per_epoch * epochs)

    runtime = train_metrics.get("train_runtime") or train_metrics.get("train_runtime_seconds")
    runtime_float = float(runtime) if runtime else None
    tokens_per_second_total = None
    tokens_per_second_per_gpu = None
    prompt_tokens_per_second_per_gpu = None
    if runtime_float and runtime_float > 0:
        tokens_per_second_total = total_tokens_upper_bound / runtime_float
        tokens_per_second_per_gpu = tokens_per_second_total / gpu_count if gpu_count else None
        prompt_tokens_per_second_per_gpu = prompt_only_total_tokens / runtime_float / gpu_count if gpu_count else None

    total_params = parameter_stats.get("model_parameter_count")
    report = {
        "config_path": config_path,
        "model": model_cfg.get("name_or_path"),
        "adapter_name_or_path": model_cfg.get("adapter_name_or_path"),
        "method": "grpo",
        "finetuning_type": "lora" if cfg.get("lora", {}).get("enabled") or model_cfg.get("adapter_name_or_path") else "full",
        "reward_function": reward_cfg.get("name", "exact_or_contains"),
        "deepspeed_config": training_cfg.get("deepspeed"),
        "model_parameter_count": total_params,
        "model_parameter_count_billions": (total_params / 1_000_000_000) if total_params else None,
        "trainable_parameter_count": parameter_stats.get("trainable_parameter_count"),
        "samples": sample_count,
        "epochs": epochs,
        "gpu_names": cuda_stats.get("gpu_names"),
        "gpu_count": gpu_count,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size_estimated": global_batch,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "temperature": training_cfg.get("temperature"),
        "train_runtime_seconds": runtime_float,
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "global_step": train_metrics.get("global_step"),
        "train_loss": train_metrics.get("train_loss"),
        "tokens_per_second_total_estimated": tokens_per_second_total,
        "tokens_per_second_per_gpu_estimated": tokens_per_second_per_gpu,
        "tokens_per_second_per_gpu_estimate_type": "upper_bound_prompt_plus_max_completion_times_num_generations",
        "prompt_tokens_per_second_per_gpu_estimated": prompt_tokens_per_second_per_gpu,
        "prompt_tokens_per_epoch_estimated": prompt_tokens_per_epoch,
        "prompt_tokens_with_generations_per_epoch_estimated": prompt_tokens_with_generations_per_epoch,
        "completion_tokens_upper_bound_per_epoch_estimated": completion_tokens_upper_bound_per_epoch,
        "total_grpo_tokens_upper_bound_estimated": total_tokens_upper_bound,
        "prompt_only_total_tokens_estimated": prompt_only_total_tokens,
        "data_path": data_cfg.get("path"),
    }
    report.update(prompt_stats)
    report.update(cuda_stats)
    return jsonable(report)


def write_benchmark_report(report: dict[str, Any], output_dir: str | Path) -> None:
    benchmark_dir = Path(output_dir) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    json_path = benchmark_dir / "benchmark_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    rows = [
        ("model", report.get("model")),
        ("adapter_name_or_path", report.get("adapter_name_or_path")),
        ("method", report.get("method")),
        ("finetuning_type", report.get("finetuning_type")),
        ("reward_function", report.get("reward_function")),
        ("deepspeed_config", report.get("deepspeed_config")),
        ("model_parameter_count", report.get("model_parameter_count")),
        ("model_parameter_count_billions", report.get("model_parameter_count_billions")),
        ("trainable_parameter_count", report.get("trainable_parameter_count")),
        ("samples", report.get("samples")),
        ("epochs", report.get("epochs")),
        ("gpu_names", report.get("gpu_names")),
        ("gpu_count", report.get("gpu_count")),
        ("per_device_train_batch_size", report.get("per_device_train_batch_size")),
        ("gradient_accumulation_steps", report.get("gradient_accumulation_steps")),
        ("global_batch_size_estimated", report.get("global_batch_size_estimated")),
        ("num_generations", report.get("num_generations")),
        ("max_prompt_length", report.get("max_prompt_length")),
        ("max_completion_length", report.get("max_completion_length")),
        ("train_runtime_seconds", report.get("train_runtime_seconds")),
        ("train_samples_per_second", report.get("train_samples_per_second")),
        ("train_steps_per_second", report.get("train_steps_per_second")),
        ("tokens_per_second_per_gpu_estimated", report.get("tokens_per_second_per_gpu_estimated")),
        ("tokens_per_second_per_gpu_estimate_type", report.get("tokens_per_second_per_gpu_estimate_type")),
        ("prompt_tokens_per_second_per_gpu_estimated", report.get("prompt_tokens_per_second_per_gpu_estimated")),
        ("avg_prompt_tokens_estimated", report.get("avg_prompt_tokens_estimated")),
        ("p50_prompt_tokens_estimated", report.get("p50_prompt_tokens_estimated")),
        ("p95_prompt_tokens_estimated", report.get("p95_prompt_tokens_estimated")),
        ("total_grpo_tokens_upper_bound_estimated", report.get("total_grpo_tokens_upper_bound_estimated")),
        ("cuda_peak_memory_allocated_mib_max", report.get("cuda_peak_memory_allocated_mib_max")),
        ("cuda_peak_memory_reserved_mib_max", report.get("cuda_peak_memory_reserved_mib_max")),
        ("train_loss", report.get("train_loss")),
    ]

    md_path = benchmark_dir / "benchmark_metrics.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# TRL GRPO Benchmark Metrics\n\n")
        handle.write("| Metric | Value |\n| --- | --- |\n")
        for key, value in rows:
            handle.write(f"| {key} | {value} |\n")
        handle.write("\n")
        handle.write(
            "`tokens_per_second_per_gpu_estimated` is an upper-bound estimate for GRPO: "
            "prompt tokens are tokenizer-counted, completion tokens use "
            "`max_completion_length`, and both are multiplied by `num_generations`, "
            "then divided by trainer runtime and GPU count.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg["training"]
    set_seed(int(training_cfg.get("seed", 42)))

    model_cfg = cfg["model"]
    tokenizer = load_tokenizer(model_cfg["name_or_path"], bool(model_cfg.get("trust_remote_code", True)))
    quantization_config = build_quantization_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=torch_dtype(model_cfg.get("torch_dtype", "float16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    adapter_name_or_path = model_cfg.get("adapter_name_or_path")
    peft_config = build_lora_config(cfg.get("lora", {}))
    if adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=True)
        peft_config = None

    ensure_transformers_warning_state(model)

    dataset = load_json_or_hf_dataset(cfg["data"])
    max_samples = cfg["data"].get("max_samples")
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    dataset = dataset.map(lambda row: normalize_prompt(row, cfg["data"]), remove_columns=dataset.column_names)
    reward_func = select_reward_function(str(cfg.get("reward", {}).get("name", "exact_or_contains")))

    grpo_args = GRPOConfig(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        save_steps=int(training_cfg.get("save_steps", 100)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        fp16=bool(training_cfg.get("fp16", True)),
        bf16=bool(training_cfg.get("bf16", False)),
        deepspeed=training_cfg.get("deepspeed"),
        max_prompt_length=int(training_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(training_cfg.get("max_completion_length", 256)),
        num_generations=int(training_cfg.get("num_generations", 2)),
        temperature=float(training_cfg.get("temperature", 0.7)),
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    parameter_stats = count_parameters(trainer.model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_output = trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])

    cuda_stats = collect_cuda_peak_stats()
    if get_rank() == 0:
        train_metrics = dict(getattr(train_output, "metrics", {}) or {})
        train_metrics.setdefault("global_step", trainer.state.global_step)
        report = build_benchmark_report(
            cfg=cfg,
            config_path=args.config,
            dataset=dataset,
            tokenizer=tokenizer,
            parameter_stats=parameter_stats,
            train_metrics=train_metrics,
            cuda_stats=cuda_stats,
        )
        write_benchmark_report(report, training_cfg["output_dir"])
        print("[benchmark] report written:")
        print(f"  {Path(training_cfg['output_dir']) / 'benchmark' / 'benchmark_metrics.json'}")
        print(f"  {Path(training_cfg['output_dir']) / 'benchmark' / 'benchmark_metrics.md'}")


if __name__ == "__main__":
    main()
