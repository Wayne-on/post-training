#!/usr/bin/env python
"""Run a LLaMA-Factory job and write training-efficiency metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config is not a mapping: {path}")
    return data


def as_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_info(config: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    dataset_dir = as_path(str(config.get("dataset_dir", "data")), repo_dir)
    dataset_name = str(config.get("dataset", "")).split(",")[0].strip()
    info_path = dataset_dir / "dataset_info.json"
    if not dataset_name or not info_path.exists():
        return {}

    info = read_json(info_path)
    dataset_info = info.get(dataset_name, {})
    return dataset_info if isinstance(dataset_info, dict) else {}


def get_dataset_file(config: dict[str, Any], repo_dir: Path) -> Path | None:
    dataset_dir = as_path(str(config.get("dataset_dir", "data")), repo_dir)
    dataset_info = get_dataset_info(config, repo_dir)
    file_name = dataset_info.get("file_name")
    if not file_name:
        return None
    return dataset_dir / file_name


def get_dataset_columns(config: dict[str, Any], repo_dir: Path) -> dict[str, str]:
    dataset_info = get_dataset_info(config, repo_dir)
    columns = dataset_info.get("columns", {})
    return {
        "prompt": columns.get("prompt", "instruction"),
        "query": columns.get("query", "input"),
        "response": columns.get("response", "output"),
        "messages": columns.get("messages", "messages"),
    }


def read_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    model_path = str(config["model_name_or_path"])
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fast=True,
    )


def get_token_sequence_length(tokenized: Any) -> int:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids

    shape = getattr(tokenized, "shape", None)
    if shape is not None:
        dimensions = tuple(int(value) for value in shape)
        if not dimensions:
            return 1
        return dimensions[-1]

    if isinstance(tokenized, (list, tuple)):
        if not tokenized:
            return 0
        if isinstance(tokenized[0], (list, tuple)):
            return len(tokenized[0])
        return len(tokenized)

    raise TypeError(f"Unsupported tokenized output type: {type(tokenized).__name__}")


def count_tokens(config: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    dataset_file = get_dataset_file(config, repo_dir)
    if dataset_file is None or not dataset_file.exists():
        return {"error": "dataset file not found", "dataset_file": str(dataset_file)}

    max_samples = config.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    rows = read_jsonl(dataset_file, max_samples_int)
    dataset_info = get_dataset_info(config, repo_dir)
    columns = get_dataset_columns(config, repo_dir)
    tokenizer = load_tokenizer(config)
    cutoff_len = int(config.get("cutoff_len", 2048))

    total_tokens = 0
    max_seq_len = 0
    messages_col = columns["messages"]
    is_messages_dataset = False
    if rows:
        is_messages_dataset = dataset_info.get("formatting") == "sharegpt" or messages_col in rows[0]
    prompt_col = columns["prompt"]
    query_col = columns["query"]
    response_col = columns["response"]

    for row in rows:
        if is_messages_dataset:
            messages = row.get(messages_col, [])
            if not isinstance(messages, list):
                messages = []
            text_fallback = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
        else:
            prompt = str(row.get(prompt_col, "") or "").strip()
            query = str(row.get(query_col, "") or "").strip()
            response = str(row.get(response_col, "") or "").strip()
            user_content = "\n".join(part for part in (prompt, query) if part)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
            text_fallback = "\n".join(part for part in (user_content, response) if part)

        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception:
            tokenized = tokenizer(text_fallback, add_special_tokens=True)

        seq_len = min(get_token_sequence_length(tokenized), cutoff_len)
        total_tokens += seq_len
        max_seq_len = max(max_seq_len, seq_len)

    epochs = float(config.get("num_train_epochs", 1.0))
    return {
        "dataset_file": str(dataset_file),
        "sample_count": len(rows),
        "tokens_per_epoch_estimated": total_tokens,
        "total_train_tokens_estimated": int(total_tokens * epochs),
        "avg_tokens_per_sample_estimated": (total_tokens / len(rows)) if rows else None,
        "max_tokens_per_sample_capped": max_seq_len,
        "dataset_format": "messages" if is_messages_dataset else "prompt_response",
        "token_count_note": "Estimated with the model tokenizer and chat template, capped by cutoff_len.",
    }


def get_visible_gpu_count() -> int | None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible.strip() not in {"", "-1"}:
        return len([item for item in cuda_visible.split(",") if item.strip()])

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        return None


class GpuSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)

    def _run(self) -> None:
        query = "index,name,memory.used,memory.total,utilization.gpu"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            timestamp = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",", maxsplit=4)]
                    if len(parts) != 5:
                        continue
                    index, name, memory_used, memory_total, util = parts
                    self.samples.append(
                        {
                            "timestamp": timestamp,
                            "index": int(index),
                            "name": name,
                            "memory_used_mib": int(memory_used),
                            "memory_total_mib": int(memory_total),
                            "utilization_gpu_pct": int(util),
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "index",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_pct",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}

        peak_by_gpu: dict[str, int] = {}
        total_by_gpu: dict[str, int] = {}
        names: dict[str, str] = {}
        for sample in self.samples:
            idx = str(sample["index"])
            peak_by_gpu[idx] = max(peak_by_gpu.get(idx, 0), sample["memory_used_mib"])
            total_by_gpu[idx] = int(sample["memory_total_mib"])
            names[idx] = str(sample["name"])

        return {
            "gpu_count_sampled": len(peak_by_gpu),
            "gpu_names": sorted(set(names.values())),
            "peak_memory_used_mib_per_gpu": peak_by_gpu,
            "memory_total_mib_per_gpu": total_by_gpu,
            "peak_memory_used_mib_max": max(peak_by_gpu.values()) if peak_by_gpu else None,
        }


def run_and_stream(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return proc.wait()


def read_trainer_metrics(output_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ("all_results.json", "train_results.json"):
        path = output_dir / name
        if path.exists():
            try:
                metrics.update(read_json(path))
            except Exception:
                pass

    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        try:
            state = read_json(state_path)
            metrics["global_step"] = state.get("global_step", metrics.get("global_step"))
            for item in state.get("log_history", []):
                if isinstance(item, dict) and "train_runtime" in item:
                    metrics.update(item)
        except Exception:
            pass
    return metrics


def product(values: list[int] | tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def count_safetensors_parameters(model_dir: Path) -> int | None:
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map", {})
        filenames = sorted({model_dir / name for name in weight_map.values()})
    else:
        filenames = sorted(model_dir.glob("*.safetensors"))

    if not filenames:
        return None

    try:
        from safetensors import safe_open
    except Exception:
        return None

    total = 0
    try:
        for filename in filenames:
            with safe_open(filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    total += product(handle.get_slice(key).get_shape())
    except Exception:
        return None
    return total


def estimate_decoder_parameters_from_config(config: dict[str, Any]) -> int | None:
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            str(config["model_name_or_path"]),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    except Exception:
        return None

    hidden_size = getattr(model_config, "hidden_size", None)
    intermediate_size = getattr(model_config, "intermediate_size", None)
    num_layers = getattr(model_config, "num_hidden_layers", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    if not all([hidden_size, intermediate_size, num_layers, vocab_size, num_heads]):
        return None

    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim", int(hidden_size) // int(num_heads))
    tie_embeddings = bool(getattr(model_config, "tie_word_embeddings", False))

    attention_params = (
        int(hidden_size) * int(hidden_size)
        + 2 * int(hidden_size) * int(num_kv_heads) * int(head_dim)
        + int(hidden_size) * int(hidden_size)
    )
    mlp_params = 3 * int(hidden_size) * int(intermediate_size)
    norm_params = 2 * int(hidden_size)
    layer_params = attention_params + mlp_params + norm_params
    embedding_params = int(vocab_size) * int(hidden_size)
    lm_head_params = 0 if tie_embeddings else int(vocab_size) * int(hidden_size)
    final_norm_params = int(hidden_size)
    return int(embedding_params + int(num_layers) * layer_params + final_norm_params + lm_head_params)


def get_model_parameter_stats(config: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    model_path = Path(str(config["model_name_or_path"]))
    if not model_path.is_absolute():
        model_path = repo_dir / model_path

    total_params = count_safetensors_parameters(model_path)
    source = "safetensors" if total_params is not None else None
    if total_params is None:
        total_params = estimate_decoder_parameters_from_config(config)
        source = "config_estimate" if total_params is not None else None

    if total_params is None:
        return {}

    return {
        "model_parameter_count": total_params,
        "model_parameter_count_billions": total_params / 1_000_000_000,
        "model_parameter_count_source": source,
    }


def read_train_log_stats(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    stats: dict[str, Any] = {}
    match = re.search(r"Number of trainable parameters\s*=\s*([0-9,]+)", text)
    if match:
        stats["trainable_parameter_count"] = int(match.group(1).replace(",", ""))
    return stats


def read_previous_benchmark(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "benchmark" / "benchmark_metrics.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def get_previous_gpu_stats(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "gpu_count_sampled",
        "gpu_names",
        "peak_memory_used_mib_per_gpu",
        "memory_total_mib_per_gpu",
        "peak_memory_used_mib_max",
    )
    stats = {key: report[key] for key in keys if report.get(key) is not None}
    if "gpu_count_sampled" not in stats and report.get("gpu_count") is not None:
        stats["gpu_count_sampled"] = report["gpu_count"]
    return stats


def build_report(
    config_path: Path,
    config: dict[str, Any],
    repo_dir: Path,
    wall_runtime: float | None,
    token_stats: dict[str, Any],
    gpu_stats: dict[str, Any],
    trainer_metrics: dict[str, Any],
    parameter_stats: dict[str, Any],
    command: list[str] | None,
) -> dict[str, Any]:
    gpu_count = gpu_stats.get("gpu_count_sampled") or get_visible_gpu_count()
    runtime = trainer_metrics.get("train_runtime") or wall_runtime
    total_tokens = token_stats.get("total_train_tokens_estimated")
    tokens_per_second_total = None
    tokens_per_second_per_gpu = None
    if runtime and total_tokens:
        tokens_per_second_total = float(total_tokens) / float(runtime)
        if gpu_count:
            tokens_per_second_per_gpu = tokens_per_second_total / int(gpu_count)

    per_device_batch = int(config.get("per_device_train_batch_size", 1))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    global_batch = per_device_batch * grad_accum * int(gpu_count or 1)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "command": command,
        "model_name_or_path": config.get("model_name_or_path"),
        "model_parameter_count": parameter_stats.get("model_parameter_count"),
        "model_parameter_count_billions": parameter_stats.get("model_parameter_count_billions"),
        "model_parameter_count_source": parameter_stats.get("model_parameter_count_source"),
        "trainable_parameter_count": parameter_stats.get("trainable_parameter_count"),
        "stage": config.get("stage"),
        "finetuning_type": config.get("finetuning_type"),
        "dataset": config.get("dataset"),
        "cutoff_len": config.get("cutoff_len"),
        "max_samples": config.get("max_samples"),
        "num_train_epochs": config.get("num_train_epochs"),
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size_estimated": global_batch,
        "learning_rate": config.get("learning_rate"),
        "gpu_count": gpu_count,
        "wall_runtime_seconds": wall_runtime,
        "train_runtime_seconds": runtime,
        "train_samples_per_second": trainer_metrics.get("train_samples_per_second"),
        "train_steps_per_second": trainer_metrics.get("train_steps_per_second"),
        "train_loss": trainer_metrics.get("train_loss"),
        "global_step": trainer_metrics.get("global_step"),
        "tokens_per_second_total_estimated": tokens_per_second_total,
        "tokens_per_second_per_gpu_estimated": tokens_per_second_per_gpu,
    }
    report.update(token_stats)
    report.update(gpu_stats)
    return report


def write_report(report: dict[str, Any], output_dir: Path, run_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    report_json = benchmark_dir / "benchmark_metrics.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    report_md = benchmark_dir / "benchmark_metrics.md"
    rows = [
        ("model", report.get("model_name_or_path")),
        ("model_parameter_count", report.get("model_parameter_count")),
        ("model_parameter_count_billions", report.get("model_parameter_count_billions")),
        ("finetuning_type", report.get("finetuning_type")),
        ("trainable_parameter_count", report.get("trainable_parameter_count")),
        ("samples", report.get("sample_count")),
        ("epochs", report.get("num_train_epochs")),
        ("cutoff_len", report.get("cutoff_len")),
        ("global_batch_size_estimated", report.get("global_batch_size_estimated")),
        ("gpu_names", report.get("gpu_names")),
        ("gpu_count", report.get("gpu_count")),
        ("train_runtime_seconds", report.get("train_runtime_seconds")),
        ("train_samples_per_second", report.get("train_samples_per_second")),
        ("tokens_per_second_per_gpu_estimated", report.get("tokens_per_second_per_gpu_estimated")),
        ("peak_memory_used_mib_max", report.get("peak_memory_used_mib_max")),
        ("train_loss", report.get("train_loss")),
    ]
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# LLaMA-Factory Benchmark Metrics\n\n")
        f.write("| Metric | Value |\n| --- | --- |\n")
        for key, value in rows:
            f.write(f"| {key} | {value} |\n")
        f.write("\n")
        f.write("`tokens_per_second_per_gpu_estimated` is estimated from tokenizer-counted tokens, trainer runtime, and GPU count.\n")

    for name in ("train.log", "gpu_samples.csv"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, benchmark_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a LLaMA-Factory YAML config.")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="GPU sampling interval in seconds.")
    parser.add_argument("--no-train", action="store_true", help="Only summarize an existing output_dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    output_dir = as_path(str(config["output_dir"]), repo_dir)
    run_id = f"{config_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = repo_dir / "outputs" / "_benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] config: {config_path}")
    print(f"[benchmark] output_dir: {output_dir}")
    print("[benchmark] counting dataset tokens...")
    try:
        token_stats = count_tokens(config, repo_dir)
    except Exception as exc:
        token_stats = {"token_count_error": repr(exc)}
        print(f"[benchmark] token count failed: {exc}", file=sys.stderr)

    print("[benchmark] counting model parameters...")
    parameter_stats = get_model_parameter_stats(config, repo_dir)
    previous_report = read_previous_benchmark(output_dir) if args.no_train else {}

    wall_runtime: float | None = None
    gpu_stats: dict[str, Any] = {}
    command: list[str] | None = None

    if not args.no_train:
        command = ["llamafactory-cli", "train", str(config_path)]
        sampler = GpuSampler(args.sample_interval)
        sampler.start()
        start = time.time()
        try:
            exit_code = run_and_stream(command, run_dir / "train.log")
        finally:
            wall_runtime = time.time() - start
            sampler.stop()
            sampler.write_csv(run_dir / "gpu_samples.csv")
            gpu_stats = sampler.summary()
        if exit_code != 0:
            print(f"[benchmark] training failed with exit code {exit_code}", file=sys.stderr)
            return exit_code
    else:
        wall_runtime = previous_report.get("wall_runtime_seconds")
        gpu_stats = get_previous_gpu_stats(previous_report)
        command = previous_report.get("command")

    trainer_metrics = read_trainer_metrics(output_dir)
    if previous_report.get("trainable_parameter_count") is not None:
        parameter_stats["trainable_parameter_count"] = previous_report["trainable_parameter_count"]
    parameter_stats.update(read_train_log_stats(run_dir / "train.log"))
    if not gpu_stats:
        gpu_stats = {"gpu_count_sampled": get_visible_gpu_count()}

    report = build_report(
        config_path=config_path,
        config=config,
        repo_dir=repo_dir,
        wall_runtime=wall_runtime,
        token_stats=token_stats,
        gpu_stats=gpu_stats,
        trainer_metrics=trainer_metrics,
        parameter_stats=parameter_stats,
        command=command,
    )
    write_report(report, output_dir, run_dir)

    print("[benchmark] report written:")
    print(f"  {output_dir / 'benchmark' / 'benchmark_metrics.json'}")
    print(f"  {output_dir / 'benchmark' / 'benchmark_metrics.md'}")
    print("[benchmark] key metrics:")
    print(json.dumps(
        {
            "model_parameter_count": report.get("model_parameter_count"),
            "finetuning_type": report.get("finetuning_type"),
            "trainable_parameter_count": report.get("trainable_parameter_count"),
            "sample_count": report.get("sample_count"),
            "train_runtime_seconds": report.get("train_runtime_seconds"),
            "train_samples_per_second": report.get("train_samples_per_second"),
            "tokens_per_second_per_gpu_estimated": report.get("tokens_per_second_per_gpu_estimated"),
            "peak_memory_used_mib_max": report.get("peak_memory_used_mib_max"),
            "train_loss": report.get("train_loss"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
