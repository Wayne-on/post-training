from __future__ import annotations

import argparse
import re
from typing import Any

from peft import prepare_model_for_kbit_training
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
    return {
        "prompt": str(example[data_cfg.get("prompt_field", "prompt")]),
        "answer": str(example.get(data_cfg.get("answer_field", "answer"), "")),
    }


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

    dataset = load_json_or_hf_dataset(cfg["data"])
    dataset = dataset.map(lambda row: normalize_prompt(row, cfg["data"]), remove_columns=dataset.column_names)

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
        reward_funcs=exact_or_contains_reward,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.get("lora", {})),
    )
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
