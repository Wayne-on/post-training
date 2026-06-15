from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from post_training.common import (
    build_lora_config,
    build_quantization_config,
    load_config,
    load_json_or_hf_dataset,
    load_tokenizer,
    set_seed,
    torch_dtype,
)


def normalize_row(example: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": str(example[data_cfg.get("prompt_field", "prompt")]),
        "chosen": str(example[data_cfg.get("chosen_field", "chosen")]),
        "rejected": str(example[data_cfg.get("rejected_field", "rejected")]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg["training"]
    set_seed(int(training_cfg.get("seed", 42)))

    model_cfg = cfg["model"]
    base_name = model_cfg.get("base_model_name_or_path", model_cfg["name_or_path"])
    tokenizer = load_tokenizer(base_name, bool(model_cfg.get("trust_remote_code", True)))
    quantization_config = build_quantization_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=torch_dtype(model_cfg.get("torch_dtype", "float16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(cfg.get("lora", {}))
    trainer_peft_config = lora_config
    adapter_name = model_cfg.get("name_or_path")
    adapter_config = Path(adapter_name, "adapter_config.json") if adapter_name else None
    if lora_config is not None and adapter_name and adapter_name != base_name and (
        adapter_name.startswith("/") or adapter_name.startswith(".") or adapter_config.exists()
    ):
        model = PeftModel.from_pretrained(model, adapter_name, is_trainable=True)
        trainer_peft_config = None

    ref_model = None
    if lora_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            torch_dtype=torch_dtype(model_cfg.get("torch_dtype", "float16")),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )

    dataset = load_json_or_hf_dataset(cfg["data"])
    dataset = dataset.map(lambda row: normalize_row(row, cfg["data"]), remove_columns=dataset.column_names)

    dpo_args = DPOConfig(
        output_dir=training_cfg["output_dir"],
        beta=float(training_cfg.get("beta", 0.1)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 5e-6)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_steps=int(training_cfg.get("save_steps", 500)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        fp16=bool(training_cfg.get("fp16", True)),
        bf16=bool(training_cfg.get("bf16", False)),
        optim=training_cfg.get("optim", "adamw_torch"),
        deepspeed=training_cfg.get("deepspeed"),
        report_to=training_cfg.get("report_to", "none"),
        max_prompt_length=int(cfg["data"].get("max_prompt_length", 1024)),
        max_length=int(cfg["data"].get("max_length", 2048)),
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=trainer_peft_config,
    )
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
