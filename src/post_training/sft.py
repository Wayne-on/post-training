from __future__ import annotations

import argparse
from typing import Any

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from post_training.common import (
    build_lora_config,
    build_quantization_config,
    load_config,
    load_json_or_hf_dataset,
    load_tokenizer,
    set_seed,
    to_chat_text,
    torch_dtype,
)


def build_text_pair(example: dict[str, Any], data_cfg: dict[str, Any], tokenizer) -> tuple[str, str]:
    messages_field = data_cfg.get("messages_field", "messages")
    if messages_field in example and example[messages_field]:
        messages = example[messages_field]
        if messages and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
            response = messages[-1].get("content", "")
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt, response
        return to_chat_text(tokenizer, messages), ""

    prompt = str(example.get(data_cfg.get("prompt_field", "prompt"), ""))
    response = str(example.get(data_cfg.get("response_field", "response"), ""))
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt, response


def tokenize_sft(example: dict[str, Any], data_cfg: dict[str, Any], tokenizer) -> dict[str, list[int]]:
    prompt, response = build_text_pair(example, data_cfg, tokenizer)
    train_on_prompt = bool(data_cfg.get("train_on_prompt", False))
    max_seq_length = int(data_cfg.get("max_seq_length", 2048))
    eos = tokenizer.eos_token or ""

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response + eos, add_special_tokens=False).input_ids
    input_ids = (prompt_ids + response_ids)[:max_seq_length]

    if train_on_prompt:
        labels = input_ids.copy()
    else:
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_length]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


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

    lora_config = build_lora_config(cfg.get("lora", {}))
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = load_json_or_hf_dataset(cfg["data"])
    tokenized = dataset.map(
        lambda row: tokenize_sft(row, cfg["data"], tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT dataset",
    )

    train_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_steps=int(training_cfg.get("save_steps", 500)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        fp16=bool(training_cfg.get("fp16", True)),
        bf16=bool(training_cfg.get("bf16", False)),
        optim=training_cfg.get("optim", "adamw_torch"),
        deepspeed=training_cfg.get("deepspeed"),
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
