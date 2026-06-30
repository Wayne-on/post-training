from __future__ import annotations

import argparse
import json
import re
from typing import Any

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
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
