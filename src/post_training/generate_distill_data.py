from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from post_training.common import (
    append_jsonl,
    build_quantization_config,
    load_config,
    load_tokenizer,
    read_jsonl,
    to_prompt_text,
    torch_dtype,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    teacher_cfg = cfg["teacher"]
    data_cfg = cfg["data"]
    gen_cfg = cfg["generation"]

    tokenizer = load_tokenizer(teacher_cfg["name_or_path"], bool(teacher_cfg.get("trust_remote_code", True)))
    model = AutoModelForCausalLM.from_pretrained(
        teacher_cfg["name_or_path"],
        trust_remote_code=bool(teacher_cfg.get("trust_remote_code", True)),
        torch_dtype=torch_dtype(teacher_cfg.get("torch_dtype", "float16")),
        attn_implementation=teacher_cfg.get("attn_implementation", "sdpa"),
        quantization_config=build_quantization_config(teacher_cfg),
        device_map="auto",
    )
    model.eval()

    rows = read_jsonl(data_cfg["path"])
    prompt_field = data_cfg.get("prompt_field", "prompt")
    output_path = data_cfg["output_path"]
    batch_size = int(gen_cfg.get("batch_size", 1))

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [str(row[prompt_field]) for row in batch]
        rendered = [to_prompt_text(tokenizer, prompt) for prompt in prompts]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                do_sample=bool(gen_cfg.get("do_sample", True)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        responses = []
        for input_ids, generated_ids in zip(inputs.input_ids, output_ids):
            completion_ids = generated_ids[len(input_ids) :]
            responses.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())

        append_jsonl(
            output_path,
            [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, responses)],
        )
        print(f"wrote {min(start + batch_size, len(rows))}/{len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
