from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM

from post_training.common import build_quantization_config, load_config, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"] | cfg["quantization"]
    tokenizer = load_tokenizer(model_cfg["name_or_path"], bool(model_cfg.get("trust_remote_code", True)))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        quantization_config=build_quantization_config(model_cfg),
        device_map="auto",
    )
    model.save_pretrained(cfg["output_dir"], safe_serialization=True)
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"saved quantized model to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
