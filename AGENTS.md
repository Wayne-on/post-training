# Post-Training Repository Guidance

## Start Here

- Before changing code or configs, read `docs/PROJECT_STATE.md`.
- Use `README.md` for environment setup and command examples.
- Use `EXPERIMENT_ROADMAP.md` for the intended framework progression.
- Run `git status --short --branch` and `git log --oneline --decorate -5` before assuming the current state.
- Treat tracked files as the portable source of truth. Training outputs, copied model weights, generated runtime datasets, `.env`, and server caches are intentionally not carried by Git.

## Project Boundaries

- LLaMA-Factory is the stable path for Qwen SFT, full fine-tuning, DPO, export, and interactive chat tests.
- Custom GRPO runs through TRL in `src/post_training/grpo.py` and the isolated `posttrain_trl` container.
- Keep the baseline LLaMA-Factory container, the FA2 container, and the TRL container separate. Their dependency sets are intentionally different.
- `hygon-kw1000` is the hardware-adaptation branch for Hygon KW1000/DCU. Do not assume NVIDIA CUDA, NCCL, FlashAttention-2, bitsandbytes, or the existing DeepSpeed setup works there unchanged.
- On KW1000, inspect the platform-provided PyTorch/DTK environment before modifying dependencies or Docker files. Preserve the NVIDIA/A800 baseline unless a change is explicitly scoped to the Hygon branch.

## Experiment Integrity

- Keep comparison variables controlled. When comparing methods, preserve dataset, epoch, effective sequence length, global batch size, GPU count, precision, and distributed strategy unless the changed variable is the subject of the experiment.
- `tokens/s/GPU` must be derived from tokenizer-counted training tokens, trainer runtime, and GPU count. Do not label a per-step or upper-bound estimate as a measured full-run average.
- Benchmark throughput and loss do not establish model quality. Evaluate JSON validity, intent accuracy, phone/waybill extraction, reply quality, and business consistency separately.
- Do not overwrite or delete historical server outputs. Use a new `output_dir` for a materially different experiment.
- Do not commit model checkpoints, adapters, generated runtime data, secrets, or real `.env` files.

## Customer-Intent Training Rules

- The portable SFT dataset uses ShareGPT `messages` records and contains 10,000 samples.
- The required slots are `phone` and `waybill_no`; avoid inventing unsupported identifiers.
- Do not optimize only for a canned reply. Preserve the distinction between model capability and constraints that belong in prompts, validators, business rules, or post-processing.
- A reply must be consistent with extracted slots. If a waybill or phone is already present, the reply must not ask for the same value again.
- For GRPO, inspect within-group reward spread. Valid formatting alone is insufficient when all candidates receive the same reward and `Advantage` collapses to zero.

## Adapter Handoff

- LLaMA-Factory Qwen3.5 LoRA keys use `base_model.model.model.language_model.` while the TRL/Transformers path used here expects `base_model.model.model.`.
- Before GRPO, convert a copy of the SFT adapter with `scripts/convert_peft_adapter_key_prefix.py`; never modify the source adapter in place.
- Validate base model, target modules, checkpoint type, architecture coverage, and PEFT loading with `scripts/check_lora_adapter_compat.py` before training.
- To test a TRL-produced GRPO adapter with LLaMA-Factory, convert a separate copy back to the LLaMA-Factory prefix.

## Verification

- For Python edits, run `python -m py_compile <changed files>` at minimum.
- For configs, verify referenced paths, output directories, batch math, precision, and DeepSpeed stage.
- For dataset changes, verify JSONL parsing, row count, schema, token-length distribution, and representative samples.
- For documentation-only changes, run `git diff --check` and verify all referenced tracked paths exist.
- If the user explicitly asks to push, complete both commit and push and report the branch and commit.

## Current State

- The detailed, cross-machine handoff is `docs/PROJECT_STATE.md`.
- The current GRPO config is `configs/examples/grpo_customer_intent_lora.yaml`.
- The current GRPO reward is `customer_service_json_staged` in `src/post_training/grpo.py`.
- The BW1000 phase has completed Qwen3.5-4B LoRA/Full and Qwen3.6-27B LoRA formal SFT runs through `src/post_training/sft_peft.py`; the exact results and server-only output paths are recorded in `docs/PROJECT_STATE.md`.
- For another accelerator, reuse the direct runner only after fresh environment discovery and the smallest matching smoke config. Do not assume the BW PyTorch/DTK, visibility variables, DeepSpeed integration, or monitoring commands apply unchanged.
