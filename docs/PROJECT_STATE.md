# Post-Training Project State

Last updated: 2026-07-16

This document is the portable handoff for a fresh Codex session or a new machine. It records what has actually been completed, what was only observed on a training server, which conclusions are considered stable, and what remains open.

## 1. Current Objective

The repository started as a post-training laboratory for NVIDIA 8-GPU and 16-GPU servers. The completed mainline covers Qwen3.5 SFT efficiency, long-context training, FlashAttention-2, DPO, and a custom TRL GRPO prototype.

The active branch is `hygon-kw1000`. It now contains the completed direct Transformers/PEFT hardware-adaptation path used on an 8-card BW1000/DCU server, while preserving the validated NVIDIA/A800 LLaMA-Factory baseline. The same branch may be reused for later accelerator work, but each new platform must start with environment discovery and a smoke test rather than assuming the BW integration is portable unchanged.

At the point where the original handoff was introduced:

- `hygon-kw1000`, `main`, `origin/main`, and `origin/hygon-kw1000` all pointed to `6d91146` (`Refine GRPO staged reward scoring`).
- The branch did not yet contain KW1000-specific implementation changes. That statement is historical: the branch now includes `src/post_training/sft_peft.py` and the tracked `configs/hygon/` smoke/formal configurations.
- Always re-check the branch and commit after cloning because this snapshot will naturally become historical.

## 2. Sources Of Truth

| Topic | Canonical tracked file |
| --- | --- |
| Cross-machine Codex guidance | `AGENTS.md` |
| Current project state | `docs/PROJECT_STATE.md` |
| Setup and runnable commands | `README.md` |
| Intended framework progression | `EXPERIMENT_ROADMAP.md` |
| A800 SFT benchmark report | `LLM_SFT_EFFICIENCY_EVALUATION.md` |
| DPO concepts and experiment interpretation | `notes/DPO_TRAINING_NOTES.md` |
| Custom GRPO implementation | `src/post_training/grpo.py` |
| Current customer-intent GRPO config | `configs/examples/grpo_customer_intent_lora.yaml` |
| LLaMA-Factory dataset registry | `frameworks/llama-factory/data/dataset_info.json` |

The following are deliberately ignored by Git and will not appear after a clone:

- `outputs/`: checkpoints, adapters, trainer state, benchmark JSON/Markdown, and runtime logs.
- `models/`: merged or quantized model artifacts.
- `data/` and most files copied into `frameworks/llama-factory/data/`: generated runtime datasets.
- `.env`: machine-specific paths, image names, tokens, and secrets.
- Base model weights under server paths such as `/root/nfs/llm-models/`.

Do not interpret missing ignored artifacts on macOS as data loss. They remain on the original training server unless separately copied.

## 3. Framework And Environment Decisions

### 3.1 NVIDIA/A800 baseline

The authoritative efficiency runs used:

- 8 x NVIDIA A800-SXM4-80GB.
- Driver `530.30.02` and CUDA `12.1` as reported by `nvidia-smi`.
- BF16 training.
- Local base models under `/root/nfs/llm-models/Qwen3.5-4B` and `/root/nfs/llm-models/Qwen3.5-9B`.
- LLaMA-Factory for SFT, full fine-tuning, DPO, chat, and export-oriented workflows.
- TRL `0.24.0` in a separate container for custom GRPO.

Container responsibilities are intentionally separated:

| Container | Responsibility |
| --- | --- |
| `posttrain_lf` | Stable LLaMA-Factory baseline without FA2-specific dependency changes |
| `posttrain_lf_fa2` | Isolated FA2/FLA/Triton stack for long-context comparison |
| `posttrain_trl` | Isolated TRL GRPO and future OPD-style custom work |

The FA2 environment pins:

```text
flash-attn==2.7.4.post1
triton==3.2.0
fla-core==0.4.2
flash-linear-attention==0.4.2
```

It also carries compatibility patches for Qwen3.5 FlashAttention integration, FLA imports, TorchDynamo/TorchInductor interaction, and optional DeepSpeed DeepCompile imports. These patches belong only to the FA2 container.

### 3.2 TRL/Qwen3.5 environment history

The first isolated TRL image exposed several dependency-boundary failures:

- TRL `0.24.0` was available, but an older Transformers build did not recognize `qwen3_5`.
- Reusing the LLaMA-Factory image preserved the Transformers Qwen3.5 implementation.
- Optional TRL import paths pulled in `mergekit`, `immutables`, judge helpers, and Weave-related dependencies even though the experiment did not need those features.
- Local patch scripts made those optional imports non-blocking for the GRPO path.
- TRL expected a model-level `warnings_issued` dictionary; the GRPO loader initializes it when missing.

The resulting container works, but its dependency choices are compatibility decisions rather than general-purpose recommendations.

### 3.3 BW1000/DCU direct-training environment

The completed remote instance reported:

- 8 x BW accelerators, 64.0 GiB per card.
- Python 3.10.18 from `/opt/conda/bin/python` in the platform image.
- Vendor PyTorch `2.4.1+das.opt1.dtk25041`, HIP `6.3.25211`, and eight devices exposed through the CUDA-compatible `torch.cuda` API.
- Vendor DeepSpeed `0.14.2+das.opt1.dtk25041`; distributed NCCL, Gloo, and MPI backends were available.
- BF16 matrix multiplication passed on device.
- `HIP_VISIBLE_DEVICES` correctly restricted visible devices; `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` did not on this image.
- The isolated environment `/root/private_data/venvs/qwen35-bw` used system site packages to preserve vendor PyTorch, then installed Transformers `5.6.0` and PEFT `0.18.0`.
- The base LLaMA-Factory `0.9.3` installation was not used for these runs: its supported Transformers range did not recognize the checkpoint, while Transformers 5.6 removed imports that LLaMA-Factory 0.9.3 expected.
- Qwen3.5/Qwen3.6 hybrid linear-attention layers used the Transformers torch fallback because the optional fast-path libraries were unavailable.

For a new accelerator, capture the same inventory before changing dependencies:

```bash
which python
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
pip list | grep -E "torch|transformers|accelerate|deepspeed|trl|peft|llamafactory"
```

Then determine which distributed backend, visibility variable, model architecture, DeepSpeed behavior, and monitoring tool the new platform exposes. Do not begin by replacing platform PyTorch or installing CUDA-only wheels, FA2, or bitsandbytes.

## 4. Data Assets

### 4.1 SFT

`examples/datasets/customer_intent_sft_smoke.jsonl` contains 10,000 ShareGPT-style records:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The task is logistics customer-service intent recognition, phone/waybill extraction, and reply generation. The output schema is:

```json
{"intent":"...","slots":{"phone":null,"waybill_no":null},"reply":"..."}
```

The tracked file is copied at runtime to the LLaMA-Factory data directory as `sft_messages.jsonl`. The registry name is `posttrain_sft_messages`.

### 4.2 Synthetic 8K SFT

`scripts/build_customer_intent_multiturn_8k.py` builds 10,000 synthetic multi-turn customer-service samples by composing existing material. Effective lengths were approximately 8,000 to 8,192 tokens under the model chat template. The runtime file is `frameworks/llama-factory/data/sft_messages_8k.jsonl`, registered as `posttrain_sft_messages_8k`, and is ignored by Git.

This dataset was designed for long-context feasibility and throughput testing. It is not evidence of production data quality or genuine long-range reasoning ability.

### 4.3 DPO

Tracked DPO datasets:

- `examples/datasets/customer_intent_dpo_sarcastic_chengyu.jsonl`: 10,000 preference pairs.
- `examples/datasets/customer_intent_dpo_sarcastic_chengyu_strong.jsonl`: 10,000 stronger-style pairs; the sanity config limits training to 128 rows.
- Mirrored LLaMA-Factory files live under `frameworks/llama-factory/data/` and are registered with `ranking: true`.

The revised pair design keeps intent, slots, JSON schema, and business action aligned. The chosen response differs from rejected mainly by a short stylistic phrase. This was introduced after the original pairs proved too different in length and behavior for a clean preference experiment.

### 4.4 GRPO

`examples/datasets/customer_intent_grpo_json_reward.jsonl` contains 10,000 rows. The active config limits a run to 1,000 samples. Fields include the prompt, reference answer, expected intent, phone, waybill, and visible style prefix used for method validation.

The visible reply prefix `我先按规则核实，` is an experiment signal, not a proposed production policy.

## 5. Completed SFT Efficiency Matrix

The following full-run metrics are copied from `LLM_SFT_EFFICIENCY_EVALUATION.md`. They use the same 10,000-sample customer-intent dataset, 3 epochs, `cutoff_len=2048`, and estimated global batch size 64.

| Model | Method | DeepSpeed | Runtime | Tokens/s/GPU | Peak VRAM/GPU | Train loss |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | LoRA | ZeRO-2 | 1.21 h | 125.90 | 12.21 GiB | 0.0760 |
| Qwen3.5-4B | Full | ZeRO-3 | 1.79 h | 84.77 | 20.04 GiB | 0.0562 |
| Qwen3.5-9B | LoRA | ZeRO-2 | 1.22 h | 124.41 | 21.38 GiB | 0.0760 |
| Qwen3.5-9B | Full | ZeRO-3 | 1.80 h | 84.50 | 35.38 GiB | 0.0571 |
| Qwen3.5-9B | Full | ZeRO-2 | 1.19 h | 127.17 | 48.34 GiB | 0.0566 |

Stable conclusions:

- LoRA was roughly 47% to 49% faster than the ZeRO-3 full baselines and used about 39% less peak VRAM in this exact matrix.
- This comparison also changes DeepSpeed strategy, so the entire difference cannot be attributed to LoRA alone.
- For Qwen3.5-9B Full at 2K, ZeRO-2 improved throughput but consumed more memory than ZeRO-3.
- These metrics measure training efficiency, not task quality.

The benchmark runner is `scripts/run_llamafactory_benchmark.py`. It calls `llamafactory-cli train`, samples GPU information, counts tokens with the model tokenizer/chat template, and writes:

```text
<output_dir>/benchmark/benchmark_metrics.json
<output_dir>/benchmark/benchmark_metrics.md
```

The full-run throughput definition is:

```text
tokens/s/GPU = tokenizer-counted total train tokens / trainer runtime / GPU count
```

### 5.1 Completed BW1000 direct SFT runs

The BW runs use `src/post_training/sft_peft.py`, direct Transformers 5.6, and
the same tracked 10,000-row dataset. All formal runs used 8 devices, 3 epochs,
BF16, a 2,048-token cap, per-device batch 1, gradient accumulation 8, and global
batch 64. The observed distribution was mean/p50/p95/max
`145.9/144.0/162.0/285`, so the cap does not describe the typical activation
length.

| Model | Method | DeepSpeed | Steps | Tokens/s/GPU | Avg optimizer step | Training peak allocated/reserved | GPU utilization |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| Qwen3.5-4B | PEFT LoRA | ZeRO-2 | 471/471 | 88.47 | 13.13 s | 9.63/12.64 GiB | invalid collection; do not compare |
| Qwen3.5-4B | Full text LM | ZeRO-3 | 471/471 | 59.51 | 19.52 s | 13.59/24.47 GiB | 25.31% |
| Qwen3.6-27B | PEFT text LoRA | ZeRO-3 | 471/471 | 18.69 | 62.13 s | 13.50/26.37 GiB | 42.12% |

All three formal runs completed without OOM. The 4B adapter validation passed;
the 4B Full checkpoint passed safetensors completeness, fresh-instance reload,
and finite BF16 forward checks; the 27B adapter passed structural, exact
parameter-count, dtype, and finiteness checks. The 27B end-to-end save/validation
peak was 13.50 GiB allocated and 32.07 GiB reserved.

The BW throughput is measured with the same executed, non-padding-token formula
implemented by the direct runner. It is still not a hardware-only comparison:
the A800 history used LLaMA-Factory, while BW used direct Transformers/PEFT;
the 27B run also changes model scale and ZeRO stage relative to the 4B LoRA run.
PyTorch/HIP allocator memory and historical `nvidia-smi` memory do not have
identical measurement boundaries.

The direct runner adds platform-oriented integrity gates that were needed in
practice: dataset hash/schema checks, exact model/adapter parameter gates,
ZeRO-stage and batch-math checks, partition-aware ZeRO-3 model loading, finite
loss/gradient checks, executed-token accounting, vendor GPU-monitor validation,
non-overwriting output directories, and adapter/Full-checkpoint validation.

### 5.2 Qwen3.6-27B qualitative adapter check

A deterministic six-prompt base-versus-adapter smoke confirmed that the PEFT
adapter loaded and changed behavior. Both base and adapter produced 6/6 legal
JSON objects and extracted the tested phone/waybill values correctly. The
adapter consistently used the dataset's Chinese intent taxonomy.

This is not held-out quality evidence. One prompt (`除螨喷雾可以寄吗`) was the
training row with punctuation changed, and the adapter reproduced highly
repeated reply templates. It also asked for the item name even though the item
was already present. Treat this as evidence of successful SFT loading and
format/slot learning, alongside a warning about template repetition and
business consistency. A separate, unseen labeled set is still required for
JSON validity, intent accuracy, phone/waybill exact match, identifier
hallucination, reply consistency, and reply-diversity metrics.

## 6. 8K And FlashAttention-2 Findings

### 6.1 Initial feasibility

- Qwen3.5-9B Full + ZeRO-2 + 8K, BS1/GA8, failed during the first backward pass when NCCL ALLREDUCE exceeded 600 seconds for 176,169,472 elements. No CUDA OOM appeared in the log.
- Qwen3.5-9B Full + ZeRO-3 + 8K, BS1/GA8, completed at least 15 optimization steps. Stable step time was about 169 to 171 seconds and observed VRAM was about 56.0 to 58.4 GiB.
- The ZeRO-2 failure was treated as an unstable configuration, not proven solely as a memory-capacity failure.

### 6.2 Micro-batch and FA2 matrix

All entries below use Qwen3.5-9B Full, ZeRO-3, 8K effective length, and global batch size 64.

| Per-device BS / GA | Result | Runtime | Tokens/s/GPU | Peak VRAM | Avg step |
| --- | --- | ---: | ---: | ---: | ---: |
| BS1 / GA8 | At least 15 steps | N/A | about 379-382, stage estimate | about 56.0-58.4 GiB | about 169-171 s |
| BS4 / GA2 | CUDA OOM | N/A | N/A | N/A | N/A |
| BS2 / GA4 | Full 1 epoch | 6.64 h | 423.24 | 75.77 GiB | 152.31 s |
| BS2 / GA4 + FA2 | Full 1 epoch | 1.07 h | 2636.34 | 78.33 GiB | 24.45 s |

The FA2 run improved measured full-run throughput by about 6.23x for this synthetic near-8K distribution. This is an unusually favorable long-sequence case and should not be generalized to shorter or mixed-length production data. Peak memory was higher and left little headroom on an 80GB card.

Server output directories used for the completed BS2/GA4 runs:

```text
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_1ep_8k_zero3_bs2_ga4
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_1ep_8k_zero3_bs2_ga4_fa2
```

## 7. DPO Experiment State

The DPO experiment asked whether preference training could visibly bias customer-service replies toward a short idiomatic/sarcastic style while retaining JSON and business content.

Important observations:

- The original 10,000-row, 1-epoch run reported runtime 1762.19 seconds, 69.36 estimated tokens/s/GPU, peak memory 26,507 MiB, and train loss 1.5507.
- Increasing epochs alone did not produce a clearly visible style change in chat.
- An early 128-row overfit attempt reported train loss 9.1611, indicating a bad pair/optimization setup rather than insufficient LoRA capacity.
- After redesigning pairs to differ only by a short style segment, a 128-row run reported train loss 0.2660, but free generation still did not reliably select the desired phrase.

Interpretation:

- DPO is a relative preference objective, not token-by-token imitation of the chosen response.
- A lower DPO loss can show that chosen is preferred relative to rejected without making that style the top-1 free-generation output.
- Fixed expression or strongly prescribed tone is usually better established by SFT first, then refined by DPO with realistic near-boundary preference pairs.
- No production conclusion should be drawn from the synthetic sarcastic/idiom objective. It was a method and observability experiment.

The current main DPO config is:

```text
frameworks/llama-factory/configs/local_qwen3_5_9b_lora_dpo_sarcastic_chengyu.yaml
```

It starts from the Qwen3.5-9B SFT LoRA adapter, uses LoRA rank 16/alpha 32/dropout 0.05/target all, ZeRO-2, `pref_beta=0.1`, sigmoid loss, 10,000 samples, 3 epochs, and global batch size 64.

## 8. GRPO Experiment State

### 8.1 Current architecture

GRPO runs through:

```text
src/post_training/grpo.py
configs/examples/grpo_customer_intent_lora.yaml
```

Current configuration:

- Base model: `/root/nfs/llm-models/Qwen3.5-9B`.
- Starting adapter: `outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat`.
- LoRA rank 16, alpha 32, dropout 0.05, with 12 explicit target module leaf names.
- 1,000 samples, 2 epochs, per-device batch 1, gradient accumulation 8, 8 GPUs, global batch 64.
- Four generations per prompt, prompt cap 1024, completion cap 256, temperature 0.8.
- BF16, gradient checkpointing, DeepSpeed ZeRO-2.
- Current reward: `customer_service_json_staged`.
- Current output directory: `outputs/grpo/customer-intent-qwen3_5-9b-lora-json-prefix-staged-reward-adapterfix`.

The implementation also:

- Applies the tokenizer chat template with `enable_thinking=False` when supported.
- Suppresses known thinking markers during rollout generation.
- Penalizes forbidden thinking traces in completions.
- Writes a debug prompt/adapter metadata artifact.
- Writes GRPO benchmark JSON and Markdown after training.

GRPO `tokens/s/GPU` is currently an upper-bound estimate because completion tokens use `max_completion_length` and are multiplied by `num_generations`. Do not compare it directly with the measured SFT full-run throughput without labeling the estimate type.

### 8.2 Adapter compatibility fix

The LLaMA-Factory SFT adapter was initially structurally valid but its keys used:

```text
base_model.model.model.language_model.
```

The TRL/Transformers Qwen3.5 model path expected:

```text
base_model.model.model.
```

As a result, earlier GRPO runs could start while most or all SFT LoRA weights were not actually applied. Those runs were closer to base-model GRPO LoRA than true SFT-to-GRPO continuation.

The source adapter must remain untouched. Create a compatible copy:

```bash
python scripts/convert_peft_adapter_key_prefix.py \
  --source outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages \
  --output outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat \
  --force
```

The conversion rewrote all 496 keys. Validate before training:

```bash
python scripts/check_lora_adapter_compat.py \
  configs/examples/grpo_customer_intent_lora.yaml
```

The successful check reported:

- Exact base-model path match.
- YAML and adapter target modules matched.
- 496 adapter weight keys and 248 LoRA modules.
- Zero model modules missing from adapter weights.
- Zero adapter modules missing from the current model.
- Zero PEFT-loaded modules missing or unloaded.

### 8.3 Reward evolution and current result

Reward development went through three broad stages:

1. A loose rule reward gave many candidates the same high score, so within-group `Advantage` often became zero.
2. A strict hard-fail reward improved JSON and no-think behavior but remained too brittle or format-focused.
3. The current staged reward keeps severe penalties for thinking, invalid JSON, schema/type failures, and hallucinated identifiers, then assigns partial credit for allowed/exact intent, phone/waybill correctness, reply prefix, length, similarity, and generic-fallback behavior.

The local staged-reward smoke test produced differentiated scores:

```text
target answer:                 10.0
business-correct, no prefix:    6.5
wrong-intent fallback:          5.54
slot hallucination:             6.4
contains <think>:             -10.0
```

Qualitative chat testing after the adapter and reward fixes showed:

- Legal one-line JSON became stable.
- Explicit thinking output disappeared in the tested LLaMA-Factory chat path.
- The visible prefix `我先按规则核实，` was learned.
- Intent and slot extraction were often correct.
- Business consistency was still weak. A response could extract a provided waybill correctly and still ask the user to provide a waybill.

The current unresolved reward requirement is explicit slot-to-reply consistency. Formatting success is not sufficient.

### 8.4 Model-capability principle

The user explicitly does not want SFT or GRPO to turn the model into a rigid template engine. Preserve this design principle:

- Use prompts for role, output contract, and changeable business policy.
- Use deterministic validators and post-processing for schema enforcement and hard safety constraints.
- Use SFT for stable task behavior, diverse conditional examples, and responses that depend on available/missing slots.
- Use DPO/GRPO for preferences or objectively checkable behavior that cannot be reliably imposed by prompt/validation alone.
- Include enough diversity and off-template cases to preserve useful generalization.

A better future schema may include explicit state such as `missing_slots` and `next_action`, so the reply is supervised from available information rather than from intent alone.

## 9. Testing Adapters

### 9.1 Test the SFT LoRA adapter

```bash
llamafactory-cli chat \
  --model_name_or_path /root/nfs/llm-models/Qwen3.5-9B \
  --adapter_name_or_path outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages \
  --template qwen3_nothink \
  --trust_remote_code true \
  --do_sample false \
  --temperature 0.01 \
  --max_new_tokens 256
```

Type `clear` in the LLaMA-Factory chat CLI to remove conversation history.

### 9.2 Test a TRL GRPO adapter with LLaMA-Factory

TRL-produced adapter keys must be converted back into a separate LLaMA-Factory-compatible copy:

```bash
python scripts/convert_peft_adapter_key_prefix.py \
  --source outputs/grpo/customer-intent-qwen3_5-9b-lora-json-prefix-staged-reward-adapterfix \
  --output outputs/grpo/customer-intent-qwen3_5-9b-lora-json-prefix-staged-reward-adapterfix_lf_compat \
  --from-prefix base_model.model.model. \
  --to-prefix base_model.model.model.language_model. \
  --force
```

Then point `llamafactory-cli chat` at the `_lf_compat` copy. Conversion never modifies the source directory unless source and output are incorrectly made the same path.

## 10. Server-Only Output Paths

The following paths were reported from the A800 training server and are not tracked in Git:

```text
outputs/llamafactory/local-qwen3_5-4b/lora/sft_10k_3ep_messages
outputs/llamafactory/local-qwen3_5-4b/full/sft_10k_3ep
outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_3ep
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_3ep_zero2
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_1ep_8k_zero3_bs2_ga4
outputs/llamafactory/local-qwen3_5-9b/full/sft_10k_1ep_8k_zero3_bs2_ga4_fa2
outputs/llamafactory/local-qwen3_5-9b/lora/dpo_sarcastic_chengyu_minimal_3ep
outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat
outputs/grpo/customer-intent-qwen3_5-9b-lora-json-prefix-staged-reward-adapterfix
outputs/transformers-peft/bw-qwen35-4b/lora/sft_customer_intent_10k_3ep_bs1_ga8_zero2_run2_20260714
outputs/transformers-peft/bw-qwen35-4b/full/sft_customer_intent_10k_3ep_bs1_ga8_zero3
outputs/transformers-peft/bw-qwen36-27b/lora/zero3_10k_3ep_bs1_ga8_20260715_170051
```

If those artifacts are needed on another machine, transfer only the specific checkpoint/adapter and its config/benchmark report. Do not copy the entire output tree blindly.

## 11. Open Work

### Immediate: next accelerator

The BW1000 training phase is complete. For a different accelerator:

1. Start from `hygon-kw1000`, but keep new platform configs in a separate directory rather than modifying the validated BW configs in place.
2. Capture Python, platform PyTorch/runtime, device inventory, visibility variables, distributed backends, and installed training libraries.
3. Verify BF16 allocation/matmul, one-device forward/backward, and a small collective before launching the training runner.
4. Confirm the installed Transformers recognizes the raw checkpoint and that PEFT/DeepSpeed imports do not replace the vendor PyTorch build.
5. Run the smallest matching smoke config, then the distributed smoke, before a formal run.
6. Validate the platform monitor and record both allocator memory and device-tool memory with their measurement boundaries.
7. Use new output directories and record new server-only artifacts here; never overwrite the BW history.

### Remaining roadmap

- Improve SFT data diversity and slot-conditioned replies before treating customer-intent quality as production-ready.
- Add explicit GRPO slot-to-reply consistency rewards and a held-out evaluation set.
- Complete quantization comparison using `src/post_training/quantize_bnb.py` only on supported hardware; bitsandbytes is not assumed on KW1000.
- Validate merged/exported serving and the OpenAI-compatible endpoint.
- Prototype OPD teacher generation and filtering in the isolated TRL environment.
- Move to verl only if rollout-heavy online RL becomes necessary.
- Attempt two-node/16-GPU scale-out only after a single-node path is stable.

## 12. Fresh-Clone Checklist

On macOS or another workstation:

```bash
git clone -b hygon-kw1000 <repository-url>
cd post-training
git status --short --branch
git log --oneline --decorate -5
```

Then ask Codex:

```text
Read AGENTS.md and docs/PROJECT_STATE.md first. Inspect the current branch,
recent Git history, and referenced files. Summarize completed experiments,
server-only artifacts, unresolved GRPO issues, and the KW1000 next step.
Do not modify code until the summary is consistent with the repository.
```

This checked-in state is the cross-machine continuity mechanism. Local Codex Memory may add useful recall, but it is not the authoritative project record.
