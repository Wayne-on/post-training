# Post-Training Lab

This repository combines a Docker-first NVIDIA baseline with direct, vendor-
environment training paths for post-training experiments on 8- and 16-device
servers.

For the verified experiment history, server-only artifact boundaries, current
GRPO state, and the Hygon KW1000 handoff, read
[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md). Codex sessions should follow
the repository guidance in [`AGENTS.md`](AGENTS.md) first.

Current target machines:

- BW1000/DCU: 8 x BW, 64GB each, vendor PyTorch 2.4.1/DTK stack; the completed direct-training path is documented below.
- Node A: 8 GPUs, 80GB VRAM each, driver `550.54.14`, `nvidia-smi` CUDA `12.4`.
- Node B: 8 GPUs, 80GB VRAM each, driver `535.86.10`, `nvidia-smi` CUDA `12.2`.
- Temporary Node C: A800 8 GPUs, 80GB VRAM each, driver `530.30.02`, `nvidia-smi` CUDA `12.1`.
- Temporary Node D: V100 8 GPUs. Use it only as a lower-end validation machine.

## BW/DCU Direct Transformers SFT Path

The `hygon-kw1000` branch keeps its platform-specific experiment separate from
the NVIDIA/LLaMA-Factory baseline. Qwen3.5 uses Transformers 5.6 directly
because the platform image's LLaMA-Factory 0.9.3 stack is not compatible with
that Transformers version. LoRA uses PEFT; Full SFT trains the complete model
returned by `AutoModelForCausalLM` and does not wrap it with PEFT.

Inside the isolated BW virtual environment, first install the supported PEFT
version without replacing the platform PyTorch build:

```bash
source /root/private_data/venvs/qwen35-bw/bin/activate
python -m pip install --no-deps "peft==0.18.0"
BW_PYTHON=/root/private_data/venvs/qwen35-bw/bin/python
```

Use `$BW_PYTHON -m torch.distributed.run`, not a bare `torchrun`: the latter can
resolve to `/opt/conda/bin/python` and silently use Transformers 4.51.1.

Run the single-device LoRA smoke test first:

```bash
HIP_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src \
$BW_PYTHON -m post_training.sft_peft \
  configs/hygon/bw_qwen35_4b_lora_smoke.yaml
```

After the single-device smoke report and saved adapter pass validation, validate
the vendor DeepSpeed integration on all 8 devices with 1,024 samples:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen35_4b_lora_zero2_smoke_1024.yaml
```

Only after both smoke reports pass, run the controlled 8-device, ZeRO-2
experiment:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen35_4b_lora_10k_3ep.yaml
```

For Full SFT, run the two-step, 128-sample ZeRO-3 smoke first. It checks that
all parameters in the direct `Qwen3_5ForCausalLM` scope are trainable, completes
two optimizer steps, gathers a Full safetensors checkpoint, and validates its
headers and shards, reloads it into a fresh model instance, and runs a finite
BF16 forward:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen35_4b_full_zero3_smoke_128.yaml
```

Only after the smoke report shows `2/2` steps, `frozen_parameters=0`, valid
eight-card utilization, no OOM, and a passed Full checkpoint structure, start
the controlled 10k x 3 epoch run:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen35_4b_full_10k_3ep.yaml
```

Each run writes `benchmark/benchmark_metrics.json` and
`benchmark/benchmark_metrics.md` under its output directory. The primary
`Tokens/s/GPU` value counts non-padding input tokens from micro-batches that
actually executed, sums them across ranks, and divides by Trainer runtime and
world size. GPU utilization is sampled with `hy-smi` (or `rocm-smi` when
available) during the Trainer train window. These BW configs require a numeric
utilization sample from every selected card. A preflight checks the monitor
before training, and the completed window requires at least two samples per
card and a non-zero training-window mean. Missing, inactive, or incomplete
utilization makes the benchmark fail explicitly; if the final utilization
check fails after training, the validated training artifact is still preserved.
The primary per-card memory metric covers the Trainer window, including normal
in-training checkpoint saves. A separate end-to-end peak covers the final
gather, save, fresh reload, and validation, whose rank-zero-only reload overhead
must not be treated as training memory. For Qwen3.5, the report records both the
self-attention backend and whether hybrid linear-attention layers used the
optional fast path or the Transformers torch fallback.

The 10k configuration controls the A800 run's dataset, three epochs, 2K cap,
BS1/GA8/global batch 64, LoRA hyperparameters, BF16, gradient checkpointing,
learning-rate schedule, and ZeRO-2 JSON. It is a cross-framework hardware
compatibility reproduction: the historical baseline used LLaMA-Factory, while
BW uses direct Transformers + PEFT, so runtime differences must not be
attributed to hardware alone.

The Full configuration controls the same dataset, epoch, sequence cap, batch
math, BF16 precision, learning-rate schedule, and ZeRO-3 stage as the historical
A800 4B Full run. It is still a cross-framework, language-model-scope
platform-stack reproduction rather than a hardware-only benchmark. Its final
checkpoint validation includes structural completeness checks plus a fresh
model-instance reload and finite forward. A separate-process deployment smoke
is still recommended before treating the weights as production-ready.

### Qwen3.6-27B text LoRA on BW

The non-FP8 Qwen3.6-27B experiment uses the text-only
`Qwen3_5ForCausalLM` scope returned by `AutoModelForCausalLM`; it excludes the
checkpoint's vision tower and MTP weights. The runner rejects a quantized
checkpoint and gates the exact text base (`26,895,998,464` parameters), LoRA
scope (`116,727,808` trainable parameters), 496 targeted modules, and 992 saved
adapter tensors.

This model must not be loaded independently by all eight ranks. The runner now
constructs the Transformers DeepSpeed configuration before `from_pretrained`,
resolves the hidden-size-dependent ZeRO-3 bucket values, verifies that every
base parameter was partitioned during loading, and uses logical `ds_numel` and
`ds_shape` metadata for parameter accounting. It also verifies that the vendor
DeepSpeed build supports Transformers 5.6's PEFT-only ZeRO-3 save path before
training starts, so frozen base weights are excluded from adapter artifacts.

Run the eight-device, two-step smoke with a new timestamped output directory:

```bash
SMOKE_OUT="/root/private_data/post-training/outputs/transformers-peft/bw-qwen36-27b/lora/zero3_smoke_128_$(date +%Y%m%d_%H%M%S)"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/root/private_data/post-training/src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen36_27b_lora_zero3_smoke_128.yaml \
  --output-dir "$SMOKE_OUT"
```

Do not start the formal run until the smoke report shows `2/2` optimizer steps,
`zero3_model_load_context_active=true`, complete base-parameter partition
coverage, the exact parameter gates above, valid eight-card utilization, no
OOM, and a passed adapter validation with 992 finite tensors.

Then start the controlled 10k x 3 epoch experiment in a new directory:

```bash
FORMAL_OUT="/root/private_data/post-training/outputs/transformers-peft/bw-qwen36-27b/lora/zero3_10k_3ep_bs1_ga8_$(date +%Y%m%d_%H%M%S)"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/root/private_data/post-training/src \
$BW_PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  --module post_training.sft_peft \
  configs/hygon/bw_qwen36_27b_lora_10k_3ep_zero3.yaml \
  --output-dir "$FORMAL_OUT"
```

The formal run keeps the 10,000-row customer-intent dataset, three epochs,
2,048-token cap, BS1/GA8/global batch 64, LoRA r16/alpha32/dropout 0.05, BF16,
and the existing learning-rate schedule. It intentionally changes both model
scale and distributed strategy relative to the earlier 4B ZeRO-2 LoRA run, so
it has no historical hardware baseline. Qwen3.6 token lengths are measured
again with its own tokenizer/template, and `enable_thinking: false` keeps the
supervised target on the customer-service JSON rather than a thinking suffix.
The formal benchmark intentionally disables in-training checkpoint saves, so
Trainer runtime measures the train window without ZeRO-3 checkpoint
consolidation or I/O; the final adapter save happens after that timing window.
This makes `Tokens/s/GPU` cleaner, but the run cannot be resumed after a
container, process, or node failure. Use `tmux` to protect it from an SSH
disconnect; `tmux` is not fault recovery. On the current BW environment,
hybrid linear-attention layers are expected to use Transformers' supported
torch fallback unless the optional FLA and causal-conv1d fast path is
separately installed and validated.

The completed formal BW results all used 10,000 samples, three epochs,
BS1/GA8/global batch 64, and finished 471/471 optimizer steps without OOM:

| Model | Method | DeepSpeed | Tokens/s/GPU | Step time | Training peak allocated/reserved |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3.5-4B | LoRA | ZeRO-2 | 88.47 | 13.13 s | 9.63/12.64 GiB |
| Qwen3.5-4B | Full text LM | ZeRO-3 | 59.51 | 19.52 s | 13.59/24.47 GiB |
| Qwen3.6-27B | Text LoRA | ZeRO-3 | 18.69 | 62.13 s | 13.50/26.37 GiB |

These runs establish training and artifact feasibility, not held-out customer-
service quality. See `docs/PROJECT_STATE.md` for measurement boundaries,
server-only output paths, the qualitative adapter check, and comparison limits.

For the agreed framework and validation sequence, see `EXPERIMENT_ROADMAP.md`.
LLaMA-Factory remains the preferred NVIDIA baseline; the completed BW path uses
direct Transformers because of the platform compatibility boundary described
above.

## Primary Path: Node A Single-Node First

Use Node A first. It already matches the default setup in this repo:

- driver `550.54.14`
- `nvidia-smi` CUDA `12.4`
- Docker base image `nvidia/cuda:12.4.1-*`
- PyTorch wheel index `cu124`
- 8 GPUs visible to one `torchrun --nproc_per_node=8` job

Do not spend time on two-node training until one-node SFT, DPO/GRPO, quantization, and deployment are working.

Minimal Node A flow:

```bash
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

Then run a small SFT smoke test:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

After that, move to Qwen3 30B-A3B:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora_qwen3_30b_a3b.yaml
```

## Temporary Path: A800 CUDA 12.1

If Node A is busy, the temporary A800 machine with driver `530.30.02` and CUDA `12.1` is a good fallback.

Use the cu121 environment file before building:

```bash
cp .env.a800-cu121.example .env
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

Expected key lines:

```text
torch: 2.5.1+cu121
torch cuda: 12.1
gpu count: 8
```

Then run the same single-node smoke test:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

Preferred LLaMA-Factory smoke test on A800:

```bash
docker compose build llamafactory
docker compose up -d llamafactory
docker exec -it posttrain_lf bash
llamafactory-cli env
cp data/sft_messages.jsonl frameworks/llama-factory/data/sft_messages.jsonl
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
```

### Isolated TRL environment for GRPO / OPD prototyping

Keep `posttrain_lf` as the stable LLaMA-Factory SFT/DPO baseline. Use the
separate TRL container for GRPO and later OPD-style custom experiments:

```bash
docker images | grep 'post-training.*llamafactory'
docker compose --profile trl build trl
docker compose --profile trl up -d trl
docker logs posttrain_trl
docker exec -it posttrain_trl bash
```

The TRL service is isolated behind the `trl` Compose profile, so normal
`docker compose up -d` will not start it. It reuses `LLAMA_FACTORY_IMAGE` from
the selected `.env` as its base image by default, but has its own image and
container name:

```text
TRL_VERSION=0.24.0
TRL_BASE_IMAGE=post-training:llamafactory-cu121
TRL_IMAGE=post-training:trl-cu121
TRL_CONTAINER_NAME=posttrain_trl
```

On the A800 CUDA 12.1 server, an existing `.env` with
`LLAMA_FACTORY_IMAGE=post-training:llamafactory-cu121` is enough. Only run
`docker compose build llamafactory` first if that image does not already exist
on the host.

Run the current GRPO smoke test inside `posttrain_trl`:

```bash
torchrun --nproc_per_node=8 \
  src/post_training/grpo.py \
  configs/examples/grpo_customer_intent_lora.yaml
```

### Isolated LLaMA-Factory FlashAttention-2 environment

Keep the baseline `posttrain_lf` container unchanged. Build the FA2 image from
the existing LLaMA-Factory baseline image and start it as `posttrain_lf_fa2`:

```bash
docker compose build llamafactory-fa2
docker compose up -d llamafactory-fa2
docker compose ps llamafactory-fa2
docker logs posttrain_lf_fa2
docker exec -it posttrain_lf_fa2 bash
```

If the baseline image does not exist on the host yet, run
`docker compose build llamafactory` once before building `llamafactory-fa2`.
The FA2 service belongs to the `fa2` Compose profile, so a normal
`docker compose up -d` does not start it implicitly.

The FA2 container installs the following isolated dependency set:

```text
flash-attn==2.7.4.post1
triton==3.2.0
fla-core==0.4.2
flash-linear-attention==0.4.2
```

The FA2 service disables `torch.compile`/TorchDynamo by default because FLA
imports optional compiled helpers at module import time, while the FA2 image
pins Triton for FLA compatibility.
It also disables DeepSpeed DeepCompile imports inside the FA2 container; regular
ZeRO-2/ZeRO-3 training is unaffected by this optional DeepCompile path.

Verify the environment inside the FA2 container:

```bash
python scripts/verify_llamafactory_fa2.py
```

Run an FA2 experiment:

```bash
python scripts/run_llamafactory_benchmark.py \
  frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_fa2.yaml
```

The original baseline container remains available:

```bash
docker compose up -d llamafactory
docker exec -it posttrain_lf bash
```

If you do not have your own SFT data yet, use the included customer-service intent smoke dataset:

```bash
mkdir -p data frameworks/llama-factory/data
cp examples/datasets/customer_intent_sft_smoke.jsonl data/sft_messages.jsonl
cp data/sft_messages.jsonl frameworks/llama-factory/data/sft_messages.jsonl
```

The LLaMA-Factory container also mounts:

```text
/data2/ysh/post-training -> /root/post-training
/data/test-files -> /root/nfs
/etc/localtime -> /etc/localtime:ro
```

Set `ROOT_PASS` only in the server-side `.env` if you need that variable:

```bash
ROOT_PASS=your_password
PROJECT_DIR=/data2/ysh/post-training
CONTAINER_WORKDIR=/root/post-training
NFS_MOUNT=/data/test-files
```

Do not commit the real password.

On A800, BF16 should be available, so the Qwen3 30B-A3B BF16 LoRA config is still appropriate:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora_qwen3_30b_a3b.yaml
```

Preferred LLaMA-Factory Qwen3 30B-A3B run:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_sft.yaml
```

## Temporary Path: V100 8-GPU

Use the V100 machine only when A100/A800 is not available. Its role is to keep the workflow moving, not to run the final large-model experiments.

V100 constraints:

- Volta architecture, compute capability `sm_70`.
- FP16 only for practical training; no BF16.
- Do not use FlashAttention-2/3.
- Keep `attn_implementation: sdpa` or `eager`.
- Treat vLLM/SGLang and AWQ/GPTQ kernels as validation targets, not assumptions; many modern wheels target newer architectures.
- 30B/35B SFT/DPO/GRPO is not a good use of this machine.

Use the V100 cu121 environment if the host driver supports CUDA 12.1 or newer:

```bash
cp .env.v100-cu121.example .env
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

Expected key lines:

```text
torch: 2.5.1+cu121
torch cuda: 12.1
gpu count: 8
```

Start with conservative 7B LoRA:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora_v100_7b.yaml
```

If memory is tight, use 7B QLoRA:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_qlora_v100_7b.yaml
```

V100 recommended scope:

| Task | V100 Status |
| --- | --- |
| 7B LoRA SFT | Good |
| 7B QLoRA SFT | Good |
| 7B small DPO | Possible, tune sequence length and batch carefully |
| 7B small GRPO | Possible but slow; keep `num_generations` small |
| OPD teacher generation | Possible for small teacher/student models |
| bitsandbytes 4-bit experiments | Possible |
| Transformers FastAPI baseline serving | Possible |
| full SFT 7B | Possible on 32GB V100, tight on 16GB V100 |
| 14B LoRA | Possible on 32GB V100, expect tuning |
| 14B full SFT | Not recommended |
| 30B/35B LoRA | Not recommended; use A800/A100 |
| 30B/35B full SFT/DPO/GRPO | Do not use V100 |
| Qwen3.6-35B-A3B multimodal | Do not use V100 |
| BF16 / FP8 | Not supported |
| FlashAttention-2/3 | Not supported |

## Environment Choice

The two nodes do not expose the same maximum CUDA runtime through the driver. Pick one of these paths:

### Recommended: Upgrade Node B Driver

Upgrade Node B to driver `550.x` or newer, then use the default CUDA 12.4 / PyTorch cu124 images on both nodes.

```bash
docker compose build train serve
```

### Fallback: Use CUDA 12.1 Runtime On Both Nodes

If Node B cannot be upgraded, or if you are using the temporary A800 CUDA 12.1 machine, use the CUDA 12.1 / PyTorch cu121 image. Do not mix cu124 and cu121 across nodes for distributed training.

```bash
cp .env.example .env
```

Then uncomment the cu121 block in `.env`:

```bash
TRAIN_CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
SERVE_CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
TORCH_VERSION=2.5.1
TORCHVISION_VERSION=0.20.1
TORCHAUDIO_VERSION=2.5.1
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TRAIN_IMAGE=post-training:cu121-train
SERVE_IMAGE=post-training:cu121-serve
```

Build:

```bash
docker compose build train serve
```

## Remote Host Setup

Run these on each Linux GPU server. Use the CUDA image that matches the environment path you choose.

```bash
nvidia-smi
docker --version
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If using the cu121 fallback:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

For V100, first check the driver with `nvidia-smi`. If the driver is too old for CUDA 12.1 containers, use an older CUDA/PyTorch stack instead of forcing this repo's cu121 image.

Start an interactive shell:

```bash
docker compose run --rm --service-ports train
python scripts/check_env.py
```

## Data Formats

SFT JSONL:

```json
{"prompt": "question", "response": "answer"}
```

or:

```json
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
```

DPO JSONL:

```json
{"prompt": "question", "chosen": "better answer", "rejected": "worse answer"}
```

GRPO JSONL:

```json
{"prompt": "question", "answer": "reference answer"}
```

Distillation prompt JSONL:

```json
{"prompt": "question"}
```

## Model Selection

Suggested progression:

1. On V100, only run `Qwen3.5-4B` or `Qwen3-8B` smoke tests.
2. On A800/A100, start with local `Qwen3.5-4B` to verify data, training, checkpoint, merge, and deployment.
3. Move to `Qwen/Qwen3-14B` or `Qwen/Qwen3-32B` for a more realistic dense-model run.
4. Use `Qwen/Qwen3-30B-A3B` for MoE LoRA/QLoRA and DPO experiments.
5. Use `Qwen/Qwen3.6-35B-A3B` after the text-only pipeline is stable. It is a newer multimodal MoE model, so multimodal fine-tuning needs extra processor/data-collator work beyond the text-only scripts here.

On 8x80GB, `bf16` is usually the right default if the GPU architecture supports it. If the GPUs are A100/A800/H100/H800/L40S, prefer BF16. If the GPUs are actually older cards with 80GB memory and no BF16 support, switch configs back to FP16.

## Experiment 1: LoRA SFT

Preferred LLaMA-Factory path:

```bash
docker compose build llamafactory
docker compose up -d llamafactory
docker exec -it posttrain_lf bash
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_sft.yaml
```

HF/TRL debug path:

Small first run:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

Qwen3 30B-A3B run:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora_qwen3_30b_a3b.yaml
```

Merge LoRA into a standalone model:

```bash
python src/post_training/merge_lora.py \
  --base_model Qwen/Qwen3-30B-A3B \
  --adapter outputs/sft-lora-qwen3-30b-a3b \
  --output models/qwen3-30b-a3b-sft-lora-merged
```

## Experiment 1b: Full-Parameter SFT

Full SFT is now realistic for 7B/14B and possible for 30B-class models with ZeRO-3, depending on sequence length and model type.

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_full.yaml
```

For 30B/35B full SFT, expect to tune:

- `max_seq_length`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `deepspeed: configs/deepspeed/zero3_bf16.json` on BF16-capable GPUs, or `configs/deepspeed/zero3.json` on FP16-only GPUs.

## Experiment 2: DPO

Preferred LLaMA-Factory path:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_dpo.yaml
```

HF/TRL debug path:

```bash
torchrun --nproc_per_node=8 src/post_training/dpo.py configs/examples/dpo_lora.yaml
```

DPO is memory-heavy because it compares chosen/rejected responses and conceptually needs policy/reference behavior. Prefer LoRA DPO first.

## Experiment 2b: GRPO

Run GRPO inside the isolated `posttrain_trl` container, not inside the
baseline `posttrain_lf` LLaMA-Factory container.

```bash
torchrun --nproc_per_node=8 src/post_training/grpo.py configs/examples/grpo_lora.yaml
```

GRPO generates during training. Start with small `num_generations`, short `max_completion_length`, and a smaller model before trying 30B/35B.

Customer-intent JSON reward smoke test:

```bash
python scripts/build_customer_intent_grpo_json_reward.py

python scripts/convert_peft_adapter_key_prefix.py \
  --source outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages \
  --output outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat \
  --force

python scripts/check_lora_adapter_compat.py \
  configs/examples/grpo_customer_intent_lora.yaml

torchrun --nproc_per_node=8 \
  src/post_training/grpo.py \
  configs/examples/grpo_customer_intent_lora.yaml
```

The conversion step rewrites the LLaMA-Factory adapter key prefix from
`base_model.model.model.language_model.` to `base_model.model.model.` for the TRL/Transformers Qwen3.5 module layout.

This GRPO smoke test starts from the Qwen3.5-9B SFT LoRA adapter and uses a rule reward that scores valid JSON,
schema fields, slot correctness, intent correctness, no extra Markdown/explanation, no hallucinated phone/waybill,
and a visible `reply` prefix: `我先按规则核实，`. The prefix reward is for method validation, not production policy.

## Experiment 3: OPD / Offline Policy Distillation

This scaffold treats OPD as a two-step offline distillation workflow:

1. teacher generates responses for prompts;
2. student is SFT-trained on the generated prompt/response pairs.

```bash
python src/post_training/generate_distill_data.py configs/examples/distill_generate.yaml
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

Point the SFT config at the generated JSONL before step 2.

## Experiment 4: Local Quantization

```bash
python src/post_training/quantize_bnb.py configs/examples/quantize_bnb.yaml
```

For 80GB cards, also test vLLM/SGLang serving separately after the model is trained. This repo keeps a Transformers service as a conservative baseline.

## Experiment 5: Deployment

The default deployment path is a simple OpenAI-compatible FastAPI service using Transformers.

```bash
MODEL_ID=models/qwen3-30b-a3b-sft-lora-merged docker compose up serve
```

Test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":128}'
```

## Experiment 6: Two-Node / 16-GPU Training

Use one machine first. Move to 16 GPUs only after single-node SFT/DPO/GRPO works.

Requirements:

- same code directory on both nodes;
- same Docker image on both nodes;
- same model/data paths, or shared storage mounted at the same path;
- passwordless network reachability between nodes is not required by `torchrun`, but the chosen `MASTER_ADDR:MASTER_PORT` must be reachable;
- firewall allows the master port, default `29500`;
- NCCL can use the correct network interface.
- the `train` service uses host networking in `docker-compose.yml` so the two containers can reach each other across machines.

On both nodes, start an interactive training container:

```bash
docker compose run --rm --service-ports train
```

Find the master node IP. Example:

```bash
hostname -I
```

Run a distributed preflight check.

On Node A:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh scripts/distributed_check.py
```

On Node B:

```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh scripts/distributed_check.py
```

Then run training. Start Node A and Node B with the same command except `NODE_RANK`.

Node A:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh src/post_training/sft.py configs/examples/sft_lora_qwen3_30b_a3b.yaml
```

Node B:

```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh src/post_training/sft.py configs/examples/sft_lora_qwen3_30b_a3b.yaml
```

If NCCL selects the wrong NIC, set it explicitly:

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

Replace `eth0` with the interface used by the GPU servers. Check with `ip addr`.

## Practical Notes

- Do not mix driver/CUDA/PyTorch stacks between the two nodes.
- Prefer BF16 on A100/A800/H100/H800/L40S.
- Use FP16 on V100. Do not enable BF16.
- Keep LoRA/QLoRA as the default for 30B/35B experiments until the pipeline is stable.
- Do not run 30B/35B main experiments on V100.
- Full SFT and GRPO should be promoted to 16 GPUs only after single-node runs are stable.
- For Qwen3.6 multimodal fine-tuning, add multimodal dataset loading and processor/collator support first.

## Source Notes

- NVIDIA R550 `550.54.14` release notes list CUDA Toolkit 12.4 support: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-54-14/index.html
- NVIDIA R535 release notes list CUDA Toolkit 12.2 support: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-54-03/index.html
- NVIDIA CUDA 12.1 release notes list Linux driver `530.30.02` for CUDA 12.1 GA: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html
- PyTorch official previous-version table lists cu124 wheels for Torch 2.6.0 and other CUDA wheel variants: https://pytorch.org/get-started/previous-versions/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA lists Tesla V100 as compute capability 7.0: https://developer.nvidia.com/cuda/gpus
- FlashAttention-2 documents CUDA support for Ampere/Ada/Hopper GPUs, excluding V100/Volta: https://github.com/Dao-AILab/flash-attention
- LLaMA-Factory official README documents Qwen3 support, `llamafactory-cli train/chat/export`, and multi-node examples: https://github.com/hiyouga/LLaMA-Factory
- LLaMA-Factory examples README documents `FORCE_TORCHRUN`, DPO, full tuning, LoRA, QLoRA, and export workflows: https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md
- Qwen3-30B-A3B model card: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
