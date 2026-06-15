# 80GB GPU Post-Training Lab

This repository is a Docker-first scaffold for post-training experiments on 8-GPU and 16-GPU NVIDIA servers.

Current target machines:

- Node A: 8 GPUs, 80GB VRAM each, driver `550.54.14`, `nvidia-smi` CUDA `12.4`.
- Node B: 8 GPUs, 80GB VRAM each, driver `535.86.10`, `nvidia-smi` CUDA `12.2`.

This is much better than the original V100 plan. Use one machine first to finish the full workflow, then use two machines for larger full-parameter SFT, DPO, GRPO, or long-context experiments.

For the agreed framework and validation sequence, see `EXPERIMENT_ROADMAP.md`.

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

## Environment Choice

The two nodes do not expose the same maximum CUDA runtime through the driver. Pick one of these paths:

### Recommended: Upgrade Node B Driver

Upgrade Node B to driver `550.x` or newer, then use the default CUDA 12.4 / PyTorch cu124 images on both nodes.

```bash
docker compose build train serve
```

### Fallback: Use CUDA 12.1 Runtime On Both Nodes

If Node B cannot be upgraded, use the same CUDA 12.1 / PyTorch cu121 image on both nodes. Do not mix cu124 and cu121 across nodes for distributed training.

```bash
cp .env.example .env
```

Then uncomment the cu121 block in `.env`:

```bash
TRAIN_CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
SERVE_CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
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

1. Start with `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen3-8B` to verify data, training, checkpoint, merge, and deployment.
2. Move to `Qwen/Qwen3-14B` or `Qwen/Qwen3-32B` for a more realistic dense-model run.
3. Use `Qwen/Qwen3-30B-A3B` for MoE LoRA/QLoRA and DPO experiments.
4. Use `Qwen/Qwen3.6-35B-A3B` after the text-only pipeline is stable. It is a newer multimodal MoE model, so multimodal fine-tuning needs extra processor/data-collator work beyond the text-only scripts here.

On 8x80GB, `bf16` is usually the right default if the GPU architecture supports it. If the GPUs are A100/A800/H100/H800/L40S, prefer BF16. If the GPUs are actually older cards with 80GB memory and no BF16 support, switch configs back to FP16.

## Experiment 1: LoRA SFT

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

```bash
torchrun --nproc_per_node=8 src/post_training/dpo.py configs/examples/dpo_lora.yaml
```

DPO is memory-heavy because it compares chosen/rejected responses and conceptually needs policy/reference behavior. Prefer LoRA DPO first.

## Experiment 2b: GRPO

```bash
torchrun --nproc_per_node=8 src/post_training/grpo.py configs/examples/grpo_lora.yaml
```

GRPO generates during training. Start with small `num_generations`, short `max_completion_length`, and a smaller model before trying 30B/35B.

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
- Keep LoRA/QLoRA as the default for 30B/35B experiments until the pipeline is stable.
- Full SFT and GRPO should be promoted to 16 GPUs only after single-node runs are stable.
- For Qwen3.6 multimodal fine-tuning, add multimodal dataset loading and processor/collator support first.

## Source Notes

- NVIDIA R550 `550.54.14` release notes list CUDA Toolkit 12.4 support: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-54-14/index.html
- NVIDIA R535 release notes list CUDA Toolkit 12.2 support: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-54-03/index.html
- PyTorch official previous-version table lists cu124 wheels for Torch 2.6.0 and other CUDA wheel variants: https://pytorch.org/get-started/previous-versions/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Qwen3-30B-A3B model card: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
