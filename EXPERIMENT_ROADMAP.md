# Post-Training Experiment Roadmap

This roadmap records the agreed validation path for the 8x80GB Node A first, with Node B and multi-node work deferred.

If Node A is busy, use the temporary A800 8x80GB machine with driver `530.30.02` / CUDA `12.1`. That path uses `.env.a800-cu121.example` and PyTorch `2.5.1+cu121`.

If only V100 is available, use it as a workflow validation machine only. That path uses `.env.v100-cu121.example`, FP16, and small 7B LoRA/QLoRA configs.

## Framework Roles

Use different frameworks for different jobs instead of forcing one framework to do everything.

| Area | Primary Choice | Why |
| --- | --- | --- |
| Environment smoke test | This repo, HF/TRL/DeepSpeed | Transparent and easy to debug. |
| Main Qwen SFT/DPO/PPO-style experiments | LLaMA-Factory first | Fastest path to run Qwen LoRA/SFT/DPO/export with CLI and WebUI. |
| Qwen/MoE/multimodal backup path | ms-swift | Keep it available if LLaMA-Factory hits a Qwen3.6/MoE/multimodal edge case. |
| OPD prototype | This repo, HF/TRL | Easier to customize teacher generation, filtering, reward, KL, and distillation logic. |
| Online RL / rollout-heavy training | verl | Better fit for PPO/GRPO-style RL dataflows with vLLM/SGLang/FSDP/Megatron integration. |
| Deployment | vLLM or SGLang, plus this repo's FastAPI baseline | Use the baseline for correctness; use vLLM/SGLang for throughput. |
| Two-node / 16-GPU scale-out | LLaMA-Factory/ms-swift first, verl if doing RL | Do this only after single-node experiments are stable. |

## Stage 0: Node A Environment Validation

Use Node A first:

- driver `550.54.14`
- `nvidia-smi` CUDA `12.4`
- 8 GPUs, 80GB VRAM each
- default repo image: CUDA 12.4 / PyTorch cu124

Temporary alternative:

- A800 8x80GB
- driver `530.30.02`
- `nvidia-smi` CUDA `12.1`
- copy `.env.a800-cu121.example` to `.env`
- build CUDA 12.1 / PyTorch 2.5.1 cu121 images

Lower-end temporary alternative:

- V100 8 GPUs
- Volta / `sm_70`
- use FP16 only, not BF16
- copy `.env.v100-cu121.example` to `.env`
- build CUDA 12.1 / PyTorch 2.5.1 cu121 images if the host driver supports it

Goal: confirm Docker, GPU visibility, CUDA, PyTorch, NCCL basics, and filesystem paths.

Commands:

```bash
# Node A default path:
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

For the temporary A800:

```bash
cp .env.a800-cu121.example .env
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

For the temporary V100:

```bash
cp .env.v100-cu121.example .env
docker compose build train serve
docker compose run --rm --service-ports train
python scripts/check_env.py
```

Exit criteria:

- all 8 GPUs are visible;
- PyTorch reports CUDA available;
- the container can access data, model cache, and output directories.

## Stage 1: HF/TRL Small-Model Smoke Test

Goal: verify the whole training loop before using ms-swift or larger Qwen models.

Use this repo with a small text-only Qwen model:

- local `Qwen3.5-4B`, or
- `Qwen3-8B`

Run:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

On V100, use the conservative config instead:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora_v100_7b.yaml
```

If memory is tight:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_qlora_v100_7b.yaml
```

Then verify. For the LLaMA-Factory path, use the export config in Stage 2. For the HF/TRL debug path, merge the adapter against the same base model used for that run:

```bash
python src/post_training/merge_lora.py \
  --base_model /root/nfs/Qwen3.5-4B \
  --adapter <adapter-output-dir> \
  --output models/sft-lora-merged

MODEL_ID=models/sft-lora-merged docker compose up serve
```

Exit criteria:

- LoRA training starts and saves checkpoints;
- adapter merge works;
- the local OpenAI-compatible service returns a response.

Do not spend time making this repo the main training framework. Its role is visibility and debugability.

V100 exit criteria are lower: confirm that data loading, distributed launch, checkpoint save, LoRA merge, and baseline serving work. Do not expect V100 to validate large-model throughput.

## Stage 2: LLaMA-Factory Mainline

Goal: use LLaMA-Factory as the first main framework for Qwen experiments.

Run these in LLaMA-Factory after Stage 1 passes:

1. LoRA SFT
2. full-parameter SFT
3. DPO
4. PPO-style experiments if needed
5. larger Qwen models such as `Qwen/Qwen3-30B-A3B`
6. later, Qwen3.6 multimodal models if needed

Build and enter the LLaMA-Factory container:

```bash
docker compose build llamafactory
docker compose up -d llamafactory
docker exec -it posttrain_lf bash
llamafactory-cli env
```

Run the first LLaMA-Factory smoke test:

```bash
cp data/sft.jsonl frameworks/llama-factory/data/sft.jsonl
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
```

Run Qwen3 30B-A3B SFT:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_sft.yaml
```

Run Qwen3 30B-A3B DPO after the SFT adapter exists:

```bash
cp data/dpo.jsonl frameworks/llama-factory/data/dpo.jsonl
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_dpo.yaml
```

Recommended progression:

```text
V100: Qwen3.5-4B or Qwen3-8B smoke test only
-> A800/A100: local Qwen3.5-4B
-> A800/A100: Qwen3-14B / Qwen3-32B
-> Qwen3-30B-A3B
-> Qwen3.6-35B-A3B only after text-only flow is stable
```

Exit criteria:

- LLaMA-Factory can run SFT and DPO on A800/Node A;
- generated checkpoints can be exported and served;
- training metrics and eval outputs are reproducible enough to compare runs.

Keep ms-swift as Stage 2B if LLaMA-Factory hits a model-specific gap, especially around Qwen3.6, MoE, or multimodal workflows.

## Stage 3: OPD Prototype With HF/TRL

Goal: keep OPD logic transparent while the algorithm is still changing.

Use this repo for:

1. teacher generation;
2. filtering by rule, reward model, or LLM-as-judge;
3. student SFT;
4. optional KL-style distillation;
5. ablations on filtering, temperature, prompt style, and loss.

Current starter command:

```bash
python src/post_training/generate_distill_data.py configs/examples/distill_generate.yaml
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

Exit criteria:

- OPD data generation is reproducible;
- filtering criteria are logged and inspectable;
- student model improves on the target eval set;
- the implementation is clear enough to port into ms-swift or scale with verl if needed.
- the implementation is clear enough to port into LLaMA-Factory/ms-swift or scale with verl if needed.

## Stage 4: verl For Online RL Or Rollout-Heavy OPD

Use verl only when the workflow becomes RL-heavy.

Move to verl if OPD becomes one of these:

- online rollout from the current policy;
- reward function or reward model in the training loop;
- PPO / GRPO / DAPO style optimization;
- large rollout throughput with vLLM or SGLang;
- actor/reference/reward separation across GPUs or nodes.

Do not start here. verl is powerful, but it adds orchestration complexity that is unnecessary for the first offline OPD prototype.

Do not use V100 as the main verl machine. Use A800/A100 for rollout-heavy RL.

Exit criteria:

- single-node verl GRPO/PPO job works on Node A;
- rollout throughput and reward computation are measurable;
- checkpoints can be evaluated and served outside verl.

## Stage 5: Quantization And Deployment

Use this repo's Transformers/FastAPI service as a correctness baseline.

For actual serving experiments, prefer:

- vLLM for OpenAI-compatible serving and high throughput;
- SGLang for structured generation and agent-style serving;
- bitsandbytes / AWQ / GPTQ depending on model and runtime support.

On V100, use the Transformers/FastAPI baseline first. Treat vLLM/SGLang/AWQ/GPTQ as compatibility checks because many modern kernels are built for newer GPU architectures.

Exit criteria:

- merged or exported model can serve responses;
- latency and throughput are measured;
- quantized model quality is compared against BF16/FP16 baseline.

## Stage 6: Two-Node / 16-GPU Scale-Out

Do this last.

Current blocker: Node B has driver `535.86.10` and `nvidia-smi` CUDA `12.2`, while Node A uses driver `550.54.14` and CUDA `12.4`.

Preferred path:

- upgrade Node B to driver `550.x` or newer;
- use identical CUDA 12.4 / PyTorch cu124 images on both nodes.

Fallback path:

- use CUDA 12.1 / PyTorch cu121 images on both nodes;
- do not mix cu124 and cu121 between nodes.

Before training, run:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh scripts/distributed_check.py
```

and on Node B:

```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> \
  bash scripts/torchrun_multinode.sh scripts/distributed_check.py
```

Exit criteria:

- both nodes use the same image and dependency versions;
- distributed check passes;
- shared data/model/output paths are consistent;
- NCCL uses the correct network interface.

## Final Recommended Sequence

```text
Node A/A800 environment validation, or V100 workflow-only validation
-> HF/TRL small-model smoke test
-> LLaMA-Factory mainline SFT/DPO/export
-> ms-swift only if LLaMA-Factory hits a model-specific gap
-> HF/TRL OPD prototype
-> verl for online RL or rollout-heavy OPD
-> two-node 16-GPU scale-out
```
