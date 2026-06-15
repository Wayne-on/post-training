# Post-Training Experiment Roadmap

This roadmap records the agreed validation path for the 8x80GB Node A first, with Node B and multi-node work deferred.

## Framework Roles

Use different frameworks for different jobs instead of forcing one framework to do everything.

| Area | Primary Choice | Why |
| --- | --- | --- |
| Environment smoke test | This repo, HF/TRL/DeepSpeed | Transparent and easy to debug. |
| Main Qwen SFT/DPO/GRPO experiments | ms-swift | Strong Qwen/Qwen3/MoE/multimodal support. |
| OPD prototype | This repo, HF/TRL | Easier to customize teacher generation, filtering, reward, KL, and distillation logic. |
| Online RL / rollout-heavy training | verl | Better fit for PPO/GRPO-style RL dataflows with vLLM/SGLang/FSDP/Megatron integration. |
| Deployment | vLLM or SGLang, plus this repo's FastAPI baseline | Use the baseline for correctness; use vLLM/SGLang for throughput. |
| Two-node / 16-GPU scale-out | ms-swift first, verl if doing RL | Do this only after single-node experiments are stable. |

## Stage 0: Node A Environment Validation

Use Node A first:

- driver `550.54.14`
- `nvidia-smi` CUDA `12.4`
- 8 GPUs, 80GB VRAM each
- default repo image: CUDA 12.4 / PyTorch cu124

Goal: confirm Docker, GPU visibility, CUDA, PyTorch, NCCL basics, and filesystem paths.

Commands:

```bash
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

- `Qwen/Qwen2.5-7B-Instruct`, or
- `Qwen/Qwen3-8B`

Run:

```bash
torchrun --nproc_per_node=8 src/post_training/sft.py configs/examples/sft_lora.yaml
```

Then verify:

```bash
python src/post_training/merge_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/sft-lora \
  --output models/sft-lora-merged

MODEL_ID=models/sft-lora-merged docker compose up serve
```

Exit criteria:

- LoRA training starts and saves checkpoints;
- adapter merge works;
- the local OpenAI-compatible service returns a response.

Do not spend time making this repo the main training framework. Its role is visibility and debugability.

## Stage 2: ms-swift Mainline

Goal: use ms-swift as the main framework for Qwen experiments.

Run these in ms-swift after Stage 1 passes:

1. LoRA SFT
2. full-parameter SFT
3. DPO
4. GRPO
5. larger Qwen models such as `Qwen/Qwen3-30B-A3B`
6. later, Qwen3.6 multimodal models if needed

Recommended progression:

```text
Qwen2.5-7B or Qwen3-8B
-> Qwen3-14B / Qwen3-32B
-> Qwen3-30B-A3B
-> Qwen3.6-35B-A3B only after text-only flow is stable
```

Exit criteria:

- ms-swift can run SFT and DPO/GRPO on Node A;
- generated checkpoints can be exported and served;
- training metrics and eval outputs are reproducible enough to compare runs.

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

## Stage 4: verl For Online RL Or Rollout-Heavy OPD

Use verl only when the workflow becomes RL-heavy.

Move to verl if OPD becomes one of these:

- online rollout from the current policy;
- reward function or reward model in the training loop;
- PPO / GRPO / DAPO style optimization;
- large rollout throughput with vLLM or SGLang;
- actor/reference/reward separation across GPUs or nodes.

Do not start here. verl is powerful, but it adds orchestration complexity that is unnecessary for the first offline OPD prototype.

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
Node A environment validation
-> HF/TRL small-model smoke test
-> ms-swift mainline SFT/DPO/GRPO
-> HF/TRL OPD prototype
-> verl for online RL or rollout-heavy OPD
-> two-node 16-GPU scale-out
```
