# LLaMA-Factory Mainline

LLaMA-Factory is the preferred framework for first-pass SFT/DPO/PPO-style experiments in this repo.

Use it for:

- fast Qwen SFT/LoRA/QLoRA smoke tests;
- Qwen3 30B-A3B LoRA SFT and DPO on A800/A100;
- export/merge of LoRA adapters;
- WebUI or CLI-driven iteration before moving to custom HF/TRL OPD.

Keep the custom HF/TRL scripts in `src/post_training/` for algorithm debugging and OPD prototypes.

## Build On The A800 CUDA 12.1 Machine

```bash
cp .env.a800-cu121.example .env
docker compose build llamafactory
docker compose up -d llamafactory
docker exec -it posttrain_lf bash
```

Before `docker compose up`, edit the server-side `.env` if needed:

```env
ROOT_PASS=your_password
PROJECT_DIR=/data2/ysh/post-training
CONTAINER_WORKDIR=/root/post-training
NFS_MOUNT=/data/test-files
```

The container mounts `${PROJECT_DIR}` at `${CONTAINER_WORKDIR}`, `${NFS_MOUNT}` at `/root/nfs`, and `/etc/localtime` read-only for host time synchronization.

Inside the container:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
llamafactory-cli env
```

Expected basics:

```text
Python 3.11.x
torch 2.5.1+cu121
CUDA 12.1
8 GPUs
```

## Data

LLaMA-Factory expects dataset names defined in `frameworks/llama-factory/data/dataset_info.json`.

For SFT prompt/response data:

```bash
cp data/sft.jsonl frameworks/llama-factory/data/sft.jsonl
```

For SFT chat-message data:

```bash
cp data/sft_messages.jsonl frameworks/llama-factory/data/sft_messages.jsonl
```

For DPO data:

```bash
cp data/dpo.jsonl frameworks/llama-factory/data/dpo.jsonl
```

Expected formats:

```json
{"prompt": "question", "response": "answer"}
```

```json
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
```

```json
{"prompt": "question", "chosen": "better answer", "rejected": "worse answer"}
```

## Commands

Small 7B SFT smoke test:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen2_5_7b_lora_sft.yaml
```

If the model already exists under `/root/nfs/Qwen2.5-7B-Instruct`, use the local-path config:

```bash
llamafactory-cli train frameworks/llama-factory/configs/local_qwen2_5_7b_lora_sft.yaml
```

TensorBoard:

```bash
tensorboard --logdir outputs --host 0.0.0.0 --port 6006
```

Then open port `32006` on the host.

LLaMA-Factory WebUI:

```bash
llamafactory-cli webui --host 0.0.0.0 --port 7860
```

Then open port `32060` on the host.

Qwen3 30B-A3B LoRA SFT:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_sft.yaml
```

If the model already exists under `/root/nfs/Qwen3-30B-A3B`, use:

```bash
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_30b_a3b_lora_sft.yaml
```

Merge/export the Qwen3 30B-A3B SFT adapter:

```bash
llamafactory-cli export frameworks/llama-factory/configs/qwen3_30b_a3b_lora_export.yaml
```

For the local-path adapter:

```bash
llamafactory-cli export frameworks/llama-factory/configs/local_qwen3_30b_a3b_lora_export.yaml
```

DPO from the SFT adapter:

```bash
llamafactory-cli train frameworks/llama-factory/configs/qwen3_30b_a3b_lora_dpo.yaml
```

For the local-path adapter:

```bash
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_30b_a3b_lora_dpo.yaml
```

## Notes

- On A800/A100, use BF16 configs.
- On V100, do not use the 30B-A3B configs. Use this repo's V100 7B HF/TRL configs instead.
- Official LLaMA-Factory `latest` Docker images are useful on Node A CUDA 12.4, but the A800 CUDA 12.1 machine needs this repo's custom `llamafactory` service.
- LLaMA-Factory supports PPO/DPO/KTO/ORPO/SimPO, but not all GRPO workflows are best handled here. Keep verl as the later framework for rollout-heavy GRPO/PPO.
