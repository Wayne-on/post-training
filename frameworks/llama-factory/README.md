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
docker compose up -d --wait llamafactory
docker exec -it posttrain_lf bash
```

The Compose service runs `scripts/bootstrap_llamafactory_container.sh` before entering its idle state. When an older
local image does not contain TensorBoard, the bootstrap installs `tensorboard==2.19.0`; the health check prevents
`docker compose up --wait` from completing until `torch.utils.tensorboard` imports successfully.

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

If you do not have your own SFT data yet, start with the included customer-service intent smoke dataset:

```bash
mkdir -p data frameworks/llama-factory/data
cp examples/datasets/customer_intent_sft_smoke.jsonl data/sft_messages.jsonl
cp data/sft_messages.jsonl frameworks/llama-factory/data/sft_messages.jsonl
```

For SFT chat-message data:

```bash
cp data/sft_messages.jsonl frameworks/llama-factory/data/sft_messages.jsonl
```

For DPO data:

```bash
cp data/dpo.jsonl frameworks/llama-factory/data/dpo.jsonl
```

Preferred SFT format:

```json
{"instruction": "task instruction", "input": "optional user input", "output": "expected answer"}
```

`posttrain_sft` maps this format as:

```text
instruction -> prompt
input -> query
output -> response
```

Legacy prompt/response format is available as dataset name `posttrain_sft_prompt_response`:

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

Preferred Qwen3.5-4B SFT smoke test:

```bash
llamafactory-cli train frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
```

Preferred benchmark wrapper for runs that need efficiency metrics:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
```

For comparable 4B efficiency reporting, run both jobs with the same `sft_messages.jsonl`, `cutoff_len=2048`,
`max_samples=10000`, `num_train_epochs=3`, and the benchmark wrapper:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_full_sft.yaml
```

Qwen3.5-9B LoRA benchmark uses the same comparable settings:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_lora_sft.yaml
```

Qwen3.5-9B full-parameter SFT benchmark uses ZeRO-3 BF16 with the same data and epoch settings:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft.yaml
```

Qwen3.5-9B full-parameter ZeRO-2 comparison keeps the same training settings and writes to a separate output directory:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_zero2.yaml
```

## FlashAttention-2 Comparison

Install FlashAttention-2 inside the existing A800 LLaMA-Factory container:

```bash
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install --force-reinstall --no-deps \
  triton==3.2.0 fla-core==0.4.2 flash-linear-attention==0.4.2
python -c "import flash_attn; print(flash_attn.__version__)"
```

The A800 CUDA 12.1 base image uses PyTorch 2.5.1 and Triton 3.1.0. FLA 0.4.1 can import with Triton 3.1 but its
8K Gated Delta Rule backward kernel may fail to compile in `wy_fast.py`. FLA 0.4.2 requires the Triton 3.2 behavior
for its autotune keys. Use the Triton 3.2 / FLA 0.4.2 combination only in a separate FA2 experiment container because
PyTorch 2.5.1 normally pins Triton 3.1. Keep the baseline container unchanged.

Qwen3.5 also needs two Transformers FlashAttention compatibility guards in affected releases. Apply the idempotent
patch after installing the dependencies:

```bash
python scripts/patch_qwen35_fa2_transformers.py
```

The patch guards an optional `s_aux` tensor and prevents Qwen3.5's 3D position IDs from being misclassified as a
packed sequence. It creates `.qwen35_fa2.bak` backups beside modified Transformers files.

The FA2 configs change only `flash_attn` and `output_dir`; all dataset, batch, epoch, cutoff length, LoRA,
learning-rate, and DeepSpeed settings remain aligned with the baseline configs:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft_fa2.yaml
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_full_sft_fa2.yaml
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_lora_sft_fa2.yaml
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_fa2.yaml
```

Start with the 9B full-parameter config to validate model support and measure the case with the largest attention
memory footprint. The current dataset averages about 146 tokens per sample, so FA2 may provide only a small speedup;
its advantage should become clearer in a separate benchmark with longer effective sequence lengths.

### 8K Multi-Turn Benchmark

Generate 10,000 synthetic multi-turn customer-service conversations whose tokenizer-counted lengths are constrained
to 8,000-8,192 tokens:

```bash
python scripts/build_customer_intent_multiturn_8k.py \
  --model-name-or-path /root/nfs/llm-models/Qwen3.5-9B \
  --output frameworks/llama-factory/data/sft_messages_8k.jsonl \
  --samples 10000 \
  --target-tokens 8064 \
  --min-tokens 8000 \
  --max-tokens 8192
```

Inspect the generated length report before training:

```bash
cat frameworks/llama-factory/data/sft_messages_8k.jsonl.stats.json
```

Run the ZeRO-3 baseline and FA2 comparison:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k.yaml
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k_fa2.yaml
```

Before either full run, validate one forward/backward optimizer step:

```bash
python scripts/run_llamafactory_benchmark.py \
  frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k_smoke.yaml
python scripts/run_llamafactory_benchmark.py \
  frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k_fa2_smoke.yaml
```

For the 9B full-parameter 8K ZeRO-2 experiment, validate the higher-memory setup before the full run:

```bash
python scripts/run_llamafactory_benchmark.py \
  frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k_zero2_smoke.yaml
python scripts/run_llamafactory_benchmark.py \
  frameworks/llama-factory/configs/local_qwen3_5_9b_full_sft_8k_zero2.yaml
```

Both 8K jobs use one epoch. At approximately 80 million tokens per epoch, one epoch is sufficient for a throughput
benchmark and avoids spending three times the compute on repeated data. These synthetic long conversations are for
training-system and long-context efficiency evaluation, not a replacement for production-quality conversational SFT
data.

The local model path is assumed to be:

```text
/root/nfs/llm-models/Qwen3.5-9B
```

The wrapper runs `llamafactory-cli train`, samples GPU memory with `nvidia-smi`, estimates tokenizer-counted
`tokens/s/gpu`, and writes:

```text
outputs/.../benchmark/benchmark_metrics.json
outputs/.../benchmark/benchmark_metrics.md
outputs/.../benchmark/gpu_samples.csv
outputs/.../benchmark/train.log
```

To summarize an already completed run without re-training:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_lora_sft.yaml --no-train
```

Qwen3.5-4B full-parameter SFT, 10k samples, 3 epochs:

```bash
python scripts/run_llamafactory_benchmark.py frameworks/llama-factory/configs/local_qwen3_5_4b_full_sft.yaml
```

Test the Qwen3.5-4B LoRA adapter after SFT:

```bash
llamafactory-cli chat frameworks/llama-factory/configs/local_qwen3_5_4b_lora_chat.yaml
```

This expects the local model directory:

```text
/root/nfs/llm-models/Qwen3.5-4B
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
