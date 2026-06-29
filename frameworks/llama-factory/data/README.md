# Local LLaMA-Factory Datasets

Put local JSONL datasets in this directory.

Expected files:

- `sft.jsonl`
- `sft_messages.jsonl`
- `sft_messages_8k.jsonl`
- `dpo.jsonl`
- `dpo_sarcastic_chengyu.jsonl`
- `dpo_sarcastic_chengyu_strong.jsonl`

The dataset names are defined in `dataset_info.json`:

- `posttrain_sft`
- `posttrain_sft_messages`
- `posttrain_sft_messages_8k`
- `posttrain_dpo`
- `posttrain_dpo_sarcastic_chengyu`
- `posttrain_dpo_sarcastic_chengyu_strong`

Generate the toy DPO preference dataset that intentionally treats sarcastic idiom-bearing JSON replies as `chosen`
and the original normal SFT replies as `rejected`:

```bash
python scripts/build_customer_intent_dpo_sarcastic_chengyu.py
```

This dataset is intended for DPO behavior validation only. It trains the model toward the sarcastic `chosen`
style and should not be used as a production customer-service preference dataset.

For the stronger overfit sanity-check variant, keep the chosen reply prefix fixed so the preference signal is easier
to observe:

```bash
python scripts/build_customer_intent_dpo_sarcastic_chengyu.py \
  --variant fixed_strong \
  --example-output examples/datasets/customer_intent_dpo_sarcastic_chengyu_strong.jsonl \
  --llamafactory-output frameworks/llama-factory/data/dpo_sarcastic_chengyu_strong.jsonl
```

Generate the synthetic 8K multi-turn customer-service benchmark dataset with the tokenizer from the local Qwen3.5
model:

```bash
python scripts/build_customer_intent_multiturn_8k.py \
  --model-name-or-path /root/nfs/llm-models/Qwen3.5-9B \
  --output frameworks/llama-factory/data/sft_messages_8k.jsonl \
  --samples 10000 \
  --target-tokens 8064 \
  --min-tokens 8000 \
  --max-tokens 8192
```

The generated JSONL and its `.stats.json` report are local artifacts and are intentionally excluded from git.
