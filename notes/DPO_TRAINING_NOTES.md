# DPO 训练方法笔记

本文结合当前 repo 里的 Qwen3.5-9B DPO LoRA 实验，说明 DPO 的训练目标、核心公式、代码配置映射，以及为什么数据设计会直接影响 loss 和最终生成效果。

## 1. DPO 解决什么问题

SFT 是“给定 prompt，模仿一个标准答案”：

```text
prompt -> target answer
```

DPO 是“给定 prompt，告诉模型哪个回答更好”：

```text
prompt -> chosen answer 更好
prompt -> rejected answer 更差
```

所以 DPO 数据不是普通的 `input/output`，而是 preference pair：

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

当前项目里这个格式在 [dataset_info.json](../frameworks/llama-factory/data/dataset_info.json) 中通过 `ranking: true` 声明：

```json
"posttrain_dpo_sarcastic_chengyu": {
  "file_name": "dpo_sarcastic_chengyu.jsonl",
  "ranking": true,
  "columns": {
    "prompt": "prompt",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

## 2. DPO 和 SFT 的目标差异

### 2.1 SFT 目标

SFT 最大化标准答案 token 的似然：

```text
max log pi_theta(y | x)
```

训练 loss 通常是 token-level cross entropy：

```text
L_SFT = - sum_t log pi_theta(y_t | x, y_<t)
```

它的含义很直接：目标答案里的每个 token，模型都要更容易预测出来。

### 2.2 DPO 目标

DPO 不直接要求模型逐 token 模仿 chosen。它要求模型相对更偏好 chosen：

```text
pi_theta(chosen | prompt) 相对 pi_theta(rejected | prompt) 要更高
```

并且这个偏好是相对于一个 reference model 来看的。reference model 可以理解为“训练前的模型行为基线”。

当前实验里，配置通过 `adapter_name_or_path` 从已经 SFT 过的 LoRA adapter 开始：

```yaml
model_name_or_path: /root/nfs/llm-models/Qwen3.5-9B
adapter_name_or_path: outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages
stage: dpo
finetuning_type: lora
```

也就是说，我们不是从裸 Qwen3.5-9B 直接做 DPO，而是在 SFT 后的行为基础上继续做偏好调整。

## 3. DPO 的核心公式

对一条样本：

```text
x      = prompt
y_w    = chosen / preferred answer
y_l    = rejected / dispreferred answer
pi     = 当前正在训练的 policy model
pi_ref = 冻结的 reference model
beta   = DPO 温度系数
```

先计算当前模型对两个回答的 log probability：

```text
log pi_theta(y_w | x)
log pi_theta(y_l | x)
```

再计算 reference model 对两个回答的 log probability：

```text
log pi_ref(y_w | x)
log pi_ref(y_l | x)
```

DPO 关心的是“当前模型相比 reference model，把 chosen 相对 rejected 抬高了多少”：

```text
margin =
  [log pi_theta(y_w | x) - log pi_theta(y_l | x)]
  -
  [log pi_ref(y_w | x) - log pi_ref(y_l | x)]
```

然后用 sigmoid loss：

```text
L_DPO = - log sigmoid(beta * margin)
```

你当前配置里：

```yaml
pref_beta: 0.1
pref_loss: sigmoid
```

对应的就是上面公式里的：

```text
beta = 0.1
loss = -log sigmoid(...)
```

## 4. 怎么理解 DPO loss

如果当前 policy 和 reference 对 chosen/rejected 的相对偏好差不多：

```text
margin ≈ 0
sigmoid(0) = 0.5
L_DPO = -log(0.5) ≈ 0.693
```

所以 DPO loss 不是 SFT loss，不能直接和 SFT 的 `0.05`、`0.07` 比。

但它仍然有方向意义：

```text
loss < 0.693  -> policy 已经比 reference 更偏向 chosen
loss ≈ 0.693  -> policy 和 reference 差不多
loss > 0.693  -> policy 还没有把 chosen 抬起来
loss 很大     -> policy 相比 reference 更偏向 rejected，或者 chosen 优化难度很高
```

你之前 128 条 overfit sanity check 里出现：

```json
"train_loss": 9.1611
```

这个很高。按近似：

```text
当 z = beta * margin 很负时：
L ≈ -z
```

所以 `loss ≈ 9.16` 意味着：

```text
beta * margin ≈ -9.16
margin ≈ -91.6    # beta = 0.1
```

这说明当前 policy 在这批 pair 上并没有把 chosen 相对 rejected 抬起来，反而处在明显不利的位置。

## 5. 为什么上一版 DPO 数据很难学

上一版数据的问题不是 LoRA 参数量不够，而是 preference pair 设计不合理。

旧设计大概是：

```text
chosen  : 很长、明显阴阳怪气、整体回复被重写
rejected: 原始正常客服回复
```

这会造成几个问题。

第一，chosen 更长。序列 log probability 通常是 assistant tokens 的 logprob 求和：

```text
log pi(y | x) = sum_t log pi(y_t | x, y_<t)
```

每个 token 的 logprob 通常是负数。回答越长，整体 logprob 越容易更低。

第二，chosen 风格和 SFT 后模型的默认行为冲突很大。SFT 已经把模型训练成“正常客服 JSON 回复”，DPO 突然要求它偏好“更长、更怪、更阴阳”的回答，优化难度会变大。

第三，DPO 不是 SFT。它不会强制模型照抄 chosen，只是调高 chosen 相对 rejected 的偏好。如果 pair 差异太大，它可能只学到一部分，甚至学不动。

## 6. 当前改造后的 DPO 数据

现在生成脚本是 [build_customer_intent_dpo_sarcastic_chengyu.py](../scripts/build_customer_intent_dpo_sarcastic_chengyu.py)。

核心逻辑是：

```python
chosen.reply = f"我先帮您核实，{style_phrase}。{action}"
rejected.reply = f"我先帮您核实。{action}"
```

也就是：

```json
{
  "chosen": {
    "intent": "手机号查件",
    "slots": {
      "phone": "15394359235",
      "waybill_no": null
    },
    "reply": "我先帮您核实，别让问题一波三折。这个手机号我会按隐私校验流程查件，能不能展示结果还得看系统权限"
  },
  "rejected": {
    "intent": "手机号查件",
    "slots": {
      "phone": "15394359235",
      "waybill_no": null
    },
    "reply": "我先帮您核实。这个手机号我会按隐私校验流程查件，能不能展示结果还得看系统权限"
  }
}
```

这版 pair 的特点：

- `intent` 一样
- `slots` 一样
- 业务动作 `action` 一样
- JSON schema 一样
- chosen 只多一个短风格片段
- chosen/rejected 长度差很小

这样 DPO 学到的目标更清楚：

```text
在不破坏 JSON 结构和业务语义的情况下，更偏好带成语风格片段的 reply。
```

## 7. 当前 LLaMA-Factory DPO 配置怎么对应公式

配置文件：

- [local_qwen3_5_9b_lora_dpo_sarcastic_chengyu.yaml](../frameworks/llama-factory/configs/local_qwen3_5_9b_lora_dpo_sarcastic_chengyu.yaml)
- [local_qwen3_5_9b_lora_dpo_sarcastic_chengyu_strong_overfit_128.yaml](../frameworks/llama-factory/configs/local_qwen3_5_9b_lora_dpo_sarcastic_chengyu_strong_overfit_128.yaml)

关键字段：

```yaml
stage: dpo
```

表示训练阶段走 DPO，而不是 SFT。

```yaml
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
```

表示 policy model 只训练 LoRA adapter，不更新全量模型参数。

```yaml
pref_beta: 0.1
pref_loss: sigmoid
```

对应 DPO loss：

```text
L_DPO = -log sigmoid(0.1 * margin)
```

```yaml
dataset: posttrain_dpo_sarcastic_chengyu
```

对应 `dataset_info.json` 里的 ranking dataset，读取：

```text
prompt / chosen / rejected
```

```yaml
adapter_name_or_path: outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages
```

表示 DPO 从已有 SFT LoRA checkpoint 继续训练。这个设计是合理的，因为 DPO 通常不是用来从零学任务，而是在已经会做任务的模型上调偏好。

## 8. 为什么先跑 128 条 overfit sanity check

如果完整 10k DPO 效果不好，有两类可能：

```text
1. 数据设计不好
2. DPO 链路有问题
```

128 条 overfit 的作用是把问题缩小。

当前 overfit 配置：

```yaml
dataset: posttrain_dpo_sarcastic_chengyu_strong
max_samples: 128
num_train_epochs: 10.0
logging_steps: 1
```

它的判断标准不是最终生产效果，而是：

```text
模型能不能在很小的数据上明显学到 fixed style phrase。
```

如果这都学不到，优先怀疑：

- DPO 数据没有被正确读取
- adapter 没有按预期加载
- reference model 设置和预期不一致
- 测试 prompt 和训练 prompt 格式不一致
- DPO loss / beta / mask 逻辑不符合预期

如果 128 条能学到，再去跑完整 10k 才有意义。

## 9. 测试时要注意 prompt 格式

当前 DPO 数据里的 prompt 是：

```text
系统指令

用户输入：具体用户问题
```

所以测试时最好也使用类似格式：

```text
你是物流客服意图识别与回复助手。请根据用户输入判断意图、抽取手机号和运单号，并生成客服回复。只输出合法 JSON，不要输出 Markdown 或额外解释。JSON schema 必须为：{"intent":"...","slots":{"phone":null或字符串,"waybill_no":null或字符串},"reply":"..."}

用户输入：收件人15394359235
```

如果测试时只输入：

```text
收件人15394359235
```

那和训练时的 prompt 分布不同，DPO 学到的风格偏好可能更弱。

## 10. DPO 和 LoRA 的关系

DPO 是训练目标，LoRA 是参数更新方式。

两者不是互斥关系：

```text
DPO + Full  = 用 DPO loss 更新全量参数
DPO + LoRA  = 用 DPO loss 只更新 LoRA adapter
```

当前实验是：

```text
Qwen3.5-9B + SFT LoRA checkpoint + DPO LoRA
```

LoRA 参数量约 43M，对 128 条 fixed style sanity check 来说已经足够。如果这种小样本都学不动，通常不是 LoRA 容量不够，而是数据、配置或测试链路问题。

## 11. Benchmark 脚本记录了什么

[run_llamafactory_benchmark.py](../scripts/run_llamafactory_benchmark.py) 实际调用的是：

```python
command = ["llamafactory-cli", "train", str(config_path)]
```

所以训练本身仍然由 LLaMA-Factory 执行，benchmark 脚本主要做三件事：

1. 调用 LLaMA-Factory 训练
2. 用 `nvidia-smi` 采样显存和 GPU 信息
3. 训练后写出 `benchmark_metrics.json/md`

tokens/s/GPU 的计算逻辑是：

```text
tokens/s/GPU =
  total_train_tokens_estimated
  / train_runtime_seconds
  / gpu_count
```

对应代码：

```python
tokens_per_second_total = float(total_tokens) / float(runtime)
tokens_per_second_per_gpu = tokens_per_second_total / int(gpu_count)
```

DPO 的 `train_loss` 来自 LLaMA-Factory / Transformers trainer 输出，不是 benchmark 脚本自己算的。

## 12. 下一步实验建议

建议顺序：

1. 先跑 `posttrain_dpo_sarcastic_chengyu_strong` 的 128 条 overfit。
2. 看 `train_loss` 是否明显低于上一版 9.16，并观察 chat 是否出现 `别让问题一波三折`。
3. 如果 overfit 成功，再跑完整 `posttrain_dpo_sarcastic_chengyu` 10k/3ep。
4. 如果 overfit 失败，不要继续扩大数据或 epoch，先排查 DPO 链路。

判断优先级：

```text
小样本 overfit 成功 -> 数据和链路基本可用，再考虑规模化
小样本 overfit 失败 -> 优先查配置/加载/reference/prompt，不要盲目加 epoch
```

