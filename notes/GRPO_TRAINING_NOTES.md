# GRPO 训练原理与项目代码笔记

本文结合当前仓库中已经跑通的 Qwen3.5-9B 物流客服 GRPO 实验，说明 GRPO 的训练目标、核心公式、数据流、Reward 设计、LoRA 继承方式和指标解读。

当前项目中的正式 GRPO 实验不是由 LLaMA-Factory 启动，而是使用独立的 TRL 容器，直接执行 [grpo.py](../src/post_training/grpo.py)：

```bash
torchrun --nproc_per_node=8 \
  src/post_training/grpo.py \
  configs/examples/grpo_customer_intent_lora.yaml
```

项目固定使用 `TRL 0.24.0`。不同 TRL 版本的默认参数和实现可能变化，理解代码时必须同时确认版本。

## 1. GRPO 解决什么问题

GRPO 全称是 Group Relative Policy Optimization。它解决的不是“让模型模仿一条标准答案”，而是：

> 对同一个问题生成一组候选回答，根据 Reward 比较组内回答好坏，再提高高分回答的生成概率、降低低分回答的生成概率。

在当前物流客服实验中，希望模型逐步偏向以下行为：

- 只输出合法 JSON；
- JSON 字段和类型严格符合 schema；
- 正确识别 `intent`；
- 正确抽取手机号和运单号；
- 不凭空生成不存在的手机号或运单号；
- `reply` 满足指定格式和业务规则；
- 不输出 `<think>`、Markdown 或额外解释。

GRPO 的关键不是提供唯一的标准输出，而是定义一个能够区分回答质量的 Reward。

## 2. GRPO、SFT 和 DPO 的区别

### 2.1 SFT

SFT 给模型一条明确的目标答案：

```text
输入 x -> 标准答案 y
```

训练目标是最大化标准答案中每个 token 的概率：

```text
L_SFT = -sum_t log pi_theta(y_t | x, y_<t)
```

它适合学习：

- 固定输出格式；
- 基本业务映射；
- 特定回复风格；
- 工具调用格式；
- 任务所需的基础行为。

### 2.2 DPO

DPO 为同一个输入提供一对答案：

```text
(prompt, chosen, rejected)
```

它学习的是“chosen 应该比 rejected 更受偏好”，不需要训练时在线生成候选答案，也不需要显式 Reward 函数。

### 2.3 GRPO

GRPO 在训练过程中由当前模型实时生成多个回答：

```text
prompt
  -> completion_1 -> reward_1
  -> completion_2 -> reward_2
  -> completion_3 -> reward_3
  -> completion_4 -> reward_4
```

然后按同组回答的相对得分进行更新。

| 方法 | 监督信号 | 是否在线生成 | 主要用途 |
| --- | --- | --- | --- |
| SFT | 标准答案 | 否 | 建立任务能力和基本格式 |
| DPO | chosen/rejected 对 | 否 | 学习人工或模型偏好 |
| GRPO | Reward 分数 | 是 | 优化可自动验证的行为 |

对于当前任务，合理顺序是：

```text
Base Model
  -> SFT：先建立物流客服 JSON 能力
  -> GRPO：再用规则 Reward 强化可验证行为
```

GRPO 不应该代替 SFT 从零教授完整业务。它更适合在已有能力上继续优化。

## 3. 当前项目的完整训练链路

当前 GRPO 实验的主要文件如下：

| 作用 | 文件 |
| --- | --- |
| GRPO 训练入口 | [src/post_training/grpo.py](../src/post_training/grpo.py) |
| 训练配置 | [configs/examples/grpo_customer_intent_lora.yaml](../configs/examples/grpo_customer_intent_lora.yaml) |
| GRPO 数据构建 | [scripts/build_customer_intent_grpo_json_reward.py](../scripts/build_customer_intent_grpo_json_reward.py) |
| GRPO 数据集 | [examples/datasets/customer_intent_grpo_json_reward.jsonl](../examples/datasets/customer_intent_grpo_json_reward.jsonl) |
| Adapter key 转换 | [scripts/convert_peft_adapter_key_prefix.py](../scripts/convert_peft_adapter_key_prefix.py) |
| Adapter 兼容性检查 | [scripts/check_lora_adapter_compat.py](../scripts/check_lora_adapter_compat.py) |
| TRL 容器定义 | [docker/Dockerfile.trl](../docker/Dockerfile.trl) |

执行链路可以概括为：

```text
SFT messages 数据
  -> 构建 GRPO prompt 与标准字段
  -> 加载 Qwen3.5-9B base model
  -> 加载已训练的 SFT LoRA adapter
  -> 每个 prompt 在线生成 G 个 completion
  -> 规则 Reward 分别打分
  -> 组内标准化得到 Advantage
  -> 更新同一个 LoRA adapter
  -> 保存 GRPO adapter 和 benchmark 报告
```

这里没有调用 `llamafactory-cli train`。LLaMA-Factory 只参与了前面的 SFT，以及后续使用 LLaMA-Factory chat 测试 adapter。

## 4. GRPO 数据长什么样

当前数据不是 DPO 的 `chosen/rejected`，也不是 SFT 的完整 `messages` 标准答案，而是为在线采样和 Reward 准备的结构：

```json
{
  "prompt": "你是物流客服意图识别与回复助手……用户输入：JT3133424281103 这个件破损空包了",
  "answer": "{\"intent\":\"我要理赔\",...}",
  "intent": "我要理赔",
  "phone": null,
  "waybill_no": "JT3133424281103",
  "style_prefix": "我先按规则核实，"
}
```

这些字段的职责不同：

- `prompt`：送给模型生成回答；
- `answer`：用于比较回复语义，不是像 SFT 一样直接做 teacher forcing；
- `intent`：Reward 判断意图是否正确；
- `phone`：Reward 判断手机号槽位；
- `waybill_no`：Reward 判断运单号槽位；
- `style_prefix`：Reward 判断指定回复前缀。

也就是说，标准字段主要进入 Reward，而不是直接作为模型下一 token 的训练目标。

## 5. 一次 GRPO 更新发生了什么

假设一个 prompt 为 `q`，当前配置 `num_generations = 4`。

### 5.1 在线采样

当前策略模型对同一个 prompt 生成 4 个回答：

```text
o_1 ~ pi_theta(. | q)
o_2 ~ pi_theta(. | q)
o_3 ~ pi_theta(. | q)
o_4 ~ pi_theta(. | q)
```

这里需要一定随机性。当前配置使用：

```yaml
num_generations: 4
temperature: 0.8
```

如果所有回答都完全相同，Reward 也相同，组内就没有可学习的相对差异。

### 5.2 Reward 打分

Reward 函数给每个回答一个绝对分数：

```text
r_i = R(q, o_i, label)
```

例如：

```text
completion_1: 合法 JSON、意图正确、槽位正确 -> 10 分
completion_2: 合法 JSON、意图错误             -> 7 分
completion_3: 合法 JSON、漏掉运单号           -> 5 分
completion_4: 非法 JSON                        -> 0 分
```

### 5.3 组内归一化得到 Advantage

GRPO 不直接把绝对 Reward 当作梯度权重，而是先在同一个 prompt 的组内比较：

```text
mean_r = mean(r_1, r_2, ..., r_G)

A_i = (r_i - mean_r) / (std(r_1, ..., r_G) + epsilon)
```

`A_i` 就是 Advantage：

- `A_i > 0`：该回答高于组内平均，应该提高概率；
- `A_i < 0`：该回答低于组内平均，应该降低概率；
- `A_i = 0`：没有相对优势，基本不提供训练信号。

例如 Reward 为：

```text
[10, 7, 5, 2]
```

组内平均值是 `6`，10 分和 7 分回答得到正 Advantage，5 分和 2 分回答得到负 Advantage。

如果 Reward 为：

```text
[7, 7, 7, 7]
```

那么 4 个回答的 Advantage 都接近 0。即使平均 Reward 看起来不低，这一组也几乎学不到东西。

这就是之前训练界面中大量 `Advantage = 0` 的根本原因：Reward 没有在同组候选之间形成区分度。

## 6. GRPO 的策略优化公式

### 6.1 概率比

策略更新前后，对同一个 completion token 的概率比为：

```text
rho_i,t(theta) =
  pi_theta(o_i,t | q, o_i,<t)
  / pi_old(o_i,t | q, o_i,<t)
```

实际计算时通常使用 log probability：

```text
rho_i,t(theta) = exp(
  log pi_theta(o_i,t | q, o_i,<t)
  - log pi_old(o_i,t | q, o_i,<t)
)
```

其中：

- `pi_theta`：正在训练的当前策略；
- `pi_old`：生成这批 completion 时的旧策略。

### 6.2 Clipped surrogate objective

GRPO 延续 PPO 的裁剪思想：既要鼓励高 Advantage 的回答，又不能让单次参数更新过大。

```text
L_policy = -mean(
  min(
    rho_i,t * A_i,
    clip(rho_i,t, 1-epsilon, 1+epsilon) * A_i
  )
)
```

直观理解：

- 好回答的概率可以增加；
- 坏回答的概率可以降低；
- 但变化超过裁剪范围后，不再继续放大该次更新收益。

### 6.3 可选 KL 约束

一些 GRPO 配置还会限制当前策略不要偏离 reference model 太远：

```text
L = L_policy + beta * KL(pi_theta || pi_ref)
```

`beta` 越大，模型越保守；`beta` 越小，Reward 对模型的影响越强。

当前项目使用 TRL `0.24.0`，配置中没有显式设置 `beta`，该版本默认 `beta = 0.0`。因此当前实验不使用 reference model KL 惩罚。

## 7. 当前 TRL 实际采用的默认值

当前 YAML 只显式填写了一部分参数，其他行为来自 TRL `0.24.0` 的默认值。理解实验时不能只看 YAML。

关键默认值包括：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `beta` | `0.0` | 不启用 reference model KL 惩罚 |
| `num_iterations` | `1` | 每批 rollout 只做一次策略更新 |
| `epsilon` | `0.2` | PPO/GRPO 概率比裁剪范围 |
| `scale_rewards` | `group` | 在同一 prompt 的生成组内标准化 Reward |
| `loss_type` | `dapo` | 使用 DAPO 风格的 token 级 loss 聚合 |
| `importance_sampling_level` | `token` | 按 token 计算重要性采样比例 |
| `use_vllm` | `false` | 使用训练进程中的 Transformers 生成，不使用 vLLM |

这里需要区分两个概念：

- 整体训练方法仍然是 GRPO；
- TRL 当前默认用 `dapo` 方式聚合策略 loss。

因此报告中可以写“TRL GRPO”，但复现实验时应同时记录 `loss_type=dapo`，避免不同版本默认值变化后结果无法横向比较。

## 8. 当前 Reward 是怎么写的

Reward 的入口位于 [grpo.py](../src/post_training/grpo.py) 中的 `customer_service_json_reward`。

当前版本不是一个裁判大模型，而是确定性的 Python 规则 Reward。它读取：

- 模型 completion；
- 标准 `intent`；
- 标准手机号；
- 标准运单号；
- 期望回复前缀；
- 数据集中出现过的合法 intent 集合。

### 8.1 硬失败规则

以下情况直接得到极低分或 0 分：

| 情况 | Reward |
| --- | ---: |
| completion 出现 thinking 内容 | `-10` |
| 无法解析为 JSON | `0` |
| 顶层字段不是严格的 `intent/slots/reply` | `0` |
| `slots` 字段不是严格的 `phone/waybill_no` | `0` |
| 字段类型错误 | `0` |
| `reply` 为空 | `0` |

硬失败用于防止模型通过“部分正确”拿到较高分。例如，一个包含正确运单号但输出大量推理过程的回答，不应该因为槽位正确而获得正向训练信号。

### 8.2 分阶段加减分

通过结构校验后，Reward 再按业务质量累积分数，最终限制在 `[0, 10]`：

| 条件 | 分值 |
| --- | ---: |
| JSON 结构通过后的基础分 | `+2.0` |
| 没有 Markdown 或额外解释 | `+0.5` |
| 出现 Markdown 或额外解释 | `-1.0` |
| 凭空生成手机号或运单号 | `-2.0` |
| intent 属于合法标签集合 | `+0.25` |
| intent 与标注完全一致 | `+1.5` |
| intent 合法但与标注不一致 | `-1.0` |
| intent 非法 | `-1.5` |
| phone 抽取正确 | `+1.0` |
| phone 错误且非空 | `-1.0` |
| waybill_no 抽取正确 | `+1.0` |
| waybill_no 错误且非空 | `-1.0` |
| reply 非空 | `+0.5` |
| reply 以期望前缀开头 | `+3.0` |
| 未使用期望前缀 | `-1.5` |
| reply 与参考回复的字符 F1 | 最高 `+1.0` |
| reply 长度在 8～160 字符 | `+0.25` |
| reply 长度异常 | `-0.25` |
| 非“其他”意图却使用通用兜底话术 | `-2.0` |

这套 Reward 已经能够明显强化：

- 合法 JSON；
- no-think；
- 固定回复前缀；
- 基础 intent 和 slot 正确性。

但它仍有一个已知缺口：

> 字段正确不代表 reply 与字段一致。

例如模型已经正确输出：

```json
"waybill_no": "JT1234567890123"
```

但 `reply` 仍然说“请提供运单号”。当前 Reward 可能仍给出较高分，因为它分别检查了 slot 和 reply 风格，却没有充分检查两者之间的逻辑一致性。

后续 Reward 最重要的改进不是继续增加格式分，而是增加业务一致性规则，例如：

```text
如果 expected_waybill 非空：
  reply 再次索要运单号 -> 扣分

如果 expected_phone 非空：
  reply 再次索要手机号 -> 扣分

如果 intent=手机号查件：
  reply 应说明手机号查询或隐私验证流程

如果 intent=我要理赔：
  reply 应包含异常核实或材料引导
```

## 9. 为什么 Reward 越严格不一定越好

如果把所有条件都写成硬失败，很容易出现：

```text
[0, 0, 0, 0]
```

这时整组 Advantage 都是 0，GRPO 无法判断哪个回答相对更好。

Reward 设计需要分层：

1. **不可接受的输出**使用硬失败，例如非法 JSON、thinking、字段类型错误；
2. **可比较的业务错误**使用连续加减分，例如 intent 错误、漏 slot、重复索要已有信息；
3. **风格偏好**使用较小权重，避免覆盖核心业务正确性。

一个更合理的优先级是：

```text
结构合法
  > 槽位与意图正确
  > reply 与槽位/意图一致
  > 回复风格与长度
```

如果固定前缀的分值过高，模型最容易先学会前缀，却不一定学会更难的业务逻辑。这正是当前实验已经观察到的现象。

## 10. SFT LoRA 是如何接入 GRPO 的

当前配置指定：

```yaml
model:
  name_or_path: /root/nfs/llm-models/Qwen3.5-9B
  adapter_name_or_path: outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat
```

训练代码先加载 base model，再执行：

```python
model = PeftModel.from_pretrained(
    model,
    adapter_name_or_path,
    is_trainable=True,
)
```

因此当前正确链路是：

```text
Qwen3.5-9B base weights（冻结）
  + SFT LoRA weights（加载并设为可训练）
  -> GRPO 继续更新这组 LoRA weights
```

这不是“在 SFT adapter 上再叠一层新 adapter”。GRPO 继续训练的就是加载进来的同一组 LoRA 参数。

当 `adapter_name_or_path` 存在时，代码把 `peft_config` 设为 `None`，不会再根据 YAML 中的 `lora` 配置新建第二个 LoRA adapter。

## 11. 为什么需要转换 Adapter key 前缀

最初由 LLaMA-Factory 保存的 Qwen3.5 LoRA key 前缀类似：

```text
base_model.model.model.language_model.layers...
```

当前 TRL/Transformers 路径下 PEFT 期望的前缀类似：

```text
base_model.model.model.layers...
```

虽然两边的 base model 和 `target_modules` 一致，但 key 路径不同会导致大量 `missing adapter keys`。训练仍可能启动，是因为 PEFT 创建了可训练 LoRA 模块，但原 SFT 权重没有正确落入这些模块。这样训练更接近：

```text
base model + 新初始化 LoRA -> GRPO
```

而不是预期的：

```text
base model + 已训练 SFT LoRA -> GRPO
```

转换脚本只复制并改写 adapter 权重 key，不修改 base model，也不覆盖原 adapter：

```bash
python scripts/convert_peft_adapter_key_prefix.py \
  --source outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages \
  --output outputs/llamafactory/local-qwen3_5-9b/lora/sft_10k_3ep_messages_trl_compat \
  --force
```

转换后的兼容性检查结果为：

- 496 个权重 key 完成转换；
- 248 个 LoRA 模块全部匹配；
- current model 缺失模块为 0；
- adapter 多余模块为 0。

所以 Adapter 兼容性必须在 Reward 调试之前确认。否则看到的 GRPO 效果和 loss 都不能代表“SFT 后继续做 GRPO”。

## 12. no-think 是如何处理的

当前代码从三层处理 thinking：

### 12.1 Prompt 模板

`apply_no_think_chat_template` 使用 tokenizer 的 chat template，并尝试传入：

```python
enable_thinking=False
```

`qwen3_nothink` 一类模板可能在 Prompt 中预填：

```text
<think>

</think>
```

这个空块属于输入 Prompt，不等于模型在 Completion 中主动输出思考过程。

### 12.2 生成约束

当前代码通过 `bad_words_ids` 抑制：

- `<think>`；
- `</think>`；
- `Thinking Process`；
- 其他明显推理标记。

这控制的是 rollout 阶段的生成搜索空间。

### 12.3 Reward 硬惩罚

如果 Completion 中仍出现 thinking 内容，Reward 直接为 `-10`。

三层职责不同：

| 层 | 作用 |
| --- | --- |
| chat template | 告诉模型当前会话采用 no-think 模式 |
| generation constraint | 尽量阻止 rollout 采样 thinking token |
| Reward | 对仍然发生的 thinking 输出施加明确负反馈 |

GRPO 本身不只适用于 thinking 模型。它适用于任何“能生成多个候选，并能对候选自动打分”的任务。no-think 只是当前业务输出约束。

## 13. 配置参数如何映射到训练行为

当前核心配置如下：

```yaml
data:
  max_samples: 1000

reward:
  name: customer_service_json_staged

training:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-6
  max_prompt_length: 1024
  max_completion_length: 256
  num_generations: 4
  temperature: 0.8
  gradient_checkpointing: true
  bf16: true
  deepspeed: configs/deepspeed/zero2_bf16.json
```

### 13.1 Batch size

8 张 GPU 时：

```text
Global prompt batch size
= per_device_train_batch_size
  * GPU 数量
  * gradient_accumulation_steps
= 1 * 8 * 8
= 64 prompts / optimization step
```

每个 prompt 生成 4 个 completion，因此每个 optimization step 从概念上要处理：

```text
64 * 4 = 256 completions
```

这也是 GRPO 明显慢于普通 SFT 的主要原因之一：每次更新前要先进行大量自回归生成。

### 13.2 `num_generations`

`num_generations` 增大：

- 组内比较样本更多；
- 更容易观察到 Reward 差异；
- 生成耗时和显存开销增加。

`G=4` 是当前实验的折中，不代表固定最佳值。

### 13.3 `temperature`

温度过低时，4 个 completion 可能高度相同，导致 Reward 方差和 Advantage 接近 0。

温度过高时，候选多样性增加，但非法或低质量输出也会增加。

当前使用 `0.8`，目的是让同组回答保持一定差异。

### 13.4 `max_completion_length`

它是生成上限，不代表每条 completion 实际都有 256 tokens。过大可能：

- 增加 rollout 时间；
- 让异常回答持续生成；
- 放大 benchmark token 上界估算。

结构化 JSON 任务通常应结合真实长度分布设置，而不是盲目拉长。

## 14. 为什么 GRPO 训练比 SFT 慢

SFT 已经有标准答案，只需完成 forward、backward 和参数更新。

GRPO 每次更新前还要：

1. 为每个 prompt 自回归生成多个 completion；
2. 对每个 completion 计算 Reward；
3. 重新计算 completion 的 token log probability；
4. 计算组内 Advantage；
5. 再执行策略 loss 的 backward。

当前实验没有启用 vLLM，rollout 由训练进程中的 Transformers `generate` 完成。因此 1000 条样本的 GRPO 耗时较长是正常现象，不能直接与 1000 条 SFT 的耗时比较。

## 15. 如何理解 GRPO 的 train loss

GRPO 的 `train_loss` 不能像 SFT 交叉熵一样解释。

SFT loss 下降通常表示模型越来越会复现标准答案 token；GRPO loss 来自带正负 Advantage 的策略目标，而且 Advantage 在组内中心化：

```text
sum_i A_i approximately 0
```

因此 GRPO loss：

- 可能非常接近 0；
- 可能出现小幅负数；
- 不能单独判断任务是否已经学会；
- 不能与 SFT loss 做数值横向比较。

例如当前实验出现过：

```text
train_loss = 0.00046
```

以及：

```text
train_loss = -0.00336
```

这两者本身都不表示训练异常。真正需要观察的是：

- 每组 Reward 是否有方差；
- `Advantage` 是否长期全部为 0；
- 平均 Reward 是否改善；
- 非法 JSON 比例是否下降；
- intent/slot 准确率是否改善；
- held-out 测试上的业务一致性是否改善；
- SFT 原有能力是否退化。

## 16. 如何理解 GRPO benchmark 中的 tokens/s/GPU

当前 [grpo.py](../src/post_training/grpo.py) 会输出 JSON 和 Markdown benchmark 报告。

GRPO 的 `tokens_per_second_per_gpu_estimated` 是上界估算：

```text
总 token 上界
= (实际统计的 prompt tokens + max_completion_length)
  * num_generations
  * 训练样本处理次数

tokens/s/GPU
= 总 token 上界 / train_runtime / GPU 数量
```

需要注意：

- prompt token 使用 tokenizer 实际统计；
- completion token 使用 `max_completion_length`，不是实际生成长度；
- 每个 prompt 乘以 `num_generations`；
- 因此这是上界估算，不是精确实际吞吐。

它不能直接与 SFT 的 `tokens/s/GPU` 横向比较，因为：

1. GRPO 包含在线生成；
2. GRPO 对一个 prompt 生成多个 completion；
3. 当前 completion token 使用长度上限估算；
4. GRPO 还包含 Reward、log probability 和策略更新开销。

报告中应保留：

```text
tokens_per_second_per_gpu_estimate_type =
upper_bound_prompt_plus_max_completion_times_num_generations
```

避免把估算值误写为精确训练 token 吞吐。

## 17. 当前实验已经证明了什么

当前 GRPO 实验已经验证：

1. Qwen3.5-9B 可以在 TRL `0.24.0` 中完成 LoRA GRPO；
2. LLaMA-Factory SFT adapter 可以在转换 key 前缀后正确接入 TRL；
3. 规则 Reward 可以让模型强化合法 JSON 行为；
4. rollout 生成约束与 Reward 可以抑制 Completion 中的 thinking；
5. 固定 reply 前缀能够被明显学到；
6. 格式行为比复杂业务一致性更容易被 Reward 强化。

同时也暴露出：

1. Reward 过于宽松时，不合预期的回答也可能拿满分；
2. Reward 过于严格时，同组可能全部为 0，导致 Advantage 为 0；
3. 固定前缀分值过高时，模型优先学模板，而不是业务推理；
4. slot 正确不等于 reply 与 slot 一致；
5. GRPO 会放大 Reward 所定义的目标，也会放大 Reward 的漏洞。

因此不能把“Reward 提高”直接等价为“模型更聪明”。准确说法是：

> 模型更擅长获得当前 Reward 函数给出的高分。

Reward 是否真的代表业务质量，决定了 GRPO 最终优化方向。

## 18. 一个最小化的 GRPO 伪代码

下面的代码省略了分布式通信、变长 mask、梯度累积、DAPO token 聚合和工程优化，只用于理解主流程：

```python
for prompts, labels in dataloader:
    # 1. 每个 prompt 生成 G 个候选回答
    completions = []
    for prompt in prompts:
        completions.extend(
            model.generate(prompt, num_return_sequences=G)
        )

    # 2. 规则或裁判模型打分
    rewards = reward_fn(prompts, completions, labels)

    # 3. 按 prompt 分组，并在组内标准化
    rewards = rewards.view(len(prompts), G)
    advantages = (
        rewards - rewards.mean(dim=1, keepdim=True)
    ) / (
        rewards.std(dim=1, keepdim=True) + 1e-4
    )

    # 4. 得到生成时旧策略的 log probability
    with torch.no_grad():
        old_logps = sequence_logps(model, prompts, completions)

    # 5. 当前策略重新计算 log probability
    logps = sequence_logps(model, prompts, completions)
    ratio = torch.exp(logps - old_logps)

    # 6. PPO 风格裁剪目标
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

真实的 `GRPOTrainer` 还需要处理：

- completion token mask；
- 多卡之间 gather Reward；
- prompt 与 completion 拼接；
- 每个 token 的 log probability；
- 梯度累积；
- ZeRO 分片；
- rollout 与训练模型同步；
- 可选 reference model KL；
- 指标聚合与 checkpoint 保存。

## 19. 推荐的系统化学习顺序

如果目标是以后能够自己写 GRPO 训练代码，建议按以下顺序学习：

### 第一步：读通当前数据流

需要能够回答：

1. 一条 JSONL 数据中的每个字段在哪里使用？
2. 哪些字段进入模型 Prompt？
3. 哪些字段只进入 Reward？
4. `answer` 为什么不是 SFT 的 teacher-forcing label？

### 第二步：手算一组 Reward 和 Advantage

任选一个 prompt，准备 4 个 completion，手工计算：

```text
Reward -> mean -> std -> Advantage
```

然后分别尝试：

```text
[10, 7, 5, 2]
[7, 7, 7, 7]
[0, 0, 0, 0]
```

理解为什么后两组没有有效训练信号。

### 第三步：逐条读 Reward

为每一项规则写测试样例：

- 非法 JSON；
- 多一个顶层字段；
- intent 合法但错误；
- 漏手机号；
- 幻觉运单号；
- 已有运单号却再次索要；
- reply 使用错误的兜底话术。

Reward 函数最好可以脱离 GPU 独立运行单元测试。

### 第四步：理解策略梯度

重点理解：

- 为什么不能直接对 Reward 求梯度；
- log probability 如何把生成结果和模型参数连接起来；
- Advantage 为什么有正负；
- probability ratio 为什么需要 clip；
- KL 惩罚解决什么问题。

### 第五步：写一个单卡最小版本

先不用 DeepSpeed、LoRA 和完整 Qwen3.5，在小模型和少量 prompt 上实现：

```text
generate -> reward -> group advantage -> logprob -> clipped loss -> update
```

确认 loss 能反向传播后，再逐步加入：

```text
LoRA -> 多卡 -> ZeRO -> benchmark -> 生产 Reward
```

## 20. 下一步实验建议

当前最值得做的不是盲目增加 epoch，而是改善 Reward 的业务分辨率。

建议顺序：

1. 为 Reward 增加独立单元测试；
2. 增加 slot 与 reply 一致性扣分；
3. 降低固定前缀在总分中的占比；
4. 统计每个 Reward 子项的命中率，而不只记录总分；
5. 记录 `reward_std` 和零方差组比例；
6. 建立 held-out 测试集，对比 SFT adapter 与 GRPO adapter；
7. 检查格式提升是否以业务能力或通用能力退化为代价；
8. 确认 Reward 有效后，再比较 `num_generations`、temperature、learning rate 和 epoch。

评估时至少拆成四类指标：

| 维度 | 示例指标 |
| --- | --- |
| 格式 | JSON parse rate、schema exact rate、thinking rate |
| 业务 | intent accuracy、phone F1、waybill F1 |
| 一致性 | 已有槽位时重复索要率、intent-reply 矛盾率 |
| 保真 | 与 SFT 基线相比的能力退化率 |

只有这些指标一起改善，才能说明 GRPO 真正提升了模型，而不是只让模型更会迎合 Reward。

## 21. 一句话总结

当前项目的 GRPO 是：在正确加载 Qwen3.5-9B SFT LoRA 的基础上，由 TRL 在线为每个物流客服 prompt 生成 4 个回答，使用 Python 规则 Reward 打分并计算组内相对 Advantage，再通过裁剪策略目标继续更新同一个 LoRA adapter。

最核心的工程结论是：

> GRPO 代码本身决定“怎么学”，Reward 决定“学成什么”；在 Adapter 加载正确之后，Reward 的区分度和业务一致性才是实验成败的主要因素。

## 22. 参考资料

- [TRL 0.24.0 GRPO Trainer 文档](https://github.com/huggingface/trl/blob/v0.24.0/docs/source/grpo_trainer.md)
- [TRL 0.24.0 GRPOConfig 源码](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

