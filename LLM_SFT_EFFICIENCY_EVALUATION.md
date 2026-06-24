# LLM 微调效率评估报告

报告日期：2026-06-24

## 1. 任务目标

基于 Qwen3.5 系列模型，在 A800 单机多卡环境下完成 SFT 训练效率验证，覆盖以下四种训练情况：

1. Qwen3.5-4B LoRA SFT
2. Qwen3.5-4B 全参数 SFT
3. Qwen3.5-9B LoRA SFT
4. Qwen3.5-9B 全参数 SFT

核心评估指标为 `tokens/s/gpu`，同时记录训练时长、样本吞吐、单卡峰值显存和训练 loss。

此外，针对 Qwen3.5-9B Full SFT 补充进行 8K 序列长度可行性验证，用于观察长序列场景下的显存占用、单步耗时、GPU 利用率和训练稳定性。

## 2. 实验环境与统一口径

| 项目 | 配置 |
| --- | --- |
| GPU | 8 x NVIDIA A800-SXM4-80GB |
| 运行方式 | 单机 8 卡 |
| NVIDIA Driver | 530.30.02 |
| CUDA | 12.1 |
| 训练框架 | LLaMA-Factory |
| 精度 | BF16 |
| 数据集 | 物流客服意图识别、槽位抽取与回复生成 SFT 数据 |
| 样本数 | 10,000 |
| Epoch | 3 |
| 平均序列长度 | 约 145.87 tokens/sample |
| Cutoff length | 2,048 |
| Global batch size | 64 |
| LoRA 配置 | Rank 16，Alpha 32，Dropout 0.05，Target all |
| 分布式策略 | LoRA 使用 DeepSpeed ZeRO-2；4B Full 使用 ZeRO-3；9B Full 同时验证 ZeRO-2 和 ZeRO-3 |

四组实验使用相同数据、epoch、cutoff length 和 global batch size，保证主要训练效率指标具备横向可比性。

`tokens/s/gpu` 由模型 tokenizer 按实际 chat template 统计训练 token 数，再除以训练耗时和 GPU 数量得到。该指标为训练全流程平均估算值，不是瞬时峰值。

- Qwen3.5-9B Full 增加 ZeRO-2 补充实验，用于比较 ZeRO-2 与 ZeRO-3 的吞吐和显存差异。
- 8K 实验使用 10,000 条有效长度约 8,000～8,192 tokens 的多轮样本、1 epoch，不纳入 2K 主实验吞吐对比。

## 3. 实验结果

| 模型 | 训练方式 | DeepSpeed | 模型参数量 | 可训练参数量 | 可训练参数占比 | 训练耗时 | Tokens/s/GPU | 单卡峰值显存 | Train loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | LoRA | ZeRO-2 | 4.660B | 32.465M | 0.70% | 1.21 h | **125.90** | 12.21 GiB | 0.0760 |
| Qwen3.5-4B | Full | ZeRO-3 | 4.660B | 4.206B | 90.26% | 1.79 h | **84.77** | 20.04 GiB | 0.0562 |
| Qwen3.5-9B | LoRA | ZeRO-2 | 9.653B | 43.278M | 0.45% | 1.22 h | **124.41** | 21.38 GiB | 0.0760 |
| Qwen3.5-9B | Full | ZeRO-3 | 9.653B | 8.954B | 92.76% | 1.80 h | **84.50** | 35.38 GiB | 0.0571 |
| Qwen3.5-9B | Full | ZeRO-2 | 9.653B | 8.954B | 92.76% | 1.19 h | **127.17** | 48.34 GiB | 0.0566 |

说明：

- 峰值显存由训练期间 `nvidia-smi` 定时采样获得，表中 MiB 已换算为 GiB。
- 模型总参数量包含模型权重中的全部参数；框架报告的可训练参数量可能不包含冻结模块。
- Train loss 用于记录训练收敛状态，不能单独作为业务效果或泛化能力结论。

Qwen3.5-9B Full 的 ZeRO-2 吞吐高于 ZeRO-3，`tokens/s/GPU` 从 84.50 提升至 127.17，但单卡峰值显存从 35.38 GiB 增加至 48.34 GiB。

### 3.1 Qwen3.5-9B Full 8K 可行性验证

| 项目 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- |
| GPU | 8 × A800 80GB | 8 × A800 80GB |
| 模型 | Qwen3.5-9B | Qwen3.5-9B |
| 训练方式 | Full SFT | Full SFT |
| 有效序列长度 | 8,000～8,192 tokens | 8,000～8,192 tokens |
| 样本数 | 10,000 | 10,000 |
| Epoch | 1 | 1 |
| Per-device batch size | 1 | 1 |
| 单次全局 micro batch | 8 | 8 |
| Gradient accumulation | 8 | 8 |
| Global batch size | 64 | 64 |
| 可训练参数量 | 8,953,803,264 | 8,953,803,264 |
| 单卡显存 | 约 65 GiB | 约 56.0～58.4 GiB |
| GPU 利用率 | 未完成，未形成稳定统计 | 预热后约 76%～85% |
| Tokens/s/GPU | N/A | 约 379～382，阶段性估算 |
| 单步耗时 | 首个 backward 超时 | 首步约 182 秒，稳定后约 169～171 秒 |
| 验证进度 | 首个 backward 阶段失败 | 成功完成至少 15 个 optimization steps |
| 直接原因 | NCCL ALLREDUCE 超过 600 秒 | - |
| 通信规模 | 176,169,472 个元素 | - |
| 是否 OOM | 否，日志未出现 CUDA OOM | 否 |
| 结论 | 当前配置不具备稳定训练条件 | 具备 8K 全参数训练可行性，优先采用 |

Qwen3.5-9B Full + ZeRO-3 + 8K 已稳定完成多个 optimization steps，单卡显存未达到 80GB 上限，验证了 8K 全参数训练可行性。ZeRO-2 在首个 backward 阶段发生 NCCL ALLREDUCE 超时，日志未出现 CUDA OOM；初步判断为 8K 反向重计算耗时与 ZeRO-2 大规模梯度同步叠加导致 rank 间同步超时，暂未继续调参。

### 3.2 ZeRO-3 8K Micro Batch 对比

三组实验保持 global batch size 64，只调整单卡 micro batch 和梯度累积次数。

| 项目 | BS1 / GA8 | BS4 / GA2 | BS2 / GA4 |
| --- | ---: | ---: | ---: |
| DeepSpeed | ZeRO-3 | ZeRO-3 | ZeRO-3 |
| Per-device batch size | 1 | 4 | 2 |
| GPU 数量 | 8 | 8 | 8 |
| 单次全局 micro batch | 8 | 32 | 16 |
| Gradient accumulation | 8 | 2 | 4 |
| Global batch size | 64 | 64 | 64 |
| 有效序列长度 | 8,000～8,192 tokens | 8,000～8,192 tokens | 8,000～8,192 tokens |
| 验证进度 | 至少 15 steps | 启动后 OOM | 待补充 |
| 单卡峰值显存 | 约 56.0～58.4 GiB | 未形成有效统计 | 待补充 |
| GPU 利用率 | 预热后约 76%～85% | 未形成稳定统计 | 待补充 |
| 单步耗时 | 稳定后约 169～171 秒 | N/A | 待补充 |
| Tokens/s/GPU | 约 379～382，阶段性估算 | N/A | 待补充 |
| 是否 OOM | 否 | 是 | 待补充 |
| 结论 | 具备可行性 | 当前显存条件下不可行 | 待补充 |

## 4. 对比分析

### 4.1 LoRA 与全参数训练

| 模型 | LoRA 吞吐提升 | LoRA 耗时降低 | LoRA 峰值显存降低 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | 48.5% | 32.7% | 39.1% |
| Qwen3.5-9B | 47.2% | 32.1% | 39.6% |

LoRA 在 4B 和 9B 模型上表现出稳定、接近的效率收益：

- 单卡 token 吞吐相对全参数训练提升约 47% 至 49%。
- 相同数据和 epoch 下，整体训练耗时缩短约 32%。
- 单卡峰值显存降低约 39%。
- LoRA 仅训练约 0.45% 至 0.70% 的模型参数，更适合快速业务迭代和低成本实验。

该组对比沿用实际训练配置：LoRA 使用 ZeRO-2，Full 基线使用 ZeRO-3，因此效率差异同时包含训练方式和 DeepSpeed 策略影响，不能全部归因于 LoRA。

### 4.2 4B 与 9B 模型规模

| 训练方式 | 9B 相比 4B 的吞吐变化 | 9B 相比 4B 的显存增长 |
| --- | ---: | ---: |
| LoRA | -1.2% | +75.1% |
| Full | -0.3% | +76.6% |

本次短序列数据和 8 卡训练设置下，4B 与 9B 的平均 `tokens/s/gpu` 接近，但 9B 的峰值显存明显更高：

- LoRA：4B 为 125.90 tokens/s/gpu，9B 为 124.41 tokens/s/gpu。
- Full：4B 为 84.77 tokens/s/gpu，9B 为 84.50 tokens/s/gpu。
- 9B 的单卡显存占用相比 4B 增加约 75% 至 77%。

该结果说明本次实验的训练吞吐不只由参数规模决定，还受到短序列、数据处理、通信和框架调度等固定开销影响。因此，不能将本结果直接外推到长上下文训练；若业务将使用更长序列，应补充 1K、2K 或更长实际有效 token 长度的专项 benchmark。

### 4.3 Qwen3.5-9B Full ZeRO-2 与 ZeRO-3

在 2K 主实验中，ZeRO-2 的吞吐为 127.17 tokens/s/GPU，高于 ZeRO-3 的 84.50 tokens/s/GPU；代价是单卡峰值显存由 35.38 GiB 增加到 48.34 GiB。说明在显存容量允许时，ZeRO-2 可以减少参数分片和聚合带来的通信开销；ZeRO-3 更节省显存，更适合长序列或更大模型。

8K 补充验证中，ZeRO-3 已验证可持续完成训练 step；ZeRO-2 初始配置在首个 backward 阶段发生 NCCL ALLREDUCE 超时。长序列正式实验优先采用 ZeRO-3。

## 5. 结论

1. 四种 SFT 方案均在 8 x A800 80GB 单机环境中稳定完成，4B 和 9B 的全参数及 LoRA 训练均具备可执行性。
2. LoRA 是本次实验中训练效率更高的方案，4B 和 9B 均达到约 124 至 126 tokens/s/gpu。
3. ZeRO-3 全参数训练吞吐约为 84.5 tokens/s/GPU，训练时间约 1.8 小时，并需要更高显存。
4. 9B Full 2K 场景中，ZeRO-2 吞吐更高，ZeRO-3 显存占用更低；策略选择应结合显存余量和训练稳定性。
5. 9B Full + ZeRO-3 已验证 8K 全参数训练可行性；`BS4/GA2` 发生 OOM，当前继续验证 `BS2/GA4`。
6. 对需要快速迭代、频繁更新或并行开展多组实验的业务，建议优先采用 LoRA。
7. 本报告评估的是训练效率，不代表模型效果结论。模型选型还需结合独立测试集上的意图准确率、槽位抽取准确率、JSON 合法率和回复质量进行综合判断。

## 6. 汇报摘要

在 8 x A800 80GB 单机多卡环境下，已完成 Qwen3.5-4B、Qwen3.5-9B 的 LoRA 和全参数 SFT 主实验。LoRA 的训练吞吐约为 124 至 126 tokens/s/GPU；9B Full 在 ZeRO-2 下达到 127.17 tokens/s/GPU，但显存高于 ZeRO-3。补充的 8K 验证表明，9B Full + ZeRO-3 在 `BS1/GA8` 下能够稳定完成训练 step；`BS4/GA2` 发生 OOM，当前继续验证保持 global batch 64 的 `BS2/GA4` 配置。
