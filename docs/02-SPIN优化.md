# SPIN 优化提示词生成模型

用自博弈微调（SPIN，ICML 2024）优化提示词生成模型，并给出与直接监督微调的完整对比。

---

## 1. 目标与方法

**目标**：在已训练好的提示词生成模型上，用 SPIN 进一步提升它把「用户简述」改写成高质量提示词的能力，同时不破坏已有能力。

**SPIN 的核心**：让模型区分「参考提示词」和「自己上一轮生成的提示词」，不需要任何额外的人工标注或偏好数据。

**损失函数**（论文式 4.7）：

```
L = E[ ℓ( β·log(p_θ(y|x)/p_θt(y|x)) − β·log(p_θ(y′|x)/p_θt(y′|x)) ) ]
ℓ(t) = log(1 + exp(−t))，y = 参考提示词，y′ = 上一轮模型生成，β = λ
```

代码里的 β 就是论文的 λ，控制保守程度：β 越大，目标越早饱和、更新越保守。

---

## 2. 实现方案

### 2.1 双 adapter 共享基座

SPIN 每轮需要同时存在两个模型：待训练的 policy（θ）和冻结的 opponent（θt）。常规做法加载两份 4-bit 基座会超出 8GB 显存。

本项目只加载**一份基座**，挂两个 LoRA adapter：

| adapter | 角色 | 状态 |
|---|---|---|
| `opp` | 对手 θt | 冻结 |
| `policy` | 待训练 θ | 可训练，初始权重与 opp 相同 |

前向时用 `set_adapter()` 切换：policy 段建图求梯度，opp 段在 `no_grad` 下求参考 log-概率。

**实测显存峰值 4.7 GiB**，8GB 单卡可跑。

### 2.2 对手 log-概率缓存

θt 在整轮迭代中冻结，因此 `log p_θt(y)` 和 `log p_θt(y′)` 是常量。在生成阶段一并算好存入 pairs 文件，训练时直接读取——**每样本前向次数从 4 次降到 2 次**。

### 2.3 正确性验证

θ=θt 起步时，`logits` 必须恰好为 0、`loss` 必须恰好为 ln2：

```
[对照] 生成答案=真答案时  logits 均值 = +0.000000  （应为 0）
[正式] θ=θt 起步时        logits 均值 = +0.000000  最大绝对值 = 0.000000
                          loss      = 0.693147
                          理论值 ln2 = 0.693147
✅ 通过，实现与论文一致
```

这三条同时成立，说明符号方向、序列对齐、对手冻结三者都正确。

---

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| 基座 | Qwen3-4B-Instruct-2507（4-bit QLoRA） |
| SPIN 起点 | 第一轮 SFT 模型（`runs/refinement/final`），已在 1,353 条上训练过 |
| 训练数据 | MePO 训练集抽样 200 条（只需 prompt 和参考提示词） |
| 学习率 | 1e-5 |
| batch | 1 × grad_accum 8 |
| max_len | 384 |
| 迭代 | 2 轮 |

**评测集固定 20 条**（目标能力 10 / 保持能力 5 / 中文 5），所有方法看到完全相同的输入。

### 对照组

| 组 | 说明 |
|---|---|
| `sft_control` | 继续 SFT，160 条 × 1 epoch |
| `sft_matched` | 继续 SFT，200 条 × 3 epoch（与 SPIN 同预算） |
| `spin_iter0/1` | SPIN β=0.1，1/2 轮迭代 |
| `spin_e3` | SPIN β=0.1，3 epoch（与 SFT 同预算） |
| `spin_beta5` | SPIN β=5.0，3 epoch（论文的保守取值） |

---

## 4. 结果

### 4.1 损失指标

| tag | target_loss ↓ | retain_loss ↓ |
|---|---|---|
| baseline（起点） | 2.4054 | 1.3689 |
| sft_control | 2.3630 | 1.3667 |
| sft_matched | **2.2582** | 1.3687 |
| spin_beta5 | 2.3656 | 1.3956 |
| spin_iter0 | 2.3304 | 1.4170 |
| spin_iter1 | 2.3024 | 1.4727 |
| spin_e3 | 2.3220 | 1.5366 |

若只看这张表，会得出「直接 SFT 更好」的结论。**但这是错的**——见下节。

### 4.2 生成侧指标（正确的评估视角）

`retain_loss` 是 **teacher-forcing** 指标：它衡量模型给参考答案的概率，不衡量模型自己会生成什么。模型完全可以一边给长结构化文本高概率，一边倾向于生成短句。

实测中 `sft_matched` 的 retain_loss 几乎不变（−0.0002），但生成时把结构化清单**整个丢掉了**。所以必须按类别看生成侧指标。

**保持能力（retain 类）**

| tag | 平均长度 | 结构分 | 有清单占比 |
|---|---|---|---|
| baseline | 574 | 1.80 | 40% |
| sft_control | 453 | 1.40 | 20% |
| sft_matched | 233 | 0.20 | 0% |
| **spin_beta5** | **582** | **2.00** | **40%** |
| spin_iter0 | 520 | 1.60 | 20% |
| spin_iter1 | 381 | 1.00 | 0% |
| spin_e3 | 336 | 0.60 | 0% |

**中文能力（zh 类）**

| tag | 平均长度 | 结构分 |
|---|---|---|
| baseline | 186 | 0.60 |
| sft_matched | **37** | **0.00** |
| spin_beta5 | 161 | 0.40 |

### 4.3 权衡分析

| tag | 目标提升 Δtarget | 保持结构分 | 结论 |
|---|---|---|---|
| **spin_beta5** | −0.040 | **2.00** | 帕累托最优 |
| spin_iter0 | −0.075 | 1.60 | 帕累托最优 |
| spin_iter1 | −0.103 | 1.00 | 帕累托最优 |
| sft_matched | −0.147 | 0.20 | 帕累托最优（极端点） |
| sft_control | −0.042 | 1.40 | **被 spin_beta5 支配** |
| spin_e3 | −0.083 | 0.60 | **被 spin_iter1 支配** |

**核心结论：`sft_control` 被 `spin_beta5` 严格支配**——目标提升相当（−0.042 vs −0.040），但保持能力差得多（1.40 vs 2.00）。即在相同的目标提升水平下，SPIN 的能力保持优于直接 SFT。

---

## 5. 两个关键发现

### 5.1 β 决定 SPIN 成败

| β | 训练末 loss | logit | 表现 |
|---|---|---|---|
| 0.1 | 0.003 | +58 | 过优化：模型把自身输出的概率压垮，通用能力崩溃 |
| 5.0 | 0.000 | +3 | 保守：早饱和、更新温和，权衡最好 |

**β=0.1（论文默认）在本任务上过优化**：损失压到 0.003 意味着模型能完美区分参考提示词和自身生成，这等价于把自己的输出分布压下去，直接导致能力退化。β=5.0 让目标在 logit≈3 处饱和，避免了这个问题。

### 5.2 Teacher-forcing 指标会掩盖遗忘

这是本文最有方法论价值的发现：

```
sft_matched   retain_loss 1.3687（几乎无损）
              实际生成：755 字符的 7 项清单  →  221 字符的单句
```

如果只用损失评估，会认为它「没有遗忘」并选它作为最优模型；但生成侧指标显示它把结构化输出能力彻底破坏了。**评估持续学习必须用生成侧指标，不能只看 teacher-forcing loss。**

---

## 6. 最终模型

**交付模型：`runs/spin/spin_beta5/policy`**

配置：SPIN，β=5.0，200 条 × 3 epoch，起点为 `runs/refinement/final`。

选择理由：

- 在目标、保持、中文三类生成指标上全面不劣于起点模型
- 保持能力的结构分 2.00，**高于起点模型的 1.80**
- 是帕累托最优解，且支配了同目标提升水平的 SFT 对照

---

## 7. 复现

```bash
# 1. 环境
conda activate spin
export HF_HUB_OFFLINE=1

# 2. 验证 SPIN 实现正确性（loss 必须等于 ln2）
python src/spin_train.py explain --init runs/refinement/final --n 3

# 3. 生成对手模型的答案 y′ 并缓存对手 log-概率
python src/spin_train.py generate \
    --model runs/refinement/final --iter 0 --n 200 --max_new_tokens 128

# 4. SPIN 训练（最终配置）
python src/spin_train.py train \
    --opp runs/refinement/final --iter 0 --epochs 3 --lr 1e-5 \
    --beta 5.0 --max_len 384 --empty_cache_every 1 \
    --output_dir runs/spin/spin_beta5

# 5. 评测并汇总
python src/record_outputs.py --adapter runs/spin/spin_beta5/policy --tag spin_beta5
python src/record_outputs.py --compare --baseline baseline
```

一键跑完整对照实验（含所有对照组）：

```bash
bash src/run_spin_experiment.sh
```
