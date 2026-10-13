# 09 后训练源码导读：SFT → RLHF/PPO → DPO → GRPO → OPD（On-Policy Distillation）

> 读者假设：你知道"预训练 → SFT → 偏好对齐 / RL"这条流水线是什么，但打开 TRL / verl 的源码会被几千行的 Trainer 淹没。
> 本章的做法：先用公式和几十行自己写的 PyTorch 把每个算法"压缩"成核心，再带你按阅读顺序走真实源码，每处都给出文件、函数、行号和 GitHub 永久链接。
>
> 所有路径、符号、行号都在下列 commit 上用 `grep -n` / `sed -n` 核对过：
>
> | 子模块 | 上游仓库 | commit |
> |---|---|---|
> | `repos/08-rl-alignment/trl` | huggingface/trl | `904125670a7f41b1d5f9b61b946f90644a4ca72f` |
> | `repos/08-rl-alignment/verl` | volcengine/verl | `76e837e0bfcb18b380a66c0ba56ef9f0ac840f09` |
> | `repos/08-rl-alignment/OpenRLHF` | OpenRLHF/OpenRLHF | `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53` |
> | `repos/08-rl-alignment/tinker-cookbook` | thinking-machines-lab/tinker-cookbook | `70767c738ba5f523b66210a70e50b0dc14c9fc1a` |
>
> 下文用缩写 `TRL@`、`VERL@`、`ORLHF@`、`TINKER@` 表示上面四个仓库的 permalink 前缀，即
> `https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/`、
> `https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/`、
> `https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/`、
> `https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/`。
> 每个引用都同时给出完整链接。

---

## 0. 先建立一张总图

所有后训练算法本质上都在做同一件事：**调整 `log π_θ(y_t | x, y_<t)`（模型给每个 token 的对数概率）**，区别只在于：

1. **数据从哪来**：固定数据集（SFT、DPO、离线 KD）还是模型自己现采样（PPO、GRPO、OPD）。
2. **每个 token 的"权重/信号"是什么**：SFT 是常数 1；DPO 是由偏好对决定的一个序列级标量；PPO/GRPO 是 advantage；OPD 是"学生 vs 老师"的逐 token 对数概率差。

所以有一个**全章通用的基础操作**：给定 `input_ids` 和 logits，算出每个位置上"实际出现的那个 token"的 log 概率（per-token logp）。先把它彻底搞懂。

### 0.1 最小实现：per-token logp gather

```python
import torch
import torch.nn.functional as F

def per_token_logps(logits, input_ids, loss_mask):
    """
    logits:    (B, L, V)  模型对每个位置输出的未归一化分数
    input_ids: (B, L)     prompt + completion 拼在一起的 token
    loss_mask: (B, L)     1 = 这个 token 要算 loss（通常是 completion 部分）
    返回: logp (B, L-1), mask (B, L-1)
    """
    # 因果 LM：位置 t 的 logits 预测的是位置 t+1 的 token，所以要错开一位
    logits = logits[:, :-1, :]            # (B, L-1, V)
    targets = input_ids[:, 1:]            # (B, L-1)
    mask = loss_mask[:, 1:].float()       # 与 targets 对齐

    logp_all = F.log_softmax(logits.float(), dim=-1)                   # (B, L-1, V)，用 fp32 求稳
    logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)      # (B, L-1)
    return logp * mask, mask

# 序列 log 概率 = 有效 token 的 logp 之和
# seq_logp = (logp * mask).sum(-1)   # (B,)
```

三个要点（后面每个源码里都会反复出现）：

- **shift**：`logits[:, :-1]` 对 `input_ids[:, 1:]`。忘记错位是最常见的 bug。
- **mask**：prompt、padding 不算 loss。HF 的约定是把 label 设成 `-100`。
- **显存**：`(B, L, V)` 的 fp32 log_softmax 很大。例如 B=8、L=4096、V=151936 时，`8*4096*151936*4 B ≈ 19.9 GB`。所以工业实现都会分块（chunk）或者干脆不物化 logits（见第 6 节）。

---

## 1. SFT：带 mask 的交叉熵（简述）

**要解决的问题**：让模型学会"按指令/对话格式回答"。
**一句话直觉**：在 (prompt, response) 上做 next-token 交叉熵，但只在 response 上算 loss。

**核心原理**：

```
L_SFT = - (1/N) Σ_{b,t} m_{b,t} · log π_θ(y_{b,t} | y_{b,<t})
m: (B, L) 0/1 mask，N = Σ m
```

HF 的实现方式：`labels = input_ids.clone()`，把不算 loss 的位置设为 `-100`，`CrossEntropyLoss(ignore_index=-100)` 自动跳过。

**Packing**：把多条短样本拼成一条长序列以减少 padding。难点是**不能让第 2 条样本 attend 到第 1 条**，也不能让第 1 条的最后一个 token 去"预测"第 2 条的第一个 token。

### 源码导读（TRL `SFTTrainer`）

1. **label 是怎么被设成 -100 的**：`trl/trainer/sft_trainer.py` L1396-L1423，`build_labels`
   [TRL@trl/trainer/sft_trainer.py#L1396-L1423](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/sft_trainer.py#L1396-L1423)

   ```python
   def build_labels(example, mask_columns):
       masks = [example[column] for column in mask_columns]
       labels = [
           token_id if all(bits) else -100
           for token_id, *bits in zip(example["input_ids"], *masks, strict=False)
       ]
   ```
   - `mask_columns` 可能是 `completion_mask`（只在 `completion_only_loss=True` 时）和/或 `assistant_masks`（多轮对话里只训练 assistant 轮）。
   - 只要有一个 mask 为 0，就把该位置 label 置 `-100`。没有任何 mask 时 `labels == input_ids`，即全序列都训练。

2. **Packing / padding-free 的 collator**：同文件 `DataCollatorForLanguageModeling.torch_call`，L168-L227
   [TRL@trl/trainer/sft_trainer.py#L168-L227](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/sft_trainer.py#L168-L227)

   ```python
   if self.padding_free:
       input_ids = [torch.cat(input_ids, dim=0)]       # 整个 batch 拼成 1 行
       labels = [torch.cat(labels, dim=0)]
       position_ids = [torch.cat(position_ids, dim=0)]
   ...
   output["labels"][output["position_ids"] == 0] = -100   # L208
   seq_starts = (output["position_ids"][0] == 0).nonzero().flatten()
   cu_seq_lens = torch.cat([seq_starts, torch.tensor([output["position_ids"].size(1)])]).to(torch.int32)
   output["cu_seq_lens_q"] = output["cu_seq_lens_k"] = cu_seq_lens
   ```
   - `position_ids` 在每条样本开头重新从 0 开始，于是 `position_ids == 0` 就是样本边界。
   - L208 把每条样本**第一个 token** 的 label 置 `-100`：模型在 shift 之后会用上一条样本的最后一个 token 预测它，这一项跨了样本，不能训练。
   - `cu_seq_lens`（累计序列长度）就是 FlashAttention varlen 接口要的参数（本仓库 `flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k)` 同名），保证注意力只在各自样本内部计算。注释里特意说**不要**给 attention_mask，否则 FA 会忽略 position_ids，把整行当成一条序列。

3. **离线打包算法**：`trl/data_utils.py` L837 `pack_dataset`，默认 `strategy="bfd"`（Best Fit Decreasing，装箱问题的贪心解，保留样本边界）
   [TRL@trl/data_utils.py#L837-L880](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/data_utils.py#L837-L880)

---

## 2. RLHF / PPO 速览：四个模型

**要解决的问题**：SFT 只会模仿；我们想优化一个"好不好"的标量信号（人类偏好、单测通过率……）。
**一句话直觉**：让模型自己生成，用奖励给生成打分，按"比预期好多少"（advantage）调整每个 token 的概率，同时用 KL 拉住它别跑太远。

**目标函数**：

```
max_θ  E_{x~D, y~π_θ(·|x)} [ r(x, y) ]  -  β · KL( π_θ(·|x) || π_ref(·|x) )
```

**经典 PPO-RLHF 需要的四个模型**：

| 角色 | 是否训练 | 作用 |
|---|---|---|
| policy / actor `π_θ` | 训练 | 生成回答，被优化的对象 |
| reference `π_ref` | 冻结（通常是 SFT 模型） | 计算 KL 惩罚，防止 reward hacking 和语言退化 |
| reward model `r_φ` | 冻结 | 给整条回答打一个标量分 |
| value / critic `V_ψ` | 训练 | 估计每个前缀的期望回报，用作 baseline，降低方差 |

**KL 进 reward 的写法**（per-token）：`r_t = -β (log π_θ(y_t) - log π_ref(y_t))`，最后一个 token 再加上 RM 分数。verl 对应 `apply_kl_penalty`：`token_level_rewards = token_level_scores - beta * kld`
[VERL@verl/trainer/ppo/ray_trainer.py#L78-L117](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/ray_trainer.py#L78-L117)

**GAE（Generalized Advantage Estimation）**：

```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ A_{t+1}        （从后往前递推）
returns_t = A_t + V(s_t)      （critic 的回归目标）
```

verl `compute_gae_advantage_return`：
[VERL@verl/trainer/ppo/core_algos.py#L215-L263](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/core_algos.py#L215-L263)

```python
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
    # skip values and TD-error on observation tokens
    nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
    lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
```
- `response_mask` 为 0 的位置（例如多轮 agent 里工具返回的 observation token）直接"跳过"，递推状态原样保留。
- 结尾 `masked_whiten` 对 advantage 做白化（减均值、除标准差）。

**PPO clipped objective**（逐 token）：

```
ratio_t = π_θ(y_t) / π_old(y_t) = exp(logp_t - old_logp_t)
L_t = - min( ratio_t · A_t ,  clip(ratio_t, 1-ε, 1+ε) · A_t )
```
`π_old` 是"采样那一刻"的策略。同一批 rollout 做多次梯度更新时，`π_θ` 会偏离 `π_old`，clip 防止一步走太远。后面 GRPO、OPD 都复用这个 loss。

**PPO 的代价**：要同时放 4 个模型；critic 和 actor 一样大时显存和计算翻倍；value 对 LLM 这种"只有最后一个 token 有奖励"的稀疏任务很难学准。这就引出了两条路：
- **DPO**：干脆不要 RM、不要采样、不要 critic。
- **GRPO**：保留采样和 RM（或规则奖励），扔掉 critic，用"同一 prompt 的一组回答"做 baseline。

---

## 3. DPO：把 RLHF 变成一个分类 loss

### 3.1 要解决的问题 / 一句话直觉

- 问题：PPO 太贵太不稳定；手里只有偏好对 `(x, y_w, y_l)`（chosen 好于 rejected）。
- 直觉：**相对于参考模型，把 chosen 的 log 概率往上推、rejected 的往下压**，推多少由一个 logistic loss 自动决定；推得已经够开的样本，梯度自动变小。

### 3.2 核心原理：推导

**第 1 步：KL 约束 RL 的闭式最优解**。对每个 x，目标 `max_π E_{y~π}[r(x,y)] - β KL(π || π_ref)` 的最优解是

```
π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x),     Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)
```
（把目标写成 `-β KL(π || π_ref·exp(r/β)/Z) + β log Z` 即可看出。）

**第 2 步：把 reward 用策略反解出来**：

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ] + β · log Z(x)
```

**第 3 步：代入 Bradley-Terry 偏好模型** `P(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))`。同一个 x 的 `β log Z(x)` 在差里**消掉**了，这是 DPO 能成立的关键：Z(x) 要对所有可能回答求和，原本根本算不出来。

**第 4 步：用参数化策略 π_θ 替换 π*，对偏好数据做最大似然**：

```
L_DPO(θ) = - E_{(x,y_w,y_l)} log σ( β · [ (log π_θ(y_w|x) - log π_ref(y_w|x))
                                        - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )
```

形状：`log π(y|x) = Σ_t log π(y_t | x, y_<t)` 是一个 **(B,) 的序列级标量**，由 per-token logp `(B, T)` 按 completion mask **求和**得到。

**梯度**（理解"自适应"）：

```
∇L = -β · σ(-β·Δ) · [ ∇log π_θ(y_w|x) - ∇log π_θ(y_l|x) ]
Δ = 上式方括号内的 margin
```
`σ(-βΔ)` 在 margin 已经很大时趋近 0，这类样本的梯度自动消失；在模型"排反了"时接近 1，梯度最大。

"隐式奖励" `r̂ = β (log π_θ - log π_ref)` 就是 TRL 日志里的 `rewards/chosen`、`rewards/rejected`。

### 3.3 最小实现

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """所有输入形状 (B,)，是 completion token 的 logp 之和"""
    chosen_logratios = policy_chosen_logps - ref_chosen_logps        # log π/π_ref (y_w)
    rejected_logratios = policy_rejected_logps - ref_rejected_logps  # log π/π_ref (y_l)
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()          # 别写成 log(sigmoid(x))，数值不稳
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    return loss, chosen_rewards, rejected_rewards

# 一次前向同时算 chosen 和 rejected：把两者在 batch 维拼起来
# input_ids = cat([prompt+chosen, prompt+rejected])   -> (2B, L)
# logp, mask = per_token_logps(model(input_ids).logits, input_ids, completion_mask)
# seq_logp = (logp * mask).sum(-1)
# policy_chosen_logps, policy_rejected_logps = seq_logp.chunk(2)
# 参考模型同样做一遍，但放在 torch.no_grad() 里
```

### 3.4 源码导读（TRL `DPOTrainer`，按阅读顺序）

> 说明：老版本 TRL 的 DPO 有 `concatenated_forward()` 和 `get_batch_logps()` 两个函数。**在本 commit 的 `trl/trainer/dpo_trainer.py` 里它们已经不存在**（grep 不到）：拼接挪到了 collator，logp 计算挪到了"fused LM head"。经典写法仍保留在实验目录的 CPO/ORPO trainer 里，第 4 步会读它作为对照。

**第 1 步：拼接发生在 collator 里**。`DataCollatorForPreference.torch_call`，L146-L207
[TRL@trl/trainer/dpo_trainer.py#L146-L207](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/dpo_trainer.py#L146-L207)

```python
prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
...
input_ids = prompt_chosen_ids + prompt_rejected_ids      # list 拼接：前 B 条 chosen，后 B 条 rejected
completion_mask = chosen_mask + rejected_mask
```
- 这就是 `concatenated_forward` 的思想：**一次前向**同时算 chosen 和 rejected（共享 prompt 的计算模式、只启动一次 kernel、FSDP 只 all-gather 一次）。
- `completion_mask` 标出 completion 的位置，prompt 部分为 0。
- 如果数据集里预先算好了 `ref_chosen_logps/ref_rejected_logps`，也会一起带上（见第 3 步）。

**第 2 步：loss 主体** `_compute_loss`，L1275-L1602
[TRL@trl/trainer/dpo_trainer.py#L1275-L1602](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/dpo_trainer.py#L1275-L1602)

```python
labels = inputs["input_ids"].masked_fill(completion_mask == 0, -100)   # L1288
outputs = model(**model_kwargs, labels=labels, fused_lm_head=True)     # L1289
seq_len = shift_completion_mask.size(1)
per_token_logps = outputs.log_probs[:, -seq_len:]
if self.ld_alpha is None:
    logps = per_token_logps.sum(dim=1)  # sum over sequence length      # L1294
...
chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]   # L1306
```
逐行解释：
- L1288：prompt 位置 label 置 `-100`。这就是"只在 completion 上算 logp"的 mask。
- L1289：`fused_lm_head=True` 是 TRL 打的补丁（第 6 节细读），它**不返回 logits**，直接返回 `log_probs`，形状 `(2B, L-1)`（已经 shift 过），`-100` 的位置值为 0。
- L1294：**按序列求和**得到 `log π(y|x)`。因为被 mask 的位置是 0，直接 `sum` 就等于"只对 completion 求和"。
  `ld_alpha` 分支是 LD-DPO（长度解耦）：chosen/rejected 公共长度以内正常求和，超出部分乘以 `ld_alpha` 降权，用来缓解 DPO 偏爱长回答。
- L1306：按 collator 的拼法，前一半是 chosen，后一半是 rejected。

**第 3 步：reference 模型的三种来源**，L1308-L1338

```python
if self.precompute_ref_logps:                                         # L1308
    ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
else:
    with torch.no_grad(), disable_gradient_checkpointing(...):
        if is_peft_model(self.model) and self.ref_model is None:
            with use_adapter(ref_model_unwrapped, adapter_name="ref" if "ref" in ... else None):   # L1323
                ref_outputs = self.model(**ref_model_kwargs, labels=labels, fused_lm_head=True)
        else:
            ref_outputs = self.ref_model(**ref_model_kwargs, labels=labels, fused_lm_head=True)    # L1328
```
三种情况：
1. **预计算**（`precompute_ref_log_probs=True`）：训练前用 `_precompute_ref_logps`（L1158-L1231）把整个数据集的 ref logp 算一遍并缓存到磁盘，训练时 ref 模型都不用放进显存。DPO 的 ref 是冻结的、数据是离线的，所以这招完全等价。
   [TRL@trl/trainer/dpo_trainer.py#L1158-L1273](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/dpo_trainer.py#L1158-L1273)
2. **LoRA/PEFT 且没给 ref_model**：把 adapter 关掉，base model 就是 ref，**省掉一整份模型**。
3. 否则用一份独立的 `self.ref_model`。

**第 4 步：logistic loss**，L1340-L1378

```python
chosen_logratios = chosen_logps - ref_chosen_logps            # L1340
rejected_logratios = rejected_logps - ref_rejected_logps
if self.f_divergence_type == "reverse_kl":  # standard DPO
    chosen_scores = chosen_logratios
    rejected_scores = rejected_logratios
...
delta_score = chosen_scores - rejected_scores                 # L1373
for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
    if loss_type == "sigmoid":
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)    # L1378
```
- 和 3.3 节的最小实现一模一样。默认 `loss_type=["sigmoid"]`、`beta=0.1`（`trl/trainer/dpo_config.py` L241-L242、L293-L294）。
- `f_divergence_type` 是 f-DPO 的推广：把 reverse KL 正则换成别的 f-散度时，`log ratio` 被替换成 `f'(ratio)`。
- 下面一长串 `elif` 是各种变体（IPO 的平方损失、hinge、robust 的标签平滑、APO、`sft` 等），可以组合使用、加权相加。读懂 `sigmoid` 一支即可。
- L1572 起是日志：`rewards/chosen = beta * chosen_logratios`，`rewards/accuracies = (chosen_rewards > rejected_rewards)`。训练时最该盯的是 `rewards/margins` 和 `logps/chosen`（见"常见坑"）。

**第 5 步（对照）：经典 `get_batch_logps`**，在 `trl/experimental/cpo/cpo_trainer.py` L752-L790
[TRL@trl/experimental/cpo/cpo_trainer.py#L752-L790](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/experimental/cpo/cpo_trainer.py#L752-L790)

```python
if not is_encoder_decoder:
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
loss_mask = labels != -100
# dummy token; we'll ignore the losses on these tokens later
labels[labels == -100] = 0
per_token_logps = selective_log_softmax(logits, labels)
if average_log_prob:
    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
else:
    return (per_token_logps * loss_mask).sum(-1)
```
- 这就是 0.1 节最小实现的"正版"。`labels[labels == -100] = 0` 是因为 `gather` 不接受负下标，先随便填个合法 id，算完再乘 mask 清零。
- `average_log_prob=True` 是 SimPO/CPO 一类"长度归一化"的做法；标准 DPO 用**求和**。

---

## 4. GRPO：去掉 critic，用"组内比较"当 baseline

### 4.1 要解决的问题 / 一句话直觉

- 问题：PPO 的 critic 又大又难训；数学/代码题天然有可验证的规则奖励（对/错），只在序列末尾给一个标量。
- 直觉：**同一道题让模型答 G 次，答得比这组平均好的回答整体加强，差的整体削弱**。组均值就是 baseline，不用学 value 网络。

### 4.2 核心原理（带形状）

对每个 prompt `x` 采样 G 个回答 `{y_1..y_G}`，得到奖励 `r ∈ R^{G}`：

```
A_i = ( r_i - mean(r) ) / ( std(r) + ε )        # 每条回答一个标量，形状 (B·G,)
A_{i,t} = A_i                                   # 广播到该回答的每个 token，(B·G, T)
```

逐 token clipped surrogate + 可选 KL：

```
ρ_{i,t} = exp( logπ_θ(y_{i,t}) - logπ_old(y_{i,t}) )                       (B·G, T)
ℓ_{i,t} = -min( ρ A_i, clip(ρ, 1-ε_low, 1+ε_high) A_i ) + β · k3_{i,t}
k3_{i,t} = exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1                 ≥ 0
```

**k3 估计器**（Schulman, "Approximating KL Divergence"）：令 `u = π_ref/π_θ`，样本 `y ~ π_θ`。
`k1 = -log u` 无偏但可正可负、方差大；`k3 = (u - 1) - log u` 也是 `KL(π_θ||π_ref)` 的无偏估计（因为 `E_{π_θ}[u - 1] = 0`），而且恒 ≥ 0、方差小。注意 GRPO 原文把 KL **直接加在 loss 里**，而不是像 PPO 那样塞进 reward。

**loss 怎么聚合**（这是各种"GRPO 变体"的主要区别）。设 mask `m ∈ {0,1}^{(N, T)}`：

| 名称 | 公式 | 特点 |
|---|---|---|
| `grpo`（原论文） | `mean_i [ Σ_t m ℓ / Σ_t m ]` | 先序列内平均再跨序列平均。长回答里每个 token 权重更小：答错的长回答惩罚被稀释，导致回答越写越长 |
| `bnpo` / token-mean | `Σ_{i,t} m ℓ / Σ_{i,t} m` | 本地 batch 内所有 token 等权 |
| `dr_grpo`（Dr. GRPO） | `Σ_{i,t} m ℓ / (N · T_max)` | 除以常数，彻底去掉长度偏置；同时建议**不除 std**（std 会让太难或太容易的题权重被放大） |
| `dapo`（DAPO） | `Σ m ℓ / (全局 token 数)` | token 级，跨梯度累积/多卡用同一个分母；配合 clip-higher（`ε_high > ε_low`，论文建议 0.28） |

### 4.3 最小实现

```python
def grpo_advantages(rewards, G, scale_by_std=True, eps=1e-4):
    """rewards: (N,) 且同一 prompt 的 G 条回答相邻排列"""
    r = rewards.view(-1, G)                          # (N/G, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:                                 # Dr.GRPO: False
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)                              # (N,)

def grpo_loss(logp, old_logp, ref_logp, adv, mask,
              eps_low=0.2, eps_high=0.2, beta=0.0, agg="token_mean"):
    """logp/old_logp/ref_logp/mask: (N, T)；adv: (N,)；old_logp/ref_logp 无梯度"""
    ratio = torch.exp(logp - old_logp)
    A = adv.unsqueeze(1)                             # (N, 1) 广播到每个 token
    pg = -torch.min(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)
    if beta > 0:
        d = ref_logp - logp
        pg = pg + beta * (torch.exp(d) - d - 1)      # k3
    if agg == "seq_mean_token_mean":                 # 原版 GRPO
        return ((pg * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    if agg == "token_mean":                          # BNPO / DAPO 风格
        return (pg * mask).sum() / mask.sum().clamp(min=1)
    if agg == "dr_grpo":
        return (pg * mask).sum() / (mask.size(0) * mask.size(1))
    raise ValueError(agg)
```

当每批 rollout 只更新一次时，`old_logp == logp.detach()`，`ratio ≡ 1`，梯度就退化为 `-A ∇logp`，也就是带 baseline 的 REINFORCE。clip 只在同一批数据做多次更新时才起作用。

### 4.4 源码导读 A：TRL `GRPOTrainer`

**第 1 步：组是怎么形成的**。`_get_train_sampler`，L1253-L1290
[TRL@trl/trainer/grpo_trainer.py#L1253-L1290](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L1253-L1290)

```python
return RepeatSampler(
    data_source=dataset,
    mini_repeat_count=self.num_generations,        # 每个 prompt 连续重复 G 次
    batch_size=self.args.generation_batch_size // self.num_generations,
    repeat_count=self.num_iterations * self.args.steps_per_generation,
    ...
)
```
- 注释里的 ASCII 图值得看：同一 prompt 的 G 份**相邻排列**，后面才能直接 `view(-1, G)` 求组均值。
- 多卡时同一组可能落在不同 GPU 上，所以奖励要先 gather 到全局再归一化（见第 3 步）。

**第 2 步：生成与 buffer**。`_prepare_inputs` L1495-L1524 每 `steps_per_generation * num_iterations` 步才调用一次 `_generate_and_score_completions`，生成的大 batch 切成若干份在梯度累积中复用。
[TRL@trl/trainer/grpo_trainer.py#L1495-L1524](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L1495-L1524)

生成本身在 `_generate_single_turn` L1736 起。vLLM 分支：
[TRL@trl/trainer/grpo_trainer.py#L1736-L1755](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L1736-L1755)

```python
if self.use_vllm:
    # Sync weights if training step changed
    if self.state.global_step != self._last_loaded_step:
        with profiling_context(self, "sync_weights"):
            self.vllm_generation.sync_weights()
        self._last_loaded_step = self.state.global_step
    _, completion_ids, logprobs, _ = self.vllm_generation.generate(...)
    # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
    logprobs = [[lp[0] for lp in seq] for seq in logprobs]
```
- 权重变了才同步（`_last_loaded_step` 防止梯度累积期间重复同步）。
- vLLM 同时返回采样时的 logprob（`sampling_per_token_logps`），后面用于"训推不一致"修正。

**第 3 步：old / ref logp 与重要性采样修正**，L2572-L2681
[TRL@trl/trainer/grpo_trainer.py#L2572-L2681](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L2572-L2681)

```python
generate_every = self.args.steps_per_generation * self.num_iterations
if self.args.gradient_accumulation_steps % generate_every != 0 or (
    self.use_vllm and self.vllm_importance_sampling_correction
):
    old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(self.model, ...)
else:
    old_per_token_logps = None
...
if self.use_vllm and self.vllm_importance_sampling_correction:
    per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    ...
    vllm_importance_sampling_ratio = torch.exp(logps_diff)
...
if self.beta != 0.0:                                   # L2651
    if self.ref_model is not None:
        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(self.ref_model, ...)
```
- 为什么要单独算 `old_per_token_logps`：如果生成和优化步对齐（每批只更新一次），`old == current.detach()`，可以省掉这次前向，到 loss 里再用 `per_token_logps.detach()` 代替（见 L3007）。
- **vLLM 的 logprob 和训练框架算的不完全相同**（不同 kernel、bf16 累加顺序、采样时可能的量化），所以用 `π_train/π_vllm` 做截断或 mask 的重要性权重。
- `beta` **默认是 0.0**（`grpo_config.py` L676-L677），此时根本不加载 ref 模型。这是近年的主流做法（DAPO 等论文发现可验证奖励下 KL 项意义不大）。

**第 4 步：logp 怎么算**。`_get_per_token_logps_and_entropies` L1392-L1481
[TRL@trl/trainer/grpo_trainer.py#L1392-L1481](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L1392-L1481)

```python
for start in range(0, input_ids.size(0), batch_size):     # 按样本分块，降低峰值显存
    ...
    # Only the completion tokens are scored
    labels = input_ids_batch.masked_fill(attention_mask_batch == 0, -100)
    labels[:, :-logits_to_keep] = -100                    # L1469：prompt 部分全部屏蔽
    with self.accelerator.autocast():
        outputs = model(**model_inputs, labels=labels, fused_lm_head=True)
    all_logps.append(outputs.log_probs[:, -logits_to_keep:])   # L1472：只留 completion 段
```
和 DPO 完全同一个套路：`-100` mask + fused LM head。

**第 5 步：组归一化 advantage**，L2693-L2751
[TRL@trl/trainer/grpo_trainer.py#L2693-L2751](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L2693-L2751)

```python
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
if self.scale_rewards in ["group", "none"]:
    std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
elif self.scale_rewards == "batch":
    std_rewards = nanstd(rewards).expand_as(rewards)
advantages = rewards - mean_grouped_rewards                 # L2726
if self.scale_rewards != "none":
    advantages = advantages / (std_rewards + 1e-4)          # L2728
...
advantages = torch.nan_to_num(advantages, nan=0.0)
process_slice = slice(self.accelerator.process_index * len(prompts), ...)
advantages = advantages[process_slice]
```
- 多个奖励函数先加权求和（`sum_then_normalize`），再 `view(-1, G)` 做组内统计。`normalize_then_sum` 分支是 GDPO 风格：每个奖励各自归一化后再加。
- `scale_rewards="none"` 就是 Dr. GRPO 的"不除 std"；`"batch"` 是用全 batch 的 std。
- 所有奖励函数都返回 None 的样本记为 NaN，被排除在组均值之外，advantage 置 0。
- `rewards_per_func` 是全局 gather 过的，算完再按 `process_slice` 切回本卡。

**第 6 步：loss**，`_compute_loss` L2961-L3231
[TRL@trl/trainer/grpo_trainer.py#L2961-L3231](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/grpo_trainer.py#L2961-L3231)

```python
old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps  # L3007
log_ratio = per_token_logps - old_per_token_logps                         # L3024
if self.importance_sampling_level == "token":
    log_importance_weights = log_ratio
elif self.importance_sampling_level == "sequence":                       # GSPO
    log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
coef_1 = torch.exp(log_importance_weights)                               # L3036
if self.beta != 0.0:
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    )                                                                    # L3042：k3
...
elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)   # L3054
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)        # L3061
...
if self.beta != 0.0:
    per_token_loss = per_token_loss + self.beta * per_token_kl          # L3091
if self.loss_type in ["grpo", "sapo"]:
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()          # L3095
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)   # L3105
elif self.loss_type in ["cispo", "dapo", "vespo"]:
    normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
    ...
    loss = (per_token_loss * mask).sum() / normalizer                    # L3114
```
- `advantages` 形状 `(B,)` 会先 `unsqueeze(1)` 变 `(B,1)` 广播到所有 token。
- 和 4.3 节最小实现逐行对应。四种聚合就是 4.2 节表格里的四行。
- **TRL 当前默认 `loss_type="dapo"`**（`grpo_config.py` L796-L797），不是 `"grpo"`；`epsilon=0.2`，`epsilon_high` 默认与 `epsilon` 相同（L688-L704）。
- `cispo`、`sapo`、`vespo` 是另外几种 surrogate（例如 CISPO 把截断后的 ratio 当作 `logp` 的系数并 stop-grad），先略过。

### 4.5 源码导读 B：verl（工业级分布式实现）

verl 把算法核心放在一个纯函数文件 `core_algos.py`，通过注册表按名字选择；调度放在 Ray trainer。

**第 1 步：GRPO advantage** `compute_grpo_outcome_advantage`，L267-L331
[VERL@verl/trainer/ppo/core_algos.py#L267-L331](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/core_algos.py#L267-L331)

```python
scores = token_level_rewards.sum(dim=-1)            # L304：(bs, T) -> (bs,)，outcome reward 只在末 token 非零
for i in range(bsz):
    id2score[index[i]].append(scores[i])            # index = 每条样本所属 prompt 的 uid
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.tensor(0.0); id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        id2mean[idx] = torch.mean(scores_tensor); id2std[idx] = torch.std(scores_tensor)
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)   # L326
    else:
        scores[i] = scores[i] - id2mean[index[i]]   # Dr.GRPO
scores = scores.unsqueeze(-1) * response_mask       # 广播到每个 token
```
- 和 TRL 不同，verl 不依赖"相邻排列"，而是用 `uid` 字典分组，更通用（样本被打乱、做负载均衡后也没问题）。
- 同一文件还有 `GRPO_VECTORIZED`、`RLOO`（leave-one-out baseline）、`REINFORCE_PLUS_PLUS`、`REMAX`、`GPG`、`OPO` 等估计器，入口统一用 `@register_adv_est` 注册（`grep -n "register_adv_est(" core_algos.py` 可以看到全部）。

**第 2 步：policy loss** `compute_policy_loss_vanilla`，L1285-L1376
[VERL@verl/trainer/ppo/core_algos.py#L1285-L1376](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/core_algos.py#L1285-L1376)

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)   # 防止 exp 溢出
ratio = torch.exp(negative_approx_kl)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)         # = -min(ρA, clip(ρ)A)
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
...
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
```
- 多了 **dual-clip**（`clip_ratio_c`，默认 3.0）：A<0 时如果 ratio 特别大，`-ρA` 会无界增长，dual-clip 再给它封一个上界。
- `cliprange_low/high` 分开就是 DAPO 的 clip-higher。

**第 3 步：loss 聚合** `agg_loss`，L1140-L1206
[VERL@verl/trainer/ppo/core_algos.py#L1140-L1206](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/core_algos.py#L1140-L1206)

四种 `loss_agg_mode`：`token-mean`（DAPO）、`token-sum`、`seq-mean-token-sum`、`seq-mean-token-sum-norm`（Dr.GRPO：再除以常数 `loss_scale_factor`，默认取 horizon）、`seq-mean-token-mean`（原版 GRPO）。注意分母用的是**全局**的 `batch_num_tokens` / `global_batch_size` 再乘 `dp_size`，原因是 DDP/FSDP 会在 dp 维对梯度求平均，这样乘回来后，最终梯度等于"全局所有 token 的平均"，与并行度无关。

**第 4 步：KL 的几种估计器** `kl_penalty` / `kl_penalty_forward`，L2188-L2251
[VERL@verl/trainer/ppo/core_algos.py#L2188-L2251](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/core_algos.py#L2188-L2251)

```python
if kl_penalty in ("kl", "k1"):
    return logprob - ref_logprob
if kl_penalty in ("mse", "k2"):
    return 0.5 * (logprob - ref_logprob).square()
if kl_penalty in ("low_var_kl", "k3"):                      # L2239
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
```
以及带 `+` 后缀的 straight-through 技巧（如 `"k3+"`）：前向值用 k3，反向梯度用 k2（`backward_score - backward_score.detach() + forward_score.detach()`）。docstring 写明了原因：k1/k3 的**值**是 KL 的无偏估计，但它们的**梯度**不是 KL 梯度的无偏估计，k2 的梯度才是。

KL 在 verl 里有两个放法：
- `algorithm.use_kl_in_reward=True`：PPO 风格，进 reward（上面的 `apply_kl_penalty`）。
- `actor.use_kl_loss=True`：GRPO 风格，进 loss。见 `workers/utils/losses.py` 的 `ppo_loss`，L57-L155：
  [VERL@verl/workers/utils/losses.py#L57-L155](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/workers/utils/losses.py#L57-L155)
  ```python
  if config.use_kl_loss:
      ref_log_prob = data["ref_log_prob"]
      kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
      kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, ...)
      policy_loss += kl_loss * config.kl_loss_coef
  ```

**第 5 步：Ray 训练主循环（高层）** `RayPPOTrainer.fit`，L1405 起
[VERL@verl/trainer/ppo/ray_trainer.py#L1405-L1720](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/ppo/ray_trainer.py#L1405-L1720)

> 注意：本 commit 中 `RayPPOTrainer` 被标注 `@deprecated("Legacy trainer is deprecated ... Please use trainer.use_v1=True instead.")`（L284），新实现在 `verl/trainer/ppo/v1/`。但这个旧循环最直观，读它理解数据流就够了。

一次 step 的顺序（行号均为本文件）：

| 阶段 | 代码 | 行号 |
|---|---|---|
| 1. rollout | `combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)`，随后 `self.checkpoint_manager.sleep_replicas()` 释放推理引擎显存 | L1513-L1514 |
| 2. reward | `_compute_reward_colocate` / `extract_reward` | L1561-L1568 |
| 3. old logp | `old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)`（训练引擎重算一遍，作为 `π_old`） | L1586 |
| 4. ref logp | `ref_log_prob = self._compute_ref_log_prob(batch)` | L1621 |
| 5. value（仅 PPO） | `values = self._compute_values(batch)` | L1627 |
| 6. advantage | `batch = compute_advantage(batch, adv_estimator=..., num_repeat=rollout.n, norm_adv_by_std_in_grpo=...)`（在 driver 进程上算） | L1667 |
| 7. actor 更新 | `actor_output = self._update_actor(batch)` | L1690 |
| 8. 权重同步 | `self.checkpoint_manager.update_weights(self.global_steps)` 把新权重推给 vLLM/SGLang | L1716 |

`compute_advantage`（L187-L282）只是一个分发器：GAE、GRPO 走专门分支，其他估计器从注册表里按名字取。

### 4.6 OpenRLHF 对照（简）

- 组归一化：`openrlhf/trainer/ppo_utils/experience_maker.py` L256-L281
  [ORLHF@openrlhf/trainer/ppo_utils/experience_maker.py#L256-L281](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L256-L281)
  ```python
  rewards = rewards.reshape(-1, args.rollout.n_samples_per_prompt)
  if args.algo.advantage.estimator == "rloo":
      baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.rollout.n_samples_per_prompt - 1)
      rewards = rewards - baseline
  elif args.algo.advantage.estimator in ["reinforce_baseline", "dr_grpo"]:
      rewards = rewards - rewards.mean(-1, keepdim=True)
  elif args.algo.advantage.estimator == "group_norm":          # 即 GRPO
      rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
  ```
  OpenRLHF 里 GRPO 叫 `group_norm`；`dr_grpo` 只减均值、不除 std。
- PPO loss：`openrlhf/models/loss.py` `PolicyLoss.forward`，L200-L205 是标准 `-min(surr1, surr2)`，并支持 `gspo`（序列级 ratio）和训推不一致修正（`is_correction_*`）。
  [ORLHF@openrlhf/models/loss.py#L116-L259](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py#L116-L259)
- KL 估计器：`openrlhf/models/utils.py` `compute_approx_kl` L69 起，k3 在 L109：`value = (-k3_log_ratio).exp() - 1 + k3_log_ratio`。
  [ORLHF@openrlhf/models/utils.py#L69-L110](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/utils.py#L69-L110)

三家实现的数学完全一致，差别在工程：TRL 基于 HF Trainer，单进程逻辑清晰；verl/OpenRLHF 用 Ray 把 actor / ref / rollout 拆成独立 worker 组。

---

## 5. OPD：On-Policy Distillation（在线策略蒸馏）

### 5.1 要解决的问题 / 一句话直觉

- 问题：有一个强老师（大模型或领域专家），想把能力转移给小学生。
  - **离线蒸馏**（SFT on teacher outputs，或在固定数据上做 forward KL）：学生只在**老师走过的状态**上学，推理时一旦自己犯错走到陌生前缀，就不会纠正（exposure bias / 分布偏移）。
  - **RL**：信号太稀疏，一整条回答只有一个 0/1 奖励。
- 直觉：**学生自己写答案，老师给学生写的每一个 token 打分**（"如果是我，我会以多大概率写这个 token"）。学生在自己真实会走到的状态上，拿到**逐 token 的稠密**监督。

### 5.2 核心原理

记学生 `p = π_θ`，老师 `q = π_T`。学生采样 `y ~ p(·|x)`，序列级 reverse KL：

```
KL(p || q) = E_{y~p} [ Σ_t ( log p(y_t | x, y_<t) - log q(y_t | x, y_<t) ) ]
```

一个样本、一个位置的 **reverse KL 单样本估计**（就是 k1）：

```
c_t = log p(y_t | ·) - log q(y_t | ·)          形状 (B, T)
```

**把它当成 RL 的逐 token 奖励**：`r_t = -c_t`。于是用任意 policy-gradient loss（REINFORCE / importance sampling / PPO）都能最小化 reverse KL：

```
A_t = -λ · c_t     （detach，不对它求导）
L   = - Σ_t ρ_t · A_t,     ρ_t = exp(log p_θ(y_t) - log p_sampler(y_t))
```

为什么这样做是对的：`∇_θ KL(p||q) = E_{y~p}[ Σ_t ∇log p(y_t|·) · Σ_{t'≥t} c_{t'} ]`（另一项 `E_p[∇log p]` 期望为 0）。精确梯度用的是"未来 KL 之和"（reward-to-go，折扣 γ=1）。tinker 的默认 `kl_discount_factor=0.0` 只用当前 token 的 `c_t`，是一个**更低方差、有偏的近似**，实践中效果很好；设成大于 0 时就会做折扣累加。

**为什么是 reverse KL**：
- forward KL `KL(q||p)`（离线 KD 的交叉熵）是 **mode-covering**：学生要覆盖老师所有有概率的地方，容量不够时会"抹平"。
- reverse KL `KL(p||q)` 是 **mode-seeking**：学生只需在自己会去的地方和老师一致；而且期望在学生分布下，正好可以用学生的样本估计，不需要老师生成任何东西。

**两种实现路线**（读源码时要分清）：

| 路线 | 每个位置用到老师的什么 | loss | 代表实现 |
|---|---|---|---|
| A. 采样 token 的 logprob → 当 advantage | 只需老师对**采样到的那个 token** 的 logprob（标量） | policy gradient（IS/PPO） | tinker-cookbook `train_on_policy.py`；verl `use_policy_gradient=True, loss_mode=k1` |
| B. 全词表（或 top-k）散度直接反传 | 老师整行分布 `(V,)` 或 top-k | `KL/JSD(p, q)` 对 logits 求导（"监督式"） | TRL `DistillationTrainer`、GKD；verl `use_policy_gradient=False` |

路线 A 对老师的要求最低（一个推理服务的 `prompt_logprobs` 即可），也可以跨机器、跨框架；路线 B 信号更丰富，但要老师和学生共享词表，且要传/算整行 logits。

### 5.3 最小实现

```python
def opd_pg_loss(student_logp, sampler_logp, teacher_logp, mask, kl_coef=1.0):
    """
    路线 A：reverse KL 作为逐 token advantage（tinker / verl 的做法）
    student_logp: (B, T) 当前学生对采样 token 的 logp，有梯度
    sampler_logp: (B, T) 采样时（rollout 引擎）记录的 logp，无梯度
    teacher_logp: (B, T) 老师对同一 token 的 logp，无梯度
    """
    with torch.no_grad():
        reverse_kl = (sampler_logp - teacher_logp) * mask       # c_t = log p - log q
        adv = -kl_coef * reverse_kl                             # 老师越不认可，advantage 越负
    ratio = torch.exp(student_logp - sampler_logp)              # importance sampling
    loss = -(ratio * adv * mask).sum() / mask.sum().clamp(min=1)
    return loss, reverse_kl.sum() / mask.sum().clamp(min=1)    # 第二项用于日志：teacher_kl

def opd_full_reverse_kl(student_logits, teacher_logits, mask):
    """
    路线 B：在学生自己的样本上，逐位置计算全词表 reverse KL，直接反传
    *_logits: (B, T, V)（已对齐到 completion 位置），mask: (B, T)
    """
    s = F.log_softmax(student_logits.float(), -1)
    t = F.log_softmax(teacher_logits.float(), -1).detach()
    kl = (s.exp() * (s - t)).sum(-1)                            # KL(p||q)，(B, T)
    return (kl * mask).sum() / mask.sum().clamp(min=1)
```

注意路线 A 里**不能**直接对 `c_t = log p_θ - log q` 求导当 loss：`∇(log p_θ(y) - log q(y)) = ∇log p_θ(y)`，梯度里根本没有老师的信息。verl 在配置校验里专门禁止了这种用法（见下文 5.5 节第 3 步）。

### 5.4 源码导读 A：tinker-cookbook（OPD 的"参考实现"）

这是 Thinking Machines 博客 *On-Policy Distillation* 对应的代码。训练和采样跑在 Tinker 云服务上（`forward_backward`、`compute_logprobs_async` 都是远程 API），所以仓库里只有"数据怎么准备"的逻辑，正好让我们专心看算法。

**第 1 步：入口 recipe** `tinker_cookbook/recipes/distillation/on_policy_distillation.py` L51-L101 `CLIConfig`
[TINKER@tinker_cookbook/recipes/distillation/on_policy_distillation.py#L51-L101](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L51-L101)

```python
model_name: str = "Qwen/Qwen3.5-9B-Base"   # Student model
teacher_model: str = "Qwen/Qwen3.5-9B"
group_size: int = 4  # Number of rollouts per prompt
kl_penalty_coef: float = 1.0
kl_discount_factor: float = 0.0
loss_fn: LossFnType = "importance_sampling"
```
文件头注释写得很直白："No correctness or format rewards are used - only KL penalty provides supervision."

**第 2 步：环境只给 prompt，奖励恒为 0**。`tinker_cookbook/distillation/datasets.py` `PromptOnlyEnv`，L85-L121
[TINKER@tinker_cookbook/distillation/datasets.py#L85-L121](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/distillation/datasets.py#L85-L121)

`step()` 的 docstring 就是 "Return zero reward always."。所以随后 `compute_advantages`（`rl/data_processing.py` L23-L42，组内减均值）算出来的基础 advantage 全是 0，**所有学习信号都来自下一步加上去的 KL 项**。
[TINKER@tinker_cookbook/rl/data_processing.py#L23-L42](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/rl/data_processing.py#L23-L42)

**第 3 步：主循环** `do_sync_training`，`tinker_cookbook/distillation/train_on_policy.py` L283-L370
[TINKER@tinker_cookbook/distillation/train_on_policy.py#L283-L370](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/distillation/train_on_policy.py#L283-L370)

每个 batch：用当前学生的 `sampling_client` 做 `do_group_rollout_and_filter_constant_reward(..., do_remove_constant_reward_groups=False)`（注意这里**不能**过滤"组内奖励全相同"的组，因为奖励全是 0，过滤就什么都不剩了）→ `do_train_step_and_get_sampling_client` → 拿回更新后权重对应的新 `sampling_client`，下一轮用它采样（这就是"权重同步"在 Tinker 里的样子）。

**第 4 步：核心** `incorporate_kl_penalty`，L54-L130
[TINKER@tinker_cookbook/distillation/train_on_policy.py#L54-L130](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/distillation/train_on_policy.py#L54-L130)

```python
full_sequence_inputs_D = [
    datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
    for datum in data_D
]                                                                       # L74
teacher_logprobs_D = await asyncio.gather(
    *[
        teacher_client.compute_logprobs_async(sequence_input)
        for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
    ]
)                                                                       # L80
sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
reverse_kl = [
    (sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask
    for teacher_logprobs, sampled_logprobs, mask in safezip(...)
]                                                                       # L91
for i, datum in enumerate(data_D):
    # The advantage is the negative reverse KL. We can optionally apply a discount factor.
    kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]  # L103
    if kl_discount_factor > 0:
        kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
    datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
        datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
    )
```
逐行解释：
- L74：`model_input` 是"输入序列"（右移前），`target_tokens` 是"目标序列"（左移一位）。把最后一个 target 补到 input 末尾，就得到**完整序列**，老师对它做一次前向即可得到每个位置的 logprob。
- L80：老师只做**打分（prefill）**，不生成。所有样本并发请求。`teacher_clients_D` 允许每个数据集用不同老师（多老师蒸馏，见 `recipes/distillation/on_policy_multi_teacher.py`）。
- L91：`teacher_logprobs[1:]`，老师返回的第 0 个位置是第一个 token（没有前文，不对应任何预测），去掉它后才和 `sampled_logprobs`（每个 target token 一个值）逐位对齐。`sampled_logprobs` 是**采样时**学生给出的 logprob，所以 `c_t = log p_sampler - log q`。
- L103：advantage = 负的 reverse KL；乘 mask 保证 prompt 部分为 0。
- 最后用 `loss_fn="importance_sampling"` 训练。它在 Tinker 服务端实现（不在本仓库），所需输入是 `target_tokens, logprobs, advantages`（见 `tutorials/202_loss_functions.py` L86-L100 的表）。语义上就是 5.3 节 `opd_pg_loss` 里的 `-(exp(logp_θ - logp_sampler) * A).sum()`。

**对照：离线蒸馏**。同目录 `train_off_policy.py` 文件头 L1-L30 把两者讲得很清楚：off-policy 在固定数据上，用老师 top-k 概率作为软标签做 `cross_entropy`（forward KL），"No rollouts, no sampling from the student."；`recipes/distillation/off_policy_reasoning.py` 则是最朴素的"在老师写的推理轨迹（OpenThoughts3）上做 SFT"。
[TINKER@tinker_cookbook/distillation/train_off_policy.py#L1-L30](https://github.com/thinking-machines-lab/tinker-cookbook/blob/70767c738ba5f523b66210a70e50b0dc14c9fc1a/tinker_cookbook/distillation/train_off_policy.py#L1-L30)

### 5.5 源码导读 B：verl 的 on-policy distillation（同时支持两条路线）

示例脚本在 `examples/on_policy_distillation_trainer/`（README 列了 `run_qwen3_8b_fsdp.sh` 等），关键开关是 `distillation.enabled=True`、`distillation.distillation_loss.loss_mode`、`distillation.distillation_loss.use_policy_gradient`。

**第 1 步：配置与语义** `verl/workers/config/distillation.py` `DistillationLossConfig`，L32-L125
[VERL@verl/workers/config/distillation.py#L32-L125](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/workers/config/distillation.py#L32-L125)

docstring 原文：`use_policy_gradient` = "Whether to incorporate distillation loss as a reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1. Otherwise, distillation loss is directly backpropagated as a supervised loss, as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk."
默认值：`loss_mode="k3"`、`use_policy_gradient=True`、`use_task_rewards=True`、`loss_max_clamp=10.0`。

```python
if not self.use_policy_gradient and self.loss_mode == "k1":
    raise ValueError(
        "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
        " wrt model weights does not depend on teacher log probabilities."
    )
```
这就是 5.3 节末尾提醒的那个坑，verl 直接在配置层面拦截。

**第 2 步：老师怎么给 logprob**。`verl/experimental/teacher_loop/teacher_manager.py`，`_get_teacher_sampling_params` L35-L57 与 `compute_teacher_logprobs_single` L116-L145
[VERL@verl/experimental/teacher_loop/teacher_manager.py#L35-L145](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/experimental/teacher_loop/teacher_manager.py#L35-L145)

```python
return {
    "max_tokens": 1,
    "temperature": 1.0,
    "prompt_logprobs": num_logprobs,     # top-k 模式下为 topk，否则为 0（只要采样 token 本身）
    "detokenize": False,
}
...
teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
```
老师也跑在 vLLM 上，只用"把学生的整条序列当 prompt，取 `prompt_logprobs`"这一功能，`max_tokens=1` 表示几乎不生成。注释还提醒：temperature 对 prompt_logprobs 没有作用，所以强制设为 1.0。

**第 3 步：逐 token reverse KL 估计** `compute_distillation_loss_reverse_kl_estimator`，`verl/trainer/distillation/losses.py` L370-L405
[VERL@verl/trainer/distillation/losses.py#L370-L405](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/distillation/losses.py#L370-L405)

```python
student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
distillation_losses = kl_penalty(
    logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
)
```
直接复用了 4.5 节的 `kl_penalty`：把"ref"换成"teacher"。`k1` 时就是 `log p - log q`，与 tinker 的 `reverse_kl` 完全一致。

**第 4 步：两条路线的分叉** `distillation_loss`，L230-L302
[VERL@verl/trainer/distillation/losses.py#L230-L302](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/trainer/distillation/losses.py#L230-L302)

```python
if loss_config.use_policy_gradient:                                    # L268
    # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)  # "vanilla" = PPO clip
    ...
    distillation_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=-distillation_losses.detach(),                      # L283：advantage = -KL（逐 token）
        response_mask=response_mask,
        ...
    )
else:
    # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
    distillation_loss = agg_loss(loss_mat=distillation_losses, loss_mask=response_mask, ...)
```
- L283 是 OPD 的灵魂：`advantages=-distillation_losses.detach()`。advantage 的形状是 `(bs, resp_len)`，**逐 token**，不像 GRPO 那样一条序列一个值。
- 外层 `distillation_ppo_loss`（L165-L227）负责把它和普通 GRPO/PPO 的任务奖励 loss 组合：`use_task_rewards=False` 时纯蒸馏；为 True 时 `policy_loss + distill_loss * distillation_loss_coef`。
- `forward_kl_topk`（L305-L367）是路线 B：老师返回 top-k logprob，在 logits processor 里算 top-k 上的 forward KL，并记录 student/teacher 在 top-k 上的概率质量（top-k 不归一化，所以 loss 要 `clamp_min(0.0)`）。

### 5.6 源码导读 C：TRL 里的蒸馏 Trainer（路线 B）

TRL 当前**没有**"老师 logprob → advantage → PPO"这种 tinker 式的稳定 trainer（路线 A）。它有：

**(1) `DistillationTrainer`（主线，`trl/trainer/distillation_trainer.py`）**：学生在线生成，然后做**全词表**广义 JSD。
- 类 docstring（L209 起）："The student is trained on-policy — it generates the completions itself — to match the teacher's next-token distribution under a generalized Jensen-Shannon divergence"。
- 配置里 `beta` 默认 **1.0 = reverse KL**（`distillation_config.py` L132-L136），并特别说明它不是 GRPO 里那个 KL 惩罚系数，而是选择散度类型。
- `_compute_loss` L1771-L1875：拿到学生和老师的**最后一层 hidden state**（不是 logits），再交给 `_chunked_divergence_loss`（L99-L206），由 Triton kernel `ChunkedDivergenceFunction` 分块投影到词表、算散度，从不物化 `(B, T, V)` 的 logits。
  [TRL@trl/trainer/distillation_trainer.py#L99-L206](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/distillation_trainer.py#L99-L206) ·
  [TRL@trl/trainer/distillation_trainer.py#L1771-L1875](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/distillation_trainer.py#L1771-L1875)
  ```python
  valid = completion_mask.reshape(-1) != 0
  h_s = student_hidden_states.reshape(-1, student_hidden_states.size(-1))[valid]
  h_t = teacher_hidden_states.reshape(-1, teacher_hidden_states.size(-1))[valid]
  per_token_divergence, per_token_entropy = ChunkedDivergenceFunction.apply(
      h_s, student_lm_head_weight, student_lm_head_bias,
      h_t, teacher_lm_head_weight, teacher_lm_head_bias, beta, temperature, ...)
  ```
  注释写明老师和学生 hidden 维可以不同，只要词表相同。

**(2) `GKDTrainer`（`trl/experimental/gkd/gkd_trainer.py`，GKD 论文 2306.13649 的直接实现，代码最短，适合学习）**

- 广义 JSD `generalized_jsd_loss`，L238-L322：
  [TRL@trl/experimental/gkd/gkd_trainer.py#L238-L322](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/experimental/gkd/gkd_trainer.py#L238-L322)
  ```python
  student_log_probs = F.log_softmax(student_logits, dim=-1)
  teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
  if beta == 0:                                                         # L278
      jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
  elif beta == 1:                                                       # L280
      jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
  else:
      mixture_log_probs = torch.logsumexp(
          torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
      kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
      kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
      jsd = beta * kl_teacher + (1 - beta) * kl_student
  ```
  **读这段最容易绕晕的地方**：`F.kl_div(input, target)` 计算的是 `KL(target || input)`，参数顺序和数学写法是反的。所以：
  - `beta == 0`：`kl_div(student, teacher)` = `KL(teacher || student)` = **forward KL**（离线 KD 那种）。
  - `beta == 1`：`kl_div(teacher, student)` = `KL(student || teacher)` = **reverse KL**（OPD 那种）。
  - 中间值：对混合分布 `m = β·q + (1-β)·p` 的两个 KL 加权，即广义 JSD。用 `logsumexp` 在 log 空间里算混合分布，避免下溢。
  - 注意这里返回的是**全词表**逐元素 KL，`(B, T, V)`，所以 `jsd[mask]` 之后 `sum / mask.sum()` 实际是"每个 token 的 KL 之和的平均"。
- 什么时候是 on-policy：`training_step`，L406-L446
  [TRL@trl/experimental/gkd/gkd_trainer.py#L406-L446](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/experimental/gkd/gkd_trainer.py#L406-L446)
  ```python
  if random.random() <= self.lmbda:          # 以概率 lmbda 用学生自己生成的样本（on-policy）
      ... self.generate_on_policy_outputs(unwrapped_model, inputs, self.generation_config)
  elif self.seq_kd:                          # 否则可选：用老师生成（sequence-level KD）
      ... self.generate_on_policy_outputs(self.teacher_model, ...)
  # 都不满足时：用数据集里原有的回答（纯 off-policy）
  ```
  `lmbda` 把 off-policy 和 on-policy 统一在一个旋钮里：`lmbda=0, beta=0` 是经典 KD（forward KL on dataset）；`lmbda=1, beta=1` 就是"学生采样 + 全词表 reverse KL"的 on-policy distillation。
- `generate_on_policy_outputs` L364-L404：把 prompt 部分和第一个 EOS 之后的位置的 label 置 `-100`，让 loss 只落在学生生成的 completion 上。
- `compute_loss` L324-L361：学生、老师在**同一条序列**上各前向一次，取 shift 后的 logits 送入 JSD。

**(3) 其他**：`trl/experimental/iw_opd/iw_opd_trainer.py` L349 起的 `IWOPDTrainer`，docstring 说明它是"pre-refactor DistillationTrainer 的冻结快照"，保存了 sampled-token 的 IW-OPD 目标（对采样 token 的蒸馏信号按前缀累积偏差重加权）以及 vLLM 外部老师支持；`experimental/` 下还有 `async_distillation`、`server_distillation`、`gold`、`sdpo`、`sdft` 等变体，本章不展开。

**OpenRLHF**：在本 commit 里 `grep -ri distill openrlhf` 没有任何结果，**没有 on-policy distillation 的实现**。

### 5.7 离线 KD vs 在线 OPD 一图看懂

| | 离线：SFT on teacher outputs | 离线：logit KD（forward KL） | 在线：OPD（reverse KL） |
|---|---|---|---|
| 训练序列从哪来 | 老师生成（或人写）的固定数据 | 固定数据 | **学生当前策略现采样** |
| 每个位置的监督 | 老师 token 的 one-hot | 老师整行分布 `q(·|prefix)` | 老师对学生 token 的 `log q(y_t)`（或整行） |
| 目标 | `-log p(y^T_t)` | `KL(q ‖ p)`，mode-covering | `KL(p ‖ q)`，mode-seeking |
| 训练分布 vs 推理分布 | 不一致（exposure bias） | 不一致 | **一致** |
| 需要老师做什么 | 一次性生成数据 | 每个位置的 logits（离线可缓存） | 每步对学生样本打分（prefill，比生成便宜） |
| 需要学生采样 | 否 | 否 | 是（贵，需要推理引擎） |

---

## 6. 工程：logprob 的高效计算、为什么 rollout 要用推理引擎、权重如何同步

### 6.1 logits → log_softmax → gather 的三种做法

**(a) 朴素**：`log_softmax(logits).gather(...)`，物化 `(B, L, V)` 的 fp32，显存最大。verl 的 `logprobs_from_logits_naive` 就是这个。

**(b) 逐行 logsumexp**：利用 `log_softmax(x)_i = x_i - logsumexp(x)`，只 gather 目标 logit，再按行循环算 logsumexp，避免一次性生成 `(B, L, V)` 的 log_softmax 张量。
- TRL `selective_log_softmax`，`trl/trainer/utils.py` L540-L603
  [TRL@trl/trainer/utils.py#L540-L603](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/utils.py#L540-L603)
  ```python
  if logits.dtype in [torch.float32, torch.float64]:
      selected_logits = torch.gather(logits, dim=-1, index=index)
      # loop to reduce peak mem consumption
      logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
      per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)
  else:
      # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
      for row_logits, row_labels in zip(logits, index, strict=True):
          row_logps = F.log_softmax(row_logits, dim=-1)
  ```
  bf16 下 `x_i - logsumexp(x)` 两个大数相减会丢精度，所以回退到逐行 log_softmax。
- verl 对应 `logprobs_from_logits_v2`（`verl/utils/torch_functional.py` L173 起），以及优先使用 flash-attn 的 Triton cross-entropy kernel 的 `logprobs_from_logits`（L73-L107）。后者注释提醒：该 kernel 的归约可能不确定，需要逐位可复现时设 `VERL_DISABLE_FLASH_ATTN_CE=1`。
  [VERL@verl/utils/torch_functional.py#L73-L107](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/utils/torch_functional.py#L73-L107)
- 熵：TRL `entropy_from_logits`（L606-L645）按 `chunk_size=128` 行分块；verl `entropy_from_logits` 用 `logsumexp(logits) - Σ softmax·logits`（L231-L245）。

**(c) 不物化 logits：fused LM head（TRL 当前做法）**。`add_fused_lm_head`，`trl/trainer/utils.py` L1550-L1701
[TRL@trl/trainer/utils.py#L1550-L1701](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/trainer/utils.py#L1550-L1701)

```python
backbone_outputs = getattr(self, backbone_attr)(*args, **kwargs)   # 只跑 transformer 主体，不过 lm_head
if shift_labels is None:
    hidden_states = backbone_outputs.last_hidden_state[:, :-1]    # L1626：在 hidden 上做 shift
    labels = labels[:, 1:]
...
mask = labels != -100                                              # L1635
per_token = _ChunkedLogProbFunction.apply(
    hidden_states[mask], weight, bias, labels[mask], temperature, ...)   # L1649：只投影有效位置
log_probs, entropy, ... = (
    None if x is None else x.new_zeros(mask.shape).masked_scatter(mask, x) for x in per_token)  # L1663
loss = -log_probs.sum() / (mask.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch)
...
model.forward = types.MethodType(_fused_forward, model)            # L1701：猴子补丁
```
- **只对 label 不为 -100 的位置**做 `hidden @ W_vocab^T`。GRPO 的 prompt 往往很长，这一步就省掉了大部分计算。
- `ChunkedLogProbFunction`（`trl/kernels/chunked_logprob.py` L178 起）按 `[4096 token × chunk_size 词表]` 的 tile 用 cuBLAS 做投影，再用 Triton kernel 维护**在线 logsumexp**（`running_max`、`sum_exp`），最后 `log_z = running_max + torch.log(sum_exp)`（L256）、`logprobs = target_logit - log_z`。反向时重算 tile。
  [TRL@trl/kernels/chunked_logprob.py#L178-L262](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/kernels/chunked_logprob.py#L178-L262)
  这和 FlashAttention 的 online softmax 是**同一个技巧**（本仓库 `flash_attn/cute/softmax.py` 里的 row_max/row_sum 追踪）：对一个太长、放不进片上存储的维度做 softmax 时，分块扫描，边扫边更新 max 和 sum。
- 文件开头的 `_addmm_fp32` 注释点出一个细节：每个词表块的乘积如果先舍入到 bf16 再累加，训练会被系统性地带偏（"entropy drifts lower"），所以累加要在 fp32 里做。

### 6.2 为什么 rollout 要交给 vLLM / SGLang

在线算法（PPO/GRPO/OPD）的时间大头往往是**生成**而不是反向传播：
- 自回归生成每步只算 1 个 token，是严重的 memory-bound；HF `generate` 没有 paged KV cache、continuous batching、CUDA Graph，也没有为解码优化的 attention kernel。GRPO 每个 prompt 还要生成 G 份，长度参差，静态 batch 会浪费大量算力在 padding 上。
- 推理引擎（vLLM/SGLang）用 PagedAttention 管理 KV cache、动态拼 batch，吞吐能高一个数量级，而且原生支持 prefix caching：同一 prompt 的 G 次采样共享 prompt 的 KV。
- 代价：**两份权重**（训练框架的 FSDP/Megatron 分片 + 推理引擎的 TP 分片），以及两套数值实现带来的 **logprob 不一致**，需要 6.4 节的修正。

### 6.3 权重同步

**TRL**：`VLLMGeneration.sync_weights`，`trl/generation/vllm_generation.py` L481-L517；参数遍历 `_iter_named_params` L435-L479
[TRL@trl/generation/vllm_generation.py#L435-L517](https://github.com/huggingface/trl/blob/904125670a7f41b1d5f9b61b946f90644a4ca72f/trl/generation/vllm_generation.py#L435-L517)

```python
if self.mode == "server":
    if accelerator.is_main_process:
        self.vllm_client.update_named_params(self._weight_metadata, self._iter_named_params())
    else:
        for _ in self._iter_named_params():  # take part in the gather collectives
            pass
elif self.mode == "colocate":
    for name, param in self._iter_named_params():
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])
...
self.llm.reset_prefix_cache()
```
- `server` 模式：vLLM 是独立进程/机器，主进程把参数通过 NCCL 广播过去（端口 `vllm_group_port`，`grpo_config.py` L642）。
- `colocate` 模式：vLLM 和训练在同一 GPU 上，直接调用 `load_weights` 原地覆盖；用 sleep mode 在训练时释放 vLLM 的显存。
- `_iter_named_params` 处理三种麻烦：FSDP/ZeRO-3 下参数是分片的，要**逐个** gather（gather 是集体通信，所以非主进程也必须参与遍历，即上面的空循环）；LoRA 要先 `merge_adapter()` 再导出，导出后 `unmerge_adapter()`；参数名要改成 vLLM 的命名。
- 同步完必须 `reset_prefix_cache()`：旧权重算出的 KV cache 已经失效。

**verl**：`CheckpointEngineManager.update_weights`，`verl/checkpoint_engine/base.py` L510-L562
[VERL@verl/checkpoint_engine/base.py#L467-L562](https://github.com/volcengine/verl/blob/76e837e0bfcb18b380a66c0ba56ef9f0ac840f09/verl/checkpoint_engine/base.py#L467-L562)

代码里编号的 8 步注释就是完整流程：colocate 场景走 `naive` 后端直接在进程内同步；分离部署时依次 ① abort 在途请求（partial rollout 会保存下来）② 组建临时 worker group ③ 释放 rollout 的 KV cache（权重留在原位）④ 建通信组 ⑤ actor 发、rollout 收 ⑥ finalize ⑦ 恢复 KV cache ⑧ 恢复生成。配合 `sleep_replicas()`（L467-L470，"free weight and kv_cache device memory"），在 colocate 时实现训练和推理分时复用同一批 GPU。
worker 侧逻辑见 `verl/workers/engine_workers.py` L735 起的 `update_weights`（docstring 解释了 `naive` 与 checkpoint-engine 两种模式以及 LoRA 处理）。

### 6.4 训推不一致（training-inference mismatch）

即使权重刚同步完，vLLM 算出的 `log π_vllm(y_t)` 也不等于训练框架算出的 `log π_train(y_t)`。PPO 的 ratio 用的是 `π_old`（训练侧重算的），但样本实际来自 `π_vllm`，严格来说这已经是 off-policy。三家的处理：
- TRL：`vllm_importance_sampling_correction`，在 L2601-L2648 按 token/sequence 级、截断或 mask 计算 `exp(old - sampling)`（L2615），在 loss 里乘上（L3088 `per_token_loss * inputs["importance_sampling_ratio"]`）。
- verl：`rollout_corr_helper.compute_rollout_correction_and_add_to_batch`（`ray_trainer.py` L1658），以及 policy loss 的 `rollout_is_weights` 参数。
- OpenRLHF：`PolicyLoss` 的 `is_correction_level/mode/gating`，注释引用了 "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"。
- tinker：`importance_sampling` loss 本身就以采样时的 `logprobs` 为分母。

---

## 7. 算法对比表

| | SFT | PPO (RLHF) | DPO | GRPO | OPD |
|---|---|---|---|---|---|
| 数据 | (prompt, response) 示范 | prompt + 奖励模型 | 偏好对 (x, y_w, y_l) | prompt + 可验证奖励 / RM | prompt + 老师模型 |
| 是否在线采样 | 否 | 是 | 否（Online DPO 等变体除外） | 是，每 prompt G 条 | 是（学生采样） |
| 需要的模型 | policy | policy、ref、RM、critic | policy、ref（可预计算或用关掉 LoRA 的 base） | policy、（ref，β=0 时不需要）、奖励函数 | student、teacher（只做 prefill） |
| 每 token 的信号 | 常数 1（模仿） | GAE advantage（逐 token，靠 critic） | 序列级标量 `σ(-βΔ)`，均匀摊到 token | 序列级 advantage，广播到每个 token | **逐 token** `-(log p - log q)` |
| 信号密度 | 稠密（但只是模仿） | 名义上逐 token，实际依赖 critic 质量 | 每对 1 bit | 每条回答约 1 个标量 | 每个 token 一个实数，最稠密 |
| 主要成本 | 1 次前向 + 反向 | 生成 + 4 个模型前向 + 2 个模型反向 | 2×（chosen+rejected）前向，ref 可省 | 生成 G 倍 + old/ref 前向 + 反向 | 生成 + 老师 prefill + 反向（不需要 G 很大） |
| 典型坑 | mask 错、packing 串样本 | 不稳定、reward hacking | 长度偏好、chosen logp 也下降 | 长度偏置、零方差组、训推不一致 | 词表/模板不匹配、k1 直接反传无效 |

---

## 8. 常见困惑 / 坑

1. **shift 一位**。logits 在位置 t 预测 t+1。DPO、GRPO 里 `completion_mask[..., 1:]`、`outputs.log_probs[:, -seq_len:]`、tinker 里 `teacher_logprobs[1:]` 都是这个原因。对不齐时 loss 照样能下降，只是训练效果莫名变差，很难发现。
2. **DPO 用 sum 而不是 mean**。推导里的 `log π(y|x)` 是整句概率，所以是求和。改成 mean 就变成另一个目标（SimPO 风格），β 的量级也随之变化（TRL 的 IPO 分支专门做了长度归一化并写了注释说明原因）。
3. **DPO 训练中 chosen 的 logp 也在下降**。loss 只约束 margin，两者可以一起掉，只是 rejected 掉得更快。看 `logps/chosen` 曲线；可以在 TRL 里用 `loss_type=["sigmoid","sft"]` 加一点 SFT 项（`sft` 分支 L1485）。
4. **ref 模型必须冻结、要用 eval 模式**。dropout 打开时，同一输入两次前向结果不同，log ratio 就带噪声。TRL 在 ref 前向时用 `torch.no_grad()` 并临时关闭 gradient checkpointing。
5. **GRPO 的零方差组**。一组全对或全错时 `A ≡ 0`，这个 prompt 完全不贡献梯度，却花了 G 次生成。DAPO 的 dynamic sampling 会把这类组过滤掉再补采；TRL 用 `is_std_zero` 记录比例。tinker 的 `do_group_rollout_and_filter_constant_reward` 就是做这个过滤的（OPD 里必须关掉，因为奖励恒为 0）。
6. **`scale_rewards`/std 归一化的偏置**。除以组 std 会放大"几乎全对/全错"的题的权重（std 很小），Dr. GRPO 建议关掉。
7. **loss 聚合决定了长度偏好**。原版 GRPO 的 "seq-mean-token-mean" 会让模型倾向于在答错时写得更长（惩罚被稀释）。TRL 默认已改为 `dapo`。
8. **`num_iterations=1` 时 clip 根本没起作用**。ratio 恒为 1，clip 只有在同一批数据多次更新（或训推不一致）时才有意义。
9. **k3 是 `KL(π_θ‖π_ref)` 的估计**，前提是样本来自 π_θ。样本来自 π_old 而 π_θ 已经变化时，严格讲需要 IS 修正（TRL 的 `use_bias_correction_kl` 会乘 `coef_1`）。另外，k1/k3 的**梯度**不是 KL 梯度的无偏估计（verl `kl_penalty` 的 docstring）。
10. **OPD 里不能把 `log p_θ - log q` 当 loss 直接反传**。梯度是 `∇log p_θ(y)`，与老师无关。必须 detach 后当 advantage（路线 A），或者用全词表/ top-k 的解析 KL（路线 B）。verl 会直接报错。
11. **OPD 的老师和学生必须对齐 token 序列**。tinker 注释提醒："if your teacher has a different renderer than the student, you may want to modify the full_sequence_inputs_D"。路线 B 还要求词表完全一致（TRL `DistillationTrainer` 要求 "It must share the student's vocabulary"）。
12. **`F.kl_div` 的参数顺序**是 `kl_div(input=log p, target=log q) = KL(q‖p)`，与直觉相反。读 GKD 时 `beta=0` 是 forward KL、`beta=1` 是 reverse KL。
13. **权重同步后忘记清 prefix cache**，下一轮的 rollout 会复用旧权重算出的 KV，得到"半新半旧"的样本。
14. **分布式下的分母**。用本地 token 数做分母、再被 DDP 平均，等于"各卡平均的平均"，不等于全局 token 平均。verl 的 `agg_loss` 用全局 `batch_num_tokens * dp_size`，TRL 的 `dapo` 用 `num_items_in_batch`，都是在处理这个问题。

---

## 9. 自测题

1. 在 DPO 推导中，为什么 `β log Z(x)` 能被消掉？如果偏好数据里 chosen 和 rejected 来自**不同的 prompt**，DPO 还成立吗？
2. TRL 的 `DataCollatorForPreference` 把 chosen 和 rejected 拼在 batch 维。`_compute_loss` 里哪一行把它们拆开？如果 collator 改成交错排列（c0, r0, c1, r1, ...），需要改哪几行？
3. 设 `num_generations=4`，某组奖励为 `[1, 1, 1, 0]`。分别写出 `scale_rewards="group"` 与 `"none"` 下的 advantage（TRL 的 `nanstd` 用无偏估计）。哪条回答的 advantage 绝对值最大？为什么这可能是个问题？
4. 证明 k3 = `(u - 1) - log u`（u = π_ref/π_θ）在 `y ~ π_θ` 下是 `KL(π_θ‖π_ref)` 的无偏估计，并说明它为什么恒 ≥ 0。
5. 一个 batch 有两条回答：长度 10 的 loss 每 token 都是 1，长度 100 的每 token 都是 1。分别用 `grpo`、`bnpo`、`dr_grpo`（`max_completion_length=100`）计算 loss，并说明每种方式下长回答里的单个 token 权重。
6. 在 tinker 的 `incorporate_kl_penalty` 里，为什么要 `teacher_logprobs[1:]`？`sampled_logprobs` 为什么用采样时的 logprob 而不是重新算的？
7. 在 verl 的 OPD 里，`use_policy_gradient=True, loss_mode="k1"` 与 `use_policy_gradient=False, loss_mode="k3"` 在数学上分别对应哪种目标、哪种梯度估计？为什么前者被推荐用 k1？
8. GKD 中 `lmbda=0, beta=0`、`lmbda=1, beta=1`、`lmbda=0, seq_kd=True` 分别对应什么经典方法？
9. 为什么 TRL 的 fused LM head 要在 `hidden_states[mask]` 上投影，而不是先算完整 logits 再 mask？对 GRPO 典型的"长 prompt + 短 completion"，大约能省多少？
10. 为什么 GRPO + vLLM 时，即使 `num_iterations=1`，TRL 仍然要计算 `old_per_token_logps`？

## 10. 动手练习

1. **验证 per-token logp**：用任意小模型（例如 `Qwen/Qwen2.5-0.5B`），对一条 prompt+completion 用 0.1 节的 `per_token_logps` 算出序列 logp，和 `model(input_ids, labels=labels).loss * N`（labels 中 prompt 置 -100）对比，两者应满足 `seq_logp == -loss * N`（N 为有效 token 数）。
2. **DPO 梯度直觉**：固定 ref logp，手工构造 `chosen_logratios - rejected_logratios ∈ {-5, 0, 5}`，用 `torch.autograd.grad` 观察 `dpo_loss` 对 margin 的梯度，验证它等于 `-β σ(-βΔ)`。
3. **GRPO 聚合实验**：用 4.3 节的 `grpo_loss`，构造 2 组不同长度的回答，比较三种 `agg` 对"长错误回答"梯度的大小，复现第 5 题的结论。
4. **读 trl 测试**：`repos/08-rl-alignment/trl/tests/experimental/test_gkd_trainer.py`（L262-L282 附近对 `beta=0`、`beta=1` 调用了 `generalized_jsd_loss`）或 `tests/test_distillation_trainer.py`，找出验证 `beta=0/1` 时 JSD 退化为 forward/reverse KL 的测试，并用 5.6 节（以及坑 12）关于 `F.kl_div` 参数顺序的结论解释断言。
5. **OPD 玩具实验（无需 GPU）**：设词表大小 5，老师分布 `q` 固定（例如 `[0.6, 0.3, 0.05, 0.03, 0.02]`），学生是一个可学习的 logits 向量。分别用
   (a) forward KL `KL(q‖p)` 解析梯度；
   (b) 路线 A：从 p 采样 64 个 token，`adv = -(log p - log q).detach()`，loss = `-(adv * log p).mean()`；
   (c) 路线 B：解析 reverse KL `KL(p‖q)`；
   各训练 200 步，画出 p 的演化。再把 q 改成双峰（`[0.45, 0.05, 0.45, 0.03, 0.02]`），观察 mode-covering 与 mode-seeking 的差别。
6. **追踪一次 verl step**：在 `ray_trainer.py` 的 `fit` 中，按 4.5 节表格的 8 个阶段，记下每个阶段之后 `batch.batch.keys()` 新增了哪些 key（`old_log_probs`、`ref_log_prob`、`token_level_scores`、`token_level_rewards`、`advantages`、`returns` ……），画出数据流图。

---

### 附：本章引用的主要入口速查

| 主题 | 文件 : 符号 | 行号 |
|---|---|---|
| SFT label mask | `trl/trainer/sft_trainer.py` : `build_labels` | L1396-L1423 |
| SFT padding-free | `trl/trainer/sft_trainer.py` : `DataCollatorForLanguageModeling.torch_call` | L168-L227 |
| DPO 拼接 | `trl/trainer/dpo_trainer.py` : `DataCollatorForPreference.torch_call` | L146-L207 |
| DPO loss | `trl/trainer/dpo_trainer.py` : `DPOTrainer._compute_loss` | L1275-L1602 |
| DPO ref 预计算 | `trl/trainer/dpo_trainer.py` : `_precompute_ref_logps` / `compute_ref_log_probs` | L1158-L1273 |
| 经典 get_batch_logps | `trl/experimental/cpo/cpo_trainer.py` : `get_batch_logps` | L752-L790 |
| fused LM head | `trl/trainer/utils.py` : `add_fused_lm_head` | L1550-L1701 |
| 分块 logp kernel | `trl/kernels/chunked_logprob.py` : `ChunkedLogProbFunction` | L178 起 |
| GRPO 采样分组 | `trl/trainer/grpo_trainer.py` : `_get_train_sampler` | L1253-L1290 |
| GRPO advantage | `trl/trainer/grpo_trainer.py` : `_generate_and_score_completions` | L2693-L2751 |
| GRPO loss | `trl/trainer/grpo_trainer.py` : `_compute_loss` | L2961-L3231 |
| vLLM 权重同步 | `trl/generation/vllm_generation.py` : `sync_weights` | L481-L517 |
| verl GAE / GRPO | `verl/trainer/ppo/core_algos.py` : `compute_gae_advantage_return` / `compute_grpo_outcome_advantage` | L215-L263 / L267-L331 |
| verl PPO loss / 聚合 / KL | `core_algos.py` : `compute_policy_loss_vanilla` / `agg_loss` / `kl_penalty_forward` | L1285-L1376 / L1140-L1206 / L2216-L2251 |
| verl 主循环 | `verl/trainer/ppo/ray_trainer.py` : `RayPPOTrainer.fit` | L1405 起 |
| verl 权重同步 | `verl/checkpoint_engine/base.py` : `CheckpointEngineManager.update_weights` | L510-L562 |
| OpenRLHF GRPO | `openrlhf/trainer/ppo_utils/experience_maker.py`（`group_norm`） | L256-L281 |
| **OPD（tinker）** | `tinker_cookbook/distillation/train_on_policy.py` : `incorporate_kl_penalty` | **L54-L130** |
| **OPD（verl）** | `verl/trainer/distillation/losses.py` : `distillation_loss`（`advantages=-distillation_losses.detach()`） | **L230-L302**（L283） |
| OPD 老师打分（verl） | `verl/experimental/teacher_loop/teacher_manager.py` : `compute_teacher_logprobs_single` | L116-L145 |
| 在线蒸馏 JSD（TRL） | `trl/trainer/distillation_trainer.py` : `_chunked_divergence_loss` / `_compute_loss` | L99-L206 / L1771-L1875 |
| GKD | `trl/experimental/gkd/gkd_trainer.py` : `generalized_jsd_loss` / `training_step` | L238-L322 / L406-L446 |
