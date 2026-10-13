# 14 · 分布式训练（DP / ZeRO / FSDP / TP / SP / CP / PP / EP）源码导读

> 读完本篇你应该能回答：一个 70B 模型训练时，每张卡上到底放了哪些东西、各占多少字节？ZeRO 的 1/2/3 分别切了什么、通信在什么时刻发生？Megatron 的 ColumnParallelLinear / RowParallelLinear 为什么要成对出现、all-reduce 藏在哪里？1F1B 调度为什么比 GPipe 省显存？ring attention 怎么把超长序列切到多卡上还算得精确？

本文引用的代码版本（行号均已用 `grep -n` / `sed -n` 在子模块中核对）：

| 子模块 | 仓库 | commit |
|---|---|---|
| `repos/13-distributed/megatron-lm` | NVIDIA/Megatron-LM | `3bb482c487cd447a2b40a3bf076ab14cad514e13` |
| `repos/13-distributed/deepspeed` | deepspeedai/DeepSpeed | `dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c` |
| `repos/13-distributed/torchtitan` | pytorch/torchtitan | `15f5f1a91b39592d2b567da7c8d6c0a6d5f89487` |
| `repos/07-finetune/accelerate` | huggingface/accelerate | `ccfdff7dbbd6a8c455853a9e5eb6396ea51ffd45` |

---

## 1. 要解决的问题 / 一句话直觉

单卡训练大模型有两堵墙：

1. **显存墙**：参数、梯度、优化器状态、激活值放不下。
2. **算力墙**：放得下也算不完（一次预训练 ~10²⁴ FLOPs）。

所有并行策略都在回答同一个问题：**把哪个维度切开、分给哪些卡、切开之后需要什么通信把结果拼回来？**

| 切什么 | 叫什么 | 一句话直觉 |
|---|---|---|
| batch | 数据并行 DP / DDP | 每卡一份完整模型，各算各的样本，**梯度求平均** |
| 优化器状态 / 梯度 / 参数（沿"存储"维度） | ZeRO-1/2/3、FSDP | 仍是数据并行，但"每卡只保管 1/N 的状态，用时再借" |
| 权重矩阵的行/列 | 张量并行 TP | 一个矩阵乘拆到多卡，**层内**通信 |
| 序列维（LayerNorm/Dropout 区域） | 序列并行 SP | 配合 TP，把 TP 里冗余的激活也切掉 |
| 序列维（attention 本身） | 上下文并行 CP / ring attention / Ulysses | 超长序列，每卡一段，KV 在卡间流转 |
| 层 | 流水线并行 PP | 每卡几层，micro-batch 像流水线一样流过 |
| MoE 专家 | 专家并行 EP | 每卡几个专家，token all-to-all 派发（见 `guide/07-moe.md`） |

实际训练通常是 **3D/4D/5D 组合**：例如 TP=8（机内 NVLink）× PP=8（跨机）× DP=N，再加 CP、EP。

---

## 2. 核心原理

### 2.1 显存账：每个参数 16 字节

设参数量 Ψ，混合精度 + Adam：

| 内容 | dtype | 字节/参数 |
|---|---|---|
| 参数（前向/反向用） | bf16/fp16 | 2 |
| 梯度 | bf16/fp16 | 2 |
| fp32 主权重（master weights） | fp32 | 4 |
| Adam 一阶矩 m | fp32 | 4 |
| Adam 二阶矩 v | fp32 | 4 |
| **合计** | | **16**（其中优化器相关 K=12） |

7.5B 模型 → 120 GB，还没算激活。这就是为什么单卡 80GB 训不了 7B（全参 Adam）。

**ZeRO 的三档切分**（N 为数据并行度）：

| | 每卡显存（字节/参数） | Ψ=7.5B, N=64 |
|---|---|---|
| DDP | 2 + 2 + 12 = 16 | 120 GB |
| ZeRO-1（切优化器状态） | 2 + 2 + 12/N | 31.4 GB |
| ZeRO-2（再切梯度） | 2 + (2+12)/N | 16.6 GB |
| ZeRO-3 / FSDP（再切参数） | 16/N | 1.9 GB |

### 2.2 通信原语与通信量

设 N 卡，每卡持有大小为 Ψ（元素数）的张量。ring 算法下每卡发送/接收的数据量：

| 原语 | 语义 | 每卡通信量 |
|---|---|---|
| all-reduce | 每卡得到 Σ_r x_r | 2Ψ(N−1)/N ≈ 2Ψ |
| reduce-scatter | 卡 k 得到 (Σ_r x_r) 的第 k 段 | Ψ(N−1)/N ≈ Ψ |
| all-gather | 每卡贡献一段，大家拿到拼接 | Ψ(N−1)/N ≈ Ψ |
| all-to-all | 卡 i 的第 j 段发给卡 j | Ψ(N−1)/N |

关键恒等式：**all-reduce = reduce-scatter + all-gather**。ZeRO 就是把这两步拆开，在中间插入"只更新自己那 1/N"：

- DDP：all-reduce 梯度（2Ψ）→ 每卡完整 Adam 更新。
- ZeRO-1/2：reduce-scatter 梯度（Ψ）→ 每卡只更新自己的 1/N 参数 → all-gather 参数（Ψ）。**总量 2Ψ，与 DDP 相同**，但优化器状态只需 1/N。
- ZeRO-3/FSDP：前向时逐层 all-gather 参数（Ψ），反向时再 all-gather 一次（Ψ，因为前向后已释放），反向后 reduce-scatter 梯度（Ψ）。**总量 3Ψ，是 DDP 的 1.5 倍**，换来参数也只需 1/N。

时序（ZeRO-3/FSDP，以一层为单位）：

```
forward:  [all-gather W_l] -> compute layer l -> [free W_l (reshard)] -> 下一层（预取 W_{l+1} 与计算重叠）
backward: [all-gather W_l] -> compute dX, dW_l -> [reduce-scatter dW_l] -> [free W_l]
step:     每卡用自己的 1/N 梯度更新自己的 1/N fp32 主权重和 m/v（无通信）
```

### 2.3 张量并行：列切 + 行切，一个 all-reduce

以 MLP `Y = GeLU(X A) B` 为例，`X: [s·b, h]`，`A: [h, 4h]`，`B: [4h, h]`，TP 度为 t：

- **列切 A**：`A = [A_1, …, A_t]`，`A_i: [h, 4h/t]`。每卡算 `Z_i = GeLU(X A_i)`，形状 `[s·b, 4h/t]`。因为 GeLU 是逐元素的，`GeLU(X A)` 按列切开后可以各自算——**不需要通信**。
- **行切 B**：`B = [B_1; …; B_t]`，`B_i: [4h/t, h]`。每卡算 `Y_i = Z_i B_i`，形状 `[s·b, h]`，是**部分和**：`Y = Σ_i Y_i`。
- 一个 **all-reduce** 得到完整 Y。

Megatron 把这两个点抽象成一对共轭算子：

| 算子 | 前向 | 反向 | 位置 |
|---|---|---|---|
| f（`copy_to_tensor_model_parallel_region`） | identity | all-reduce | 列并行层的输入 |
| g（`reduce_from_tensor_model_parallel_region`） | all-reduce | identity | 行并行层的输出 |

为什么 f 的反向要 all-reduce？因为 X 被 t 张卡各自用了一次（`X A_i`），链式法则要求 `dX = Σ_i dX_i`。

**Attention 同理**：QKV 投影按**头**列切（每卡 `n_heads/t` 个头，各自完整地做 attention，头之间独立），输出投影 `W_o` 行切，末尾一个 all-reduce。所以**每个 Transformer 层前向 2 次 all-reduce、反向 2 次**，每次大小 `s·b·h`。这个通信量与 batch×seq 成正比、发生在每一层的关键路径上，所以 TP 通常限制在 NVLink 连通的单机内（t ≤ 8）。

### 2.4 序列并行（SP）：把 all-reduce 拆成 reduce-scatter + all-gather

TP 区域之外的 LayerNorm、Dropout、残差加法在每张卡上是**重复**计算的，激活也重复存储。SP 把这些区域的激活沿**序列维**切成 t 份：

```
[LN, dropout: X 沿 seq 切, 形状 s/t × b × h]
   --all-gather(seq)-->  [列并行 Linear: 需要完整 X]
   ... 行并行 Linear 输出部分和 ...
   --reduce-scatter(seq)--> [回到 s/t × b × h]
```

all-reduce = reduce-scatter + all-gather，所以通信量不变，但 LN/dropout 的激活显存降到 1/t。

**激活显存**（每层，fp16，Korthikanti et al. 2022；a 为头数）：

| 配置 | 每层激活字节 |
|---|---|
| 无并行 | s·b·h·(34 + 5·a·s/h) |
| TP=t | s·b·h·(10 + 24/t + 5·a·s/(h·t)) |
| TP=t + SP | s·b·h·(34 + 5·a·s/h) / t |
| 全量重计算（activation checkpointing） | 2·s·b·h（只存层输入） |

### 2.5 上下文并行（CP）/ Ring Attention

当序列长到单卡放不下 `s×s` 的注意力或 `s×h` 的激活时，沿序列切 Q/K/V。每卡持有 `Q_r, K_r, V_r`（长度 s/cp），需要算 `softmax(Q_r Kᵀ) V`，其中 K/V 是**全序列**。

- **Ring（P2P）**：K/V 块在 cp 张卡之间沿环传 cp−1 次。每收到一块，就用 **online softmax**（与 FlashAttention 同一个技巧）把局部结果合并进 `(m, l, o)`：
  `m' = max(m, rowmax(S))；o' = o·e^{m−m'} + e^{S−m'} V；l' = l·e^{m−m'} + rowsum(e^{S−m'})`。
  通信（下一块 KV）与计算（当前块 attention）可以重叠。
- **All-gather KV**：直接 all-gather 全部 K/V，再算——简单但不能重叠，显存更高（torchtitan 的 `KVAllGatherCPFlexInnerAttention`）。
- **Ulysses（all-to-all）**：把"按 token 切"转成"按头切"：all-to-all 后每卡拿到**全序列、1/cp 的头**，正常算 attention，再 all-to-all 回去。要求头数能被 cp 整除。
- **负载均衡**：causal mask 下靠后的 token 要看更多 key。把序列切成 `2·cp` 块，卡 r 拿第 r 块和第 `2cp−1−r` 块（zigzag / head-tail），每卡计算量相同。

### 2.6 流水线并行（PP）：1F1B

把 L 层切成 p 段放在 p 张卡上，一个 batch 切成 m 个 micro-batch。

- **GPipe**：先把 m 个 micro-batch 的前向全做完，再做全部反向。气泡比例 `(p−1)/(m+p−1)`；每段要同时缓存 **m** 个 micro-batch 的激活。
- **1F1B**：stage i 先做 `p−i−1` 个 warmup 前向，然后"一前一后"交替，最后 cooldown 做剩余反向。气泡比例**一样**，但每段最多缓存 **p−i** 个 micro-batch 的激活（与 m 无关）。
- **Interleaved 1F1B**（virtual pipeline）：每卡放 v 个不连续的段，气泡缩小到 1/v，代价是 v 倍的 P2P 通信。

```
p=4, m=8 的 1F1B（F=前向, B=反向, 数字=micro-batch）
stage0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
stage1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
stage2:       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
stage3:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
(warmup 数 = p - i - 1 = 3,2,1,0)
```

### 2.7 专家并行（EP）

MoE 层的 E 个专家分到 ep 张卡上。每个 token 经路由后，用 **all-to-all** 把 token 发到它的专家所在卡（dispatch），专家计算后再 all-to-all 发回（combine）。细节（路由、容量、负载均衡、DeepEP）见 `guide/07-moe.md`。

### 2.8 混合精度与激活检查点

- **混合精度**：前向/反向用 bf16（或 fp16 + loss scaling），优化器在 fp32 主权重上更新，再拷回 bf16。梯度规约可选 fp32（`reduce_dtype`）以减少累加误差。
- **激活检查点（recompute）**：前向时只存每层（或每个区域）的输入，反向时重算前向得到中间激活。用约 33% 的额外计算（多一次前向）换显存。要注意 dropout 的 RNG 状态必须在重算时恢复——Megatron 的 `CheckpointFunction` 专门处理了 TP 下的 RNG tracker。

---

## 3. 最小实现

### 3.1 单进程模拟"列并行 + 行并行 MLP"，验证 all-reduce 等价性（含反向）

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, S, H, FFN, TP = 2, 3, 8, 32, 4
dt = torch.float64                       # 用 fp64 让误差只剩 ~1e-15

X  = torch.randn(B, S, H, dtype=dt)
W1 = torch.randn(H, FFN, dtype=dt)       # 列并行: 沿输出维(dim=1)切
W2 = torch.randn(FFN, H, dtype=dt)       # 行并行: 沿输入维(dim=0)切
G  = torch.randn(B, S, H, dtype=dt)      # 上游传回来的 dL/dY

# ---------- 参考：单卡 ----------
Xr, W1r, W2r = (t.clone().requires_grad_() for t in (X, W1, W2))
Y_ref = F.gelu(Xr @ W1r) @ W2r
(Y_ref * G).sum().backward()

# ---------- TP=4 模拟 ----------
W1_shards = [w.clone().requires_grad_() for w in W1.chunk(TP, dim=1)]  # 每卡 [H, FFN/TP]
W2_shards = [w.clone().requires_grad_() for w in W2.chunk(TP, dim=0)]  # 每卡 [FFN/TP, H]
# f 算子：前向 identity —— 每张"卡"拿到同一个 X 的副本
X_copies = [X.clone().requires_grad_() for _ in range(TP)]

partials = []
for r in range(TP):
    Z_r = F.gelu(X_copies[r] @ W1_shards[r])      # [B, S, FFN/TP]，GeLU 逐元素，无需通信
    partials.append(Z_r @ W2_shards[r])           # [B, S, H]，部分和
# g 算子：前向 all-reduce
Y_tp = torch.stack(partials).sum(0)
(Y_tp * G).sum().backward()                       # g 的反向是 identity：每个 partial 都收到 G

# f 算子的反向：all-reduce 输入梯度
dX_tp = torch.stack([x.grad for x in X_copies]).sum(0)

print("forward  max err:", (Y_tp - Y_ref).abs().max().item())
print("dX       max err:", (dX_tp - Xr.grad).abs().max().item())
print("dW1      max err:", (torch.cat([w.grad for w in W1_shards], 1) - W1r.grad).abs().max().item())
print("dW2      max err:", (torch.cat([w.grad for w in W2_shards], 0) - W2r.grad).abs().max().item())
# 注意：单个 X_copies[r].grad 不等于 Xr.grad —— 不做 f 的反向 all-reduce，输入梯度就是错的
print("one rank dX err (should be large):", (X_copies[0].grad - Xr.grad).abs().max().item())
```

要点：

- 权重梯度**不需要通信**：每卡的 `dW1_r`、`dW2_r` 恰好是完整梯度的对应切片（数据并行维度上的梯度规约是另一回事）。
- 前向唯一的通信是 g（all-reduce 部分和），反向唯一的通信是 f（all-reduce 输入梯度）。
- 我用 NumPy 版本核对过前向：误差 ~3.6e-15。

### 3.2 单进程模拟 ZeRO：reduce-scatter → 局部更新 → all-gather ≡ DDP

```python
import torch

torch.manual_seed(0)
N, P, lr = 4, 1000, 0.1
param = torch.randn(P)
grads = [torch.randn(P) for _ in range(N)]                    # 每张卡算出的本地梯度（完整大小）

# DDP: all-reduce 平均梯度，每卡做完整更新
ddp_param = param - lr * torch.stack(grads).mean(0)

# ZeRO-2: reduce-scatter，卡 k 只拿到第 k 段的平均梯度
grad_shards = [torch.stack([g.chunk(N)[k] for g in grads]).mean(0) for k in range(N)]
# 每卡只保存/更新自己那段 fp32 主权重（以及该段的 Adam m/v —— 这里用 SGD 省略）
param_shards = [p - lr * gs for p, gs in zip(param.chunk(N), grad_shards)]
# all-gather 拼回完整参数，供下一步前向
zero_param = torch.cat(param_shards)

print("max |ZeRO - DDP| =", (zero_param - ddp_param).abs().max().item())   # 0.0
```

### 3.3 单进程模拟 Ring Attention（非 causal）

```python
import torch

torch.manual_seed(0)
S, D, CP = 16, 8, 4
Q, K, V = (torch.randn(S, D, dtype=torch.float64) for _ in range(3))
ref = torch.softmax(Q @ K.T / D**0.5, dim=-1) @ V

Qs, Ks, Vs = Q.chunk(CP), K.chunk(CP), V.chunk(CP)       # 每卡一段序列
outs = []
for r in range(CP):                                      # 卡 r
    q = Qs[r]
    m = torch.full((q.shape[0], 1), float("-inf"), dtype=q.dtype)
    l = torch.zeros(q.shape[0], 1, dtype=q.dtype)
    o = torch.zeros_like(q)
    for step in range(CP):                               # KV 沿环传 CP 次（第 0 步是自己的）
        j = (r - step) % CP                              # 第 step 步手里拿着来自卡 j 的 KV 块
        s = q @ Ks[j].T / D**0.5
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        scale = torch.exp(m - m_new)                     # 旧结果的重缩放因子
        l = l * scale + p.sum(-1, keepdim=True)
        o = o * scale + p @ Vs[j]
        m = m_new
    outs.append(o / l)
print("ring attention max err:", (torch.cat(outs) - ref).abs().max().item())   # ~1e-16
```

真实实现里，内层循环的"拿到下一块 KV"是一次 `send/recv`，与当前块的计算重叠；causal 时还要处理块间掩码（`j > r` 的块整块跳过、`j == r` 的块用下三角）并配合 zigzag 负载均衡。

---

## 4. 源码导读（按阅读顺序）

阅读顺序：**数据并行（Megatron DDP）→ ZeRO（DeepSpeed stage 1/2/3）→ TP/SP（Megatron mappings + layers）→ PP（Megatron 1F1B）→ CP → 激活检查点 → torchtitan 如何把这些拼到 Llama 上 → Accelerate 的 prepare**。

### 4.1 数据并行：梯度桶 + 反向钩子重叠通信

DDP 的核心优化：**不要等反向全部结束再 all-reduce**。把参数按反向顺序分成若干"桶"，一个桶里所有参数的梯度都算好了，就立刻异步发起该桶的规约，与后续层的反向计算重叠。

文件：`megatron/core/distributed/distributed_data_parallel.py` `_make_backward_post_hook` L581 起
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/distributed/distributed_data_parallel.py#L581-L612

```python
def hook(*unused):
    ...
    if param.grad is not None and (not param.grad_added_to_main_grad or ...):
        param.main_grad.add_(param.grad.data)       # L606: 累加到 fp32 的连续 main_grad 缓冲区
    param.grad = None
    if self.ddp_config.overlap_grad_reduce:
        self.param_to_bucket_group[param].register_grad_ready(...)   # L610
```

- `main_grad` 是一大块连续的 fp32（可配置）缓冲区，每个参数的梯度是其中的一个 view——规约时整个桶是**一次** collective，避免成百上千个小 kernel。
- 梯度累积（多个 micro-batch）期间只累加不通信，只有最后一个 micro-batch 才"登记就绪"。

文件：`megatron/core/distributed/param_and_grad_buffer.py` `register_grad_ready` L924–L946，`start_grad_sync` L661 起（collective 在 L769–L793）
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/distributed/param_and_grad_buffer.py#L769-L793

```python
with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
    for idx, bucket in enumerate(self.buckets):
        if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
            local_data_view = self.cached_grad_buffer_shard_list[idx][self.intra_distributed_optimizer_instance_rank]
            grad_reduce_handle = dist_reduce_scatter_func(local_data_view, bucket.grad_data, op=reduce_op, ...)
        else:
            torch.distributed.all_reduce(bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op)
```

这里一眼就能看出 DDP 与 ZeRO 的关系：**普通 DDP 走 all-reduce；开启 `use_distributed_optimizer`（Megatron 版的 ZeRO-1/2）就改成 reduce-scatter**，每卡只拿自己那段梯度（`local_data_view`），优化器只更新这一段，之后再 all-gather 参数。

`register_grad_ready` 用计数器 `per_param_grad_ready_counts == golden_per_param_grad_ready_counts` 判断"桶里每个参数都就绪了"才调用 `start_grad_sync`——这就是"桶满即发"。

### 4.2 ZeRO-1/2：DeepSpeed `stage_1_and_2.py`

#### (1) 一个类同时实现 stage 1 和 2

文件：`deepspeed/runtime/zero/stage_1_and_2.py` L142（类定义）、L250–L252
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage_1_and_2.py#L250-L252

```python
# ZeRO stage 1 (False) or 2 (True)
self.partition_gradients = partition_grads
self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
```

#### (2) 展平 + 切分 + 只为自己那段建 fp32 主权重

文件：`stage_1_and_2.py` L487–L568
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage_1_and_2.py#L487-L568

```python
flattened_buffer = self.flatten_dense_tensors_aligned(self.round_robin_bit16_groups[i], alignment, ...)
self.bit16_groups_flat.append(flattened_buffer)                     # 整个 param group 拼成一维 bf16 向量
self._update_model_bit16_weights(i)                                 # 模型参数改成指向这个大向量的 view
data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)   # L522: 切 N 段
self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
...
weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].detach().clone().to(
    device=self.device, dtype=self.master_weights_and_grads_dtype)  # L544: 只克隆自己那 1/N 成 fp32
...
param_group['params'] = [self.single_partition_of_fp32_groups[i]]  # L568: 内部优化器只看见这 1/N
```

逐行理解：

- 把一个 param group 的所有 bf16 参数**展平**成一个连续一维向量（带对齐 padding），这样"切 N 段"就是简单的 `narrow`，不用管单个参数的边界（一个参数可能跨两段）。
- `get_data_parallel_partitions`（L2308–L2326）：`base_size = numel // dp`，余数分给前几个 rank，用 `tensor.narrow(0, start, partition_size)` 得到 N 个 view。
- **ZeRO-1 的全部秘密就在 L544 + L568**：fp32 主权重只为本 rank 那一段创建；把内部优化器（如 Adam）的 `params` 替换成这一段——于是 Adam 的 m、v 也只为这一段分配。12 字节/参数 → 12/N。

#### (3) 反向：按桶规约，"reduce-scatter"实际是一串 reduce

文件：`stage_1_and_2.py` `reduce_independent_p_g_buckets_and_remove_grads` L1295 起，`average_tensor` L1457–L1592
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage_1_and_2.py#L1457-L1592

```python
if bucket.elements + param.numel() > self.reduce_bucket_size:     # L1303: 桶满了 -> 规约
    self.reduce_ipg_grads(comm_dtype=comm_dtype)
```

`average_tensor` 里先算出桶中每段梯度属于哪个 rank 的分区（`rank_and_offsets`），然后：

```python
if self.gradient_average:
    tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))   # L1551: 求平均
...
    bucket_key = ("reduce", dst, process_group)                    # 属于 rank dst 的那段
...
    self.allreduce_no_retain(buckets[bucket_key], ..., rank=dst, ...)
```

再看 `allreduce_bucket`（L2175–L2209）：

```python
if rank is None:
    dist.all_reduce(tensor_to_allreduce, group=process_group)    # L2200
else:
    global_rank = dist.get_global_rank(process_group, rank)
    dist.reduce(tensor_to_allreduce, global_rank, group=process_group)   # L2203
```

也就是说，DeepSpeed 的 stage 1/2 "reduce-scatter" 在实现上是**把桶按目标分区拆开，对每一段做一次 `dist.reduce` 到所属 rank**——语义等价于 reduce-scatter（每个 rank 只得到自己那段的和），但用的是 reduce 原语，好处是桶可以按反向顺序灵活组织而不必与分区边界对齐。

#### (4) ZeRO-1 vs ZeRO-2 的真正区别：其他分区的梯度什么时候扔

文件：`stage_1_and_2.py` L2077–L2089、L2105–L2111
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage_1_and_2.py#L2077-L2111

```python
self.params_already_reduced[param_id] = True
if self.partition_gradients:
    if not self.is_param_in_current_partition[param_id]:
        ...
            self.clear_grad_attribute(param)          # ZeRO-2: 不属于我的梯度，规约完立刻释放
    elif self.contiguous_gradients:
        self.copy_grads_in_partition(param)            # 属于我的，拷进连续的分区梯度缓冲
else:  # zero stage 1 - partition only optimizer state
    if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
        self.copy_grads_in_partition(param)
...
def reduce_ready_partitions_and_remove_grads(self, param, i):
    if self.partition_gradients or self.is_gradient_accumulation_boundary() or self.zenflow:
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
```

- **ZeRO-2**：每次反向都边算边规约，规约后不属于本 rank 的梯度立即释放 → 梯度显存 ≈ 1/N（加一个桶的临时空间）。
- **ZeRO-1**：只在**梯度累积边界**（最后一个 micro-batch）才规约；在那之前每卡都保留完整梯度（2 字节/参数）。

#### (5) step：更新自己那段，再 all-gather

文件：`stage_1_and_2.py` L2885–L2906
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage_1_and_2.py#L2885-L2906

```python
self._optimizer_step(i)                                   # Adam 只更新 fp32 分区
self.single_partition_of_fp32_groups[i].grad = None
bit16_partitions = self.parallel_partitioned_bit16_groups[i]
fp32_partition = self.single_partition_of_fp32_groups[i]
bit16_partitions[partition_id].data.copy_(fp32_partition.data)   # fp32 -> bf16，写回大向量里自己那段
...
all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                     partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                     dp_process_group=self.real_dp_process_group, ...)   # 其余 N-1 段从别人那里收
```

因为 `bit16_groups_flat` 是模型参数的底层存储（第 (2) 步做了 view），all-gather 写进去之后模型参数就自动是最新的了。

### 4.3 ZeRO-3：DeepSpeed `partition_parameters.py` + `stage3.py`

ZeRO-3 要切参数本身，所以必须**在模块级别**介入：前向进入一个子模块前把参数 all-gather 回来，出来后释放。

#### (1) 参数切分：`ds_tensor` 存 1/N，`param.data` 置空

文件：`deepspeed/runtime/zero/partition_parameters.py` `_partition_param` L1778–L1890，`free_param` L314–L330
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/partition_parameters.py#L1778-L1890

```python
tensor_size = self._aligned_size(param)
partition_size = tensor_size // self._partition_world_size(param)
...
partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=device)
...
param.ds_tensor = partitioned_tensor                      # 本 rank 保管的那 1/N
...
start = partition_size * self._partition_rank(param)
end = start + partition_size
one_dim_param = param.contiguous().view(-1)
if start < param.ds_numel and end <= param.ds_numel:
    src_tensor = one_dim_param.narrow(0, start, partition_size)
    param.ds_tensor.copy_(src_tensor)
...
free_param(param)
```

`free_param` 的核心一行（L329）：

```python
param.data = torch.empty(0, dtype=param.dtype, device=param.device)
param.ds_status = ZeroParamStatus.NOT_AVAILABLE
```

参数对象（`nn.Parameter`）本身还在，模块结构不变，但它的 `.data` 变成了空张量；真正的数据在 `param.ds_tensor` 里（可在 GPU/CPU/NVMe，取决于 offload 配置）。这一切发生在 `deepspeed.zero.Init` 上下文（L951 `class Init`）中：模型**构造时**每个参数一创建就被切分，所以 175B 模型也能"构造"出来而不 OOM。

#### (2) 前向钩子：进入子模块前 fetch，离开后 release

文件：`deepspeed/runtime/zero/parameter_offload.py` L358–L363、L531–L571
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/parameter_offload.py#L531-L571

```python
def _pre_forward_module_hook(module, *args):
    self.pre_sub_module_forward_function(module)
...
def pre_sub_module_forward_function(self, sub_module):
    ...
    param_coordinator.fetch_sub_module(sub_module, forward=True)     # all-gather 该子模块的参数
...
def post_sub_module_forward_function(self, sub_module):
    ...
    param_coordinator.release_sub_module(sub_module, forward=True)   # 用完重新切分/释放
```

文件：`deepspeed/runtime/zero/partitioned_param_coordinator.py` `fetch_sub_module` L310 起
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/partitioned_param_coordinator.py#L310-L316

docstring 写得很清楚：

```
1. kick off fetch for parameters in immediately required sub module
2. kick off fetch for next few parameters we will need later (prefetch)
3. block on parameters in immediately required sub module
```

第 2 步**预取**依赖于第一个 step 记录下来的模块执行轨迹（trace）：之后每一步按轨迹提前 all-gather 下几个模块的参数（`prefetch_bucket_sz` 控制量），让通信和当前模块的计算重叠。

#### (3) 反向：reduce-scatter 梯度

文件：`deepspeed/runtime/zero/stage3.py` `__avg_scatter_grads` L1910 起（collective 在 L1929）
链接：https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/deepspeed/runtime/zero/stage3.py#L1910-L1935

```python
full_grads_for_rank = [p.grad for p in params_to_reduce]
if communication_data_type != self.dtype:
    full_grads_for_rank = [g.to(communication_data_type) for g in full_grads_for_rank]
...
    grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, process_group)
```

与 stage 1/2 不同，stage 3 用的是真正的（合并多个张量的）reduce-scatter：每个参数在 ZeRO-3 中本来就是按自己均分的（`ds_tensor` 的切法），梯度的切法与之对齐，直接 reduce-scatter 即可。`__reduce_and_partition_ipg_grads`（L1647 起）开头的检查 `param.grad.numel() != param.ds_numel` 确保梯度是完整大小的。

### 4.4 张量并行：Megatron `mappings.py`

文件：`megatron/core/tensor_parallel/mappings.py` L26–L281
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/mappings.py#L205-L281

```python
class _CopyToModelParallelRegion(torch.autograd.Function):       # f
    def forward(ctx, input_, group):
        ctx.group = group
        return input_                                            # 前向：恒等
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None             # 反向：all-reduce

class _ReduceFromModelParallelRegion(torch.autograd.Function):   # g
    def forward(ctx, input_, group):
        return _reduce(input_, group)                            # 前向：all-reduce
    def backward(ctx, grad_output):
        return grad_output, None                                 # 反向：恒等

class _ScatterToModelParallelRegion(...):    # 前向沿最后一维切，反向沿最后一维 all-gather
class _GatherFromModelParallelRegion(...):   # 前向沿最后一维 all-gather，反向切
```

- `_reduce`（L26–L41）：`input_.contiguous()` 后 `torch.distributed.all_reduce(input_, group=group)`；`group.size()==1` 时直接返回——所以同一份代码 TP=1 也能跑。
- 这四个 autograd Function 两两共轭（前向是 A，反向就是 A 的"转置"），这是 TP 正确性的核心。
- `_gather_along_last_dim`（L88–L104）有个细节：NCCL all-gather 只能沿**第 0 维**拼接，所以它先 gather 到 `[world*dim0, ...]`，再 `chunk(world_size, dim=0)` 后 `cat(dim=-1)`。
- 序列并行用的是 `_ScatterToSequenceParallelRegion` / `_GatherFromSequenceParallelRegion`（L284、L304）/ `_ReduceScatterToSequenceParallelRegion`（L359），沿**第 0 维**（Megatron 激活布局是 `[s, b, h]`，第 0 维就是序列）。

### 4.5 张量并行：`ColumnParallelLinear` / `RowParallelLinear`

#### (1) 列并行：权重形状 `[out/t, in]`

文件：`megatron/core/tensor_parallel/layers.py` `ColumnParallelLinear` L1044 起；切分 L1150；权重分配 L1180–L1200
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/layers.py#L1150-L1200

```python
self.output_size_per_partition = divide(output_size, world_size)
# Note: torch.nn.functional.linear performs XA^T + b and as a result
# we allocate the transpose.
self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, ...))
_initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=stride, ...)
```

`F.linear` 计算 `X Wᵀ`，所以数学上的 `A[:, cols]`（按列切）在存储上是 `W[rows, :]`，`partition_dim=0`。

#### (2) 列并行 forward

文件：`layers.py` L1290–L1401
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/layers.py#L1340-L1401

```python
if (self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm or self.disable_grad_reduce):
    input_parallel = input_
else:
    input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)   # 显式 f
...
output_parallel = self._forward_impl(input=input_parallel, weight=weight, bias=bias,
                                     allreduce_dgrad=allreduce_dgrad, sequence_parallel=..., tp_group=self.tp_group, ...)
if gather_output:
    output = self.gather_tensor_parallel_output(output_parallel, ...)
else:
    output = output_parallel                                                              # [s, b, out/t]
```

有意思的是：**默认情况下并不走 `copy_to_tensor_model_parallel_region`**。`self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce`（L1256–L1258），TP>1 且不开 SP 时它为 True，于是 f 的"反向 all-reduce"被挪进了 `_forward_impl` 背后的自定义 autograd 函数里，目的是**与权重梯度计算重叠**（见下一小节）。

`gather_output=False` 是常态：MLP 的 `linear_fc1`（`megatron/core/transformer/mlp.py` L221–L235 `gather_output=False`）和 attention 的 `linear_qkv`（`attention.py` L1771–L1785 `gather_output=False`）都保持输出被切开，直接喂给后面的行并行层。

#### (3) 真正干活的 autograd 函数：通信与计算重叠

文件：`layers.py` `LinearWithGradAccumulationAndAsyncCommunication` L681 起
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/layers.py#L681-L812

前向（L723–L734）：

```python
if sequence_parallel:
    dim_size = list(input.size()); dim_size[0] = dim_size[0] * tp_group.size()
    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
    dist_all_gather_func(all_gather_buffer, input, group=tp_group)   # SP: [s/t,b,h] -> [s,b,h]
    total_input = all_gather_buffer
else:
    total_input = input
return _linear_forward(total_input, weight, bias, output_dtype)
```

反向（L784–L810）：

```python
grad_input = grad_output.matmul(weight)                              # dX = dY W  (L784)
...
if ctx.allreduce_dgrad:
    handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)   # L797: 异步！
    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # all-reduce is scheduled before the weight gradient computation
if ctx.sequence_parallel:
    ...
    handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)   # SP: reduce-scatter
# 随后计算 dW = dYᵀ X（wgrad GEMM），与上面的通信并行
```

这就是"把 f 的反向放进 Linear 里"的好处：dX 一算完立刻**异步**发起 all-reduce，然后去算 dW；dW 不依赖通信结果，二者重叠。`CUDA_DEVICE_MAX_CONNECTIONS=1` 保证 GPU 上 NCCL kernel 按发射顺序先于 wgrad GEMM 被调度。（torchtitan 的 README 特意提醒：FSDP2 + 原生 TP 恰恰相反，应该把这个变量设大。）

SP 模式下前向的 all-gather 和反向的 reduce-scatter 正是 2.4 节说的"把 all-reduce 拆成两半"。注意反向时还要**再 all-gather 一次输入**（L770–L782，异步）来算 dW，因为前向只存了切开的 `[s/t, b, h]`——用通信换显存。

#### (4) 行并行：权重形状 `[out, in/t]`，输出 all-reduce

文件：`layers.py` `RowParallelLinear` L1465 起；L1545 切分；`forward` L1653–L1710
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/layers.py#L1653-L1710

```python
if self.input_is_parallel:
    input_parallel = input_                                             # 上游列并行的输出已经切好
else:
    input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
allreduce_dgrad = False
output_parallel = self._forward_impl(input=input_parallel, weight=self.weight, bias=None, ...)   # [s,b,h] 部分和
if self.explicit_expert_comm:
    output_ = output_parallel
elif self.sequence_parallel:
    output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)   # SP: [s/t,b,h]
else:
    output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)    # g: all-reduce
if not self.skip_bias_add:
    output = (output_ + self.bias) if self.bias is not None else output_
```

- `bias=None` 传给矩阵乘，**all-reduce 之后再加 bias**：否则 t 张卡每张加一次，bias 被加了 t 次。torchtitan 的 `RowParallelLinear` 用另一种方式解决同一问题（把 bias 转成 Partial，见 4.9）。
- `allreduce_dgrad = False`：行并行层的输入梯度 `dX_i = dY B_iᵀ` 本来就只属于本卡的那 1/t 切片，不需要规约。
- MLP 的 `linear_fc2`（`mlp.py` L242–L257，`input_is_parallel=True`）、attention 的 `linear_proj`（`attention.py` L414–L427，`input_is_parallel=True`）都是行并行。

#### (5) Attention 的头切分

文件：`megatron/core/transformer/attention.py` L350–L369
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/attention.py#L350-L369

```python
if self.config.num_query_groups < world_size:
    # When num_kv_heads < tp_size, each TP rank (post AG) initially produces
    # activations for 1 kv_head and (num_q_heads / num_kv_heads) q_heads.
    self.num_query_groups_per_partition = 1
    ...
else:
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
```

正常情况每卡 `n_heads/t` 个 Q 头、`n_kv/t` 个 KV 头。GQA 的 KV 头数可能小于 TP 度（如 8 个 KV 头、TP=16），这时 KV 头需要在多卡上复制——注释里描述了这一特殊处理。

**一个 Transformer 层的 TP 通信汇总**（不开 SP）：

```
x ──f──> linear_qkv(列) ─> attention(每卡 n_heads/t 个头) ─> linear_proj(行) ──g(all-reduce)──> +残差
  ──f──> linear_fc1(列) ─> GeLU/SwiGLU ─> linear_fc2(行) ──g(all-reduce)──> +残差
前向 2 次 all-reduce（两个 g），反向 2 次 all-reduce（两个 f）
```

### 4.6 流水线并行：Megatron 1F1B

文件：`megatron/core/pipeline_parallel/schedules.py` `forward_backward_pipelining_without_interleaving` L2176 起
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/pipeline_parallel/schedules.py#L2307-L2518

**warmup 数**（L2307–L2310）：

```python
num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```

第 i 段先做 `p−i−1` 个前向；最后一段（i=p−1）warmup=0，一上来就 1F1B。

**warmup 阶段**（L2364–L2401）：

```python
for i in range(num_warmup_microbatches):
    input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, p2p_communicator.is_pp_first_stage)
    output_tensor, num_tokens = forward_step(forward_step_func, forward_data_iterator, model, ..., input_tensor, ...)
    p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

- 第一段的 `recv_forward` 什么都不收（从数据迭代器取数据），最后一段的 `send_forward` 什么都不发。
- `input_tensors` / `output_tensors` 是 FIFO 队列：存着"已前向、待反向"的 micro-batch，**队列长度就是该段的激活缓存数**——1F1B 下最多 `warmup + 1 = p − i`。
- `deallocate_output_tensor`（L172 起，仅在 `config.deallocate_pipeline_outputs` 开启时生效）：输出已经发给下一段了，本地只需要它的 `grad_fn` 图，不需要 `.data`，把 data 换成 1 元素张量省显存（反向时用 `custom_backward`（L205 起）直接调 C++ autograd 引擎，绕过 PyTorch 对 output/grad 形状一致的检查）。

**稳态 1F1B**（L2415–L2491）：

```python
for i in range(num_microbatches_remaining):
    output_tensor, num_tokens = forward_step(...)                       # 1F
    output_tensor_grad = p2p_communicator.send_forward_recv_backward(output_tensor, ...)   # 发前向结果、收反向梯度（合并的 P2P）
    input_tensors.append(input_tensor); output_tensors.append(output_tensor)
    input_tensor = input_tensors.pop(0); output_tensor = output_tensors.pop(0)   # 取最老的 micro-batch
    input_tensor_grad = backward_func(input_tensor, output_tensor, output_tensor_grad, config)   # 1B
    if last_iteration:
        p2p_communicator.send_backward(input_tensor_grad, ...)
    else:
        input_tensor = p2p_communicator.send_backward_recv_forward(input_tensor_grad, ...)       # 发梯度、收下一个前向输入
```

`send_forward_recv_backward` / `send_backward_recv_forward` 把一发一收合并成一次批量 P2P，避免两段互相等待死锁并减少延迟。

**cooldown**（L2493–L2518）：把 warmup 欠下的 `num_warmup_microbatches` 个反向做完。注意 L2502–L2504：在最后一个反向前 `enable_grad_sync()`——DP 梯度规约只在最后一个 micro-batch 的反向中触发，并与之重叠（呼应 4.1 的 `is_last_microbatch`）。

### 4.7 上下文并行（CP）

Megatron 的 CP attention 计算本身委托给 Transformer Engine（不在本仓库中），但通信方式和输入切分可以在这里看到。

文件：`megatron/core/transformer/transformer_config.py` L1156–L1170
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/transformer_config.py#L1156-L1170

```
"p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
overlapped with attention compute.
"all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
async, and cannot be overlapped.
"a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
full sequence of QKV.
"a2a+p2p": A hierarchical implementation of context parallelism to attention.
```

正好对应 2.5 节的三种方案（p2p 就是 ring attention）。

输入切分（zigzag 负载均衡）：`megatron/core/utils.py` `_get_batch_on_this_cp_rank_per_sequence_balancing` L2627–L2686
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/utils.py#L2668-L2686

```python
val = val.view(*val.shape[0:seq_dim], 2 * cp_size, val.shape[seq_dim] // (2 * cp_size), *val.shape[(seq_dim + 1):])
index = torch.zeros(2, dtype=torch.int64, device=val.device)
index[0].fill_(cp_rank)
index[1].fill_(2 * cp_size - cp_rank - 1)
val = val.index_select(seq_dim, index)
val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
```

`[b, s, ...]` → `[b, 2cp, s/(2cp), ...]`，取第 `cp_rank` 块和第 `2cp−1−cp_rank` 块，再拼回 `[b, s/cp, ...]`。docstring 举例：CP=2 时 GPU0 拿 (chunk0, chunk3)，GPU1 拿 (chunk1, chunk2)——"头 + 尾"，每卡的 causal attention 工作量相等。

torchtitan 中的 CP 实现（all-gather KV 与 Ulysses）可读性很好：

文件：`torchtitan/models/common/attention/cp_attention.py` L202–L225（KV all-gather）、L283–L314（Ulysses）
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/models/common/attention/cp_attention.py#L202-L314

```python
# KV all-gather: q 保持按 token 切，k/v 从 Shard(token) 变为 Replicate
k_THK, v_THV = (spmd.redistribute(x, cp_group, src=spmd.S(_TOKEN_DIM), dst=spmd.R, ...) for x in (k_THK, v_THV))
return FlexInnerAttention.forward(self, q_THK, k_THK, v_THV, **kwargs)

# Ulysses: (T/cp, H, *) -> (T, H/cp, *)，算完再 all-to-all 回去
q_THK, k_THK, v_THV = (spmd.redistribute(x, cp_group, src=spmd.S(_TOKEN_DIM), dst=spmd.S(_HEAD_DIM)) for x in (q_THK, k_THK, v_THV))
out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
return spmd.redistribute(out_THV, cp_group, src=spmd.S(_HEAD_DIM), dst=spmd.S(_TOKEN_DIM))
```

变量名后缀 `_THK` 是 torchtitan 的形状约定：T=tokens，H=heads，K=head_dim。"从 `Shard(token)` 变成 `Shard(head)`"这个 redistribute 在底层就是一次 all-to-all。负载均衡见 `torchtitan/distributed/context_parallel.py` `HeadTailCPLoadBalancer` L59–L94（要求 `seq_len % (2 * cp_size) == 0`）。

### 4.8 激活检查点：Megatron `CheckpointFunction`

文件：`megatron/core/tensor_parallel/random.py` L672–L753
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/tensor_parallel/random.py#L672-L753

```python
def forward(ctx, run_function, distribute_saved_activations, *args):
    ctx.rng_states = _get_all_rng_states()                 # 记下 CPU/CUDA/TP-tracker 的 RNG 状态
    with torch.no_grad():
        outputs = run_function(*args)                      # 不建图 -> 不存中间激活
    if distribute_saved_activations:                       # 输入也按 TP 切开存，进一步省显存
        safely_set_viewless_tensor_data(args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))
    ctx.save_for_backward(*args)                           # 只存输入
    return outputs

def backward(ctx, *args):
    inputs = ctx.saved_tensors
    if ctx.distribute_saved_activations:
        safely_set_viewless_tensor_data(inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))
    with _fork_rng():
        _set_all_rng_states(*ctx.rng_states)               # 恢复 RNG -> dropout mask 与前向一致
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad(), ...:
            outputs = ctx.run_function(*detached_inputs)   # 重算前向，这次建图
    torch.autograd.backward(outputs, args)                 # 在重建的图上反向
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return (None, None) + grads
```

与 `torch.utils.checkpoint` 的区别（docstring）：额外保存/恢复 TP 的 RNG tracker——TP 区域内每卡的 dropout 种子不同、区域外相同，重算时必须分别恢复。

1F1B 也与重计算联动：`schedules.py` L2312–L2322 的 `num_microbatches_with_partial_activation_checkpoints` 根据"在途 micro-batch 数"决定哪些 micro-batch 做全量重计算。

### 4.9 torchtitan：把 TP/FSDP2/PP/CP/AC 应用到 Llama

torchtitan 的新版本把"并行化"写成**声明式**：每个模块带一个 `ShardingConfig`，说明它的参数和输入/输出在各个 mesh 轴（DP/CP/TP）上的布局；`parallelize()` 递归地按声明分发参数、在 forward 前后插入 redistribute。

#### (1) 入口顺序：TP/SP → AC → FSDP

文件：`torchtitan/models/base/model.py` `BaseModel.parallelize` L131–L153
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/models/base/model.py#L131-L153

```python
apply_local_compile(local_compile_regions)
with parallelism_context.activate_spmd():
    self._parallelize(parallelism_context)               # 1. 按 ShardingConfig 做 TP/SP/CP 布局
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(self)   # 2. 激活检查点
    self._apply_fsdp(parallelism_context=..., training=..., parallelism=...)   # 3. FSDP2 最外层
return self
```

顺序很重要：FSDP 必须最后套，因为它要包住已经被 TP 切过的参数（DTensor）。PP 时（`training_engine.py` L348–L364）先 `model.pipeline(...)` 把模型切成若干 stage，再对**每个 stage** 调用 `parallelize`（`distributed/pipeline_parallel.py` `pipeline_llm` L75–L175，循环在 L141–L155）。

#### (2) `_parallelize`：分发参数 + 包装 forward

文件：`torchtitan/models/base/module.py` L257–L313
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/models/base/module.py#L257-L313

```python
spmd_validate_redistributions(self._sharding_config)
self._distribute_states(parallelism_context)          # 参数按 state_shardings 切分（变成本地分片）
self._cache_pos_arg_names()
fn = self._maybe_wrap_with_local_region(self.forward)
def forward_with_redistribution(*args, **kwargs):
    args, kwargs = self._redistribute_inputs(args, kwargs)     # 输入：src 布局 -> dst 布局
    outputs = fn(*args, **kwargs)
    return self._redistribute_outputs(outputs)                 # 输出：src 布局 -> dst 布局
self.forward = forward_with_redistribution
```

#### (3) 列/行并行的布局声明

文件：`torchtitan/models/common/decoder_sharding.py` `colwise_sharding_config` L120–L129、`rowwise_sharding_config` L159–L173
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/models/common/decoder_sharding.py#L120-L173

```python
def colwise_sharding_config(*, input_layout):
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0)), "bias": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": input_layout},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
    )
def rowwise_sharding_config(*, output_layout):
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(1)), "bias": dense_param_placement(tp=spmd.I)},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))},
        out_src_shardings=output_layout,
    )
```

对照 Megatron：

- 列并行 `weight: TP=S(0)`（`[out, in]` 沿 out 切，与 Megatron `partition_dim=0` 一致），输出沿最后一维（特征）切 `S(-1)`；
- 行并行 `weight: TP=S(1)`（沿 in 切，与 Megatron `partition_dim=1` 一致），输入就是列并行的输出布局 `S(-1)`；bias 是 `I`（invariant，各卡相同）。
- `dense_param_placement`（L18–L30）中 DP、CP 轴都是 `R`（replicate）——参数在 DP/CP 维上的切分交给后面的 FSDP。
- 残差流布局 `residual_activation_placement`（L214–L221）：开 SP 时是 `dense_sequence_parallel_placement()`（tokens 沿 `(DP, CP, TP)` 切），否则 TP 上是 `I`（每卡一份相同的完整激活）。

在哪里用：`torchtitan/models/common/config_utils.py` 的 `make_gqa_config`（L226 起，`wqkv` 列并行 L264–L278、`wo` 行并行 L286–L291）和 `make_ffn_config`（L300–L330，`w13` 列并行、`w2` 行并行）。

#### (4) 通信插在哪：`ColumnParallelLinear` / `RowParallelLinear`

文件：`torchtitan/models/common/linear.py` L233–L344
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/models/common/linear.py#L233-L344

```python
def maybe_gather_tp_input(module, x):
    ...
    return remat.region(spmd.redistribute, module.remat_region_name("tp_gather"), ...)(
        x, tp_group,
        src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,   # SP: Shard(0) -> R 是 all-gather；否则 I -> R 前向 no-op、反向 all-reduce
        dst=spmd.R, ...)

class ColumnParallelLinear(Linear):
    def forward(self, input):
        return super().forward(maybe_gather_tp_input(self, input))

class RowParallelLinear(Linear):
    def forward(self, input):
        ...
        if bias is not None and tp_group is not None:
            bias = spmd.convert(bias, tp_group, src=spmd.I, dst=spmd.P, expert_mode=True)   # bias 变成 Partial
        ...
        return remat.region(spmd.redistribute, self.remat_region_name("tp_reduce"), ...)(
            output, tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,    # P -> S(0) 是 reduce-scatter；P -> I 是 all-reduce
            ...)
```

这一段把 2.3/2.4 节翻译成了"布局转换"的语言：

| 转换 | 前向 | 反向 | 对应 Megatron |
|---|---|---|---|
| `I → R`（不开 SP） | no-op | all-reduce | f / `copy_to_tensor_model_parallel_region` |
| `S(0) → R`（SP） | all-gather | reduce-scatter | `gather_from_sequence_parallel_region` |
| `P → I`（不开 SP） | all-reduce | no-op | g / `reduce_from_tensor_model_parallel_region` |
| `P → S(0)`（SP） | reduce-scatter | all-gather | `reduce_scatter_to_sequence_parallel_region` |

`P`（Partial）表示"各卡持有部分和，真值是它们的和"。bias 被转成 Partial 后参与部分和，reduce 时恰好只被加一次——与 Megatron"all-reduce 后再加 bias"殊途同归。

#### (5) FSDP2：逐个 Transformer block `fully_shard`

文件：`torchtitan/models/common/decoder.py` `_apply_fsdp` L147–L175；`torchtitan/distributed/fsdp.py` `apply_fsdp_to_decoder` L231 起
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/distributed/fsdp.py#L289-L445

```python
mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False, ...)  # L289
fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
...
reshard_after_forward = get_fsdp_reshard_after_forward_policy(reshard_after_forward_policy, pp_enabled)   # L301
...
fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward)
fully_shard([model.norm, model.lm_head], **fsdp_config, reshard_after_forward=reshard_after_forward_policy == "always")
for layer_id, transformer_block in model.layers.items():                      # L333
    ...
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=stacked_param_placements.get)          # L432–L438（dense 分支）
fully_shard(model, **fsdp_config)                                              # L440: 根模块兜底
...
disable_fsdp_gradient_division(model)                                          # L445
```

逐点解释：

- **一个 block 一个 FSDP 单元**：前向进入 block 时 all-gather 该 block 的所有参数（一次合并的 collective），出来后释放；反向同理再 gather，算完 reduce-scatter 梯度。这正是 2.2 节 ZeRO-3 的时序，粒度 = 一个 Transformer block。
- `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`：分片存的是 fp32（优化器直接在分片上更新，等价于"fp32 主权重"），all-gather 时转成 bf16 参与计算，梯度 reduce-scatter 用 fp32 规约——对应 2.1 节的混合精度账本。
- `reshard_after_forward`（L165–L190）：`"default"` 时 `return not pp_enabled`。开 PP 时每个 micro-batch 都要前向一次，如果每次都释放再 gather，通信量乘以 micro-batch 数，所以 PP 下默认不 reshard（即 ZeRO-2 式行为：参数常驻、梯度分片）。最后的 `norm + lm_head` 默认也不 reshard，因为反向一开始就要用它们，释放了马上又要 gather。
- `disable_fsdp_gradient_division`（L135–L151）：把 FSDP 默认的"梯度除以 DP 大小"关掉，由训练循环按**全局 token 数**归一化 loss（不同 rank 的有效 token 数可能不同）。
- `resolve_fsdp_mesh`（L82 起）：FSDP 的分片轴是 `dp_shard`，开 CP 时**再加上 `cp`**（L107–L109）。原因：参数在 CP 维上是复制的（`dense_param_placement` 里 CP=R），各 CP rank 的梯度也需要求和——干脆把 CP 也当成 FSDP 的一个分片维度，一次 reduce-scatter 同时完成 DP 和 CP 的梯度规约。

#### (6) PP：切 stage 再各自 parallelize

文件：`torchtitan/distributed/pipeline_parallel.py` `pipeline_llm` L75–L175
链接：https://github.com/pytorch/torchtitan/blob/15f5f1a91b39592d2b567da7c8d6c0a6d5f89487/torchtitan/distributed/pipeline_parallel.py#L75-L175

```python
module_names_per_stage = _generate_llm_fqn_per_model_part(num_virtual_stages, num_layers, input_weight, output_weight)
stages, model_parts = _pipeline_module_split(model, pp_mesh, parallelism.pipeline_parallel_schedule, device, module_names_per_stage, ...)
for i, m in enumerate(model_parts):
    m = m.parallelize(parallelism_context=..., ...)     # 每个 stage 内部再做 TP/AC/FSDP
    model_parts[i] = m
    stages[i].submod = m
pp_schedule = _build_pipeline_schedule(parallelism=parallelism, num_microbatches=parallelism.num_pp_microbatches, stages=stages, loss_fn=loss_fn)
```

`input_weight` / `output_weight` 用于在切分时把 embedding 和 lm_head 也算作若干"层"的工作量，让各 stage 负载更均衡。调度器（1F1B、Interleaved1F1B、ZeroBubble 等）来自 PyTorch 的 `torch.distributed.pipelining`。

### 4.10 Accelerate：`Accelerator.prepare` 做了什么

文件：`src/accelerate/accelerator.py` `prepare` L1414 起（分派在 L1549–L1564）、`_prepare_one` L1397–L1412、`prepare_model` 中的 DDP 包装 L1880–L1901、`backward` L2828–L2860
链接：https://github.com/huggingface/accelerate/blob/ccfdff7dbbd6a8c455853a9e5eb6396ea51ffd45/src/accelerate/accelerator.py#L1549-L1564

```python
if self.distributed_type == DistributedType.DEEPSPEED:
    result = self._prepare_deepspeed(*args)
elif self.distributed_type == DistributedType.MEGATRON_LM:
    result = self._prepare_megatron_lm(*args)
elif self.is_fsdp2:
    result = self._prepare_fsdp2(*args)
else:
    ...
    result = tuple(self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement))
    result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))
```

- Accelerate 自己**不实现**任何并行算法，它是一个"分派器"：根据启动配置（`accelerate config`）把同一份用户代码交给 DeepSpeed / Megatron-LM / FSDP2 / DDP。
- 普通 DDP 路径：`_prepare_one` 对 DataLoader 调 `prepare_data_loader`（`data_loader.py` L1019 起，用 `BatchSamplerShard` L110 让每个进程只取自己的样本）、对 Module 调 `prepare_model`（`model.to(device)` 后包 `torch.nn.parallel.DistributedDataParallel`，L1897–L1899）、对 Optimizer 包一层 `AcceleratedOptimizer`。
- 两遍：第一遍处理 DataLoader/Model/Optimizer，第二遍处理 LR scheduler（它需要拿到包装后的 optimizer）。
- FSDP2 要求 model 和 optimizer **一起** `prepare`（L1506–L1521 的检查），因为 `fully_shard` 会把参数换成 DTensor，optimizer 必须重新指向新参数。
- `backward`（L2828 起）：非 DeepSpeed 时 `loss = loss / self.gradient_accumulation_steps`，再根据是否有 `GradScaler` 调用 `scaler.scale(loss).backward()`；DeepSpeed 时交给 `deepspeed_engine_wrapped.backward`（DeepSpeed 自己处理累积和 loss scale）。

### 4.11 专家并行（简述，详见 `guide/07-moe.md`）

文件：`megatron/core/transformer/moe/token_dispatcher.py` `MoEAlltoAllTokenDispatcher.token_dispatch` L688 起（L711–L717）
链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/token_dispatcher.py#L711-L717

```python
global_input_tokens = all_to_all(
    self.ep_group,
    permutated_local_input_tokens,
    self.output_splits,
    self.input_splits,
    use_nccl_stream=self.use_nccl_stream,
)
```

token 先按目标专家排序（permute），`input_splits[j]` = 发给 EP rank j 的 token 数（每步都不同，所以是"不等长"的 all-to-all，需要先同步 splits），专家算完后 `token_combine` 中用 `all_to_all(ep_group, hidden_states, self.input_splits, self.output_splits, ...)`（L870 起，splits 对调）发回。注意注释里 `[SEQL, H] -> [SEQL, H/TP]`：EP 与 TP 可以叠加。

---

## 5. 方法对比表

| 方法 | 切分对象 | 每卡显存（模型状态） | 通信原语 | 通信量（每步，每卡） | 通信时机 | 典型规模/拓扑 |
|---|---|---|---|---|---|---|
| DDP | 无（复制） | 16Ψ | all-reduce 梯度 | 2Ψ | 反向中按桶重叠 | 任意 |
| ZeRO-1 | 优化器状态 | (4 + 12/N)Ψ | reduce-scatter 梯度 + all-gather 参数 | 2Ψ | 累积边界 + step 后 | 任意 |
| ZeRO-2 | + 梯度 | (2 + 14/N)Ψ | 同上 | 2Ψ | 反向中 + step 后 | 任意 |
| ZeRO-3 / FSDP | + 参数 | 16Ψ/N | 前/反向 all-gather 参数 + reduce-scatter 梯度 | 3Ψ | 每层前/后（可预取） | 节点间也可 |
| TP（Megatron） | 权重行/列 | 16Ψ/t | 每层 2+2 次 all-reduce（激活） | ∝ s·b·h·L | 每层关键路径上 | 机内 NVLink，t≤8 |
| TP + SP | + LN/dropout 激活 | 同 TP，激活 /t | all-gather + reduce-scatter | 同 TP | 同 TP | 同 TP |
| CP（ring） | 序列（attention） | 激活 /cp | P2P 传 KV | ∝ s·h·L（KV） | 与 attention 重叠 | 长上下文 |
| CP（Ulysses） | 序列 ↔ 头 | 激活 /cp | all-to-all | ∝ s·h·L / cp | attention 前后 | 头数 ≥ cp |
| PP（1F1B） | 层 | 16Ψ/p | P2P 传激活 | ∝ s·b·h·m（只在边界） | 每个 micro-batch | 跨机，气泡 (p−1)/(m+p−1) |
| EP | MoE 专家 | 专家参数 /ep | all-to-all token | ∝ tokens·h·topk | 每个 MoE 层 2 次 | 见 07 |

---

## 6. 常见困惑 / 坑

1. **"ZeRO-3 和 FSDP 是一回事吗？"** 思想相同（参数/梯度/优化器状态全切分 + 用时 all-gather），实现不同：DeepSpeed 通过模块钩子 + `ds_tensor` 实现，FSDP2 通过 `fully_shard` + DTensor 实现，粒度由你选择哪些模块作为 FSDP 单元。
2. **"ZeRO-1/2 的通信量和 DDP 一样"**，因为 all-reduce = reduce-scatter + all-gather。ZeRO-3 是 1.5 倍。不要误以为切得越多通信越多是线性的。
3. **TP 的 f/g 写反了会怎样？** 前向结果可能是对的（例如忘了 f），但输入梯度只是一张卡的部分和，训练会悄悄发散——3.1 节最后一行演示了这一点。
4. **行并行的 bias 只能加一次**。Megatron 在 all-reduce 后加；torchtitan 把 bias 转成 Partial。自己写 TP 时这是高频 bug。
5. **`CUDA_DEVICE_MAX_CONNECTIONS`**：Megatron 风格 TP 依赖它 =1 来保证"通信先于 wgrad GEMM 调度"实现重叠；FSDP2 + 原生 TP 则应设大（torchtitan `distributed/README.md` 明确提醒不要照抄）。
6. **梯度累积 + DDP/ZeRO**：只在最后一个 micro-batch 规约（Megatron `is_last_microbatch`、DeepSpeed ZeRO-1 的 `is_gradient_accumulation_boundary()`、Accelerate 的 `no_sync`）。否则每个 micro-batch 都通信，浪费带宽。ZeRO-2 例外：它每次都规约并释放，换取梯度显存。
7. **PP 与 FSDP 叠加**：每个 micro-batch 都会触发 FSDP all-gather，所以 torchtitan 在 PP 下默认 `reshard_after_forward=False`。
8. **CP 的 causal 负载不均**：不用 zigzag 时，最后一块的 rank 计算量约是第一块的 2cp−1 倍（以块对计），整体速度由最慢的 rank 决定。
9. **GQA 与 TP**：KV 头数 < TP 度时需要复制 KV 头（Megatron attention.py 的注释），参数和 KV cache 都会有冗余。
10. **激活检查点 + dropout**：重算时 RNG 状态必须与前向一致，否则重算出的 dropout mask 不同，梯度错误。TP 下还要区分"TP 区域内各卡不同种子 / 区域外相同种子"。
11. **loss 归一化**：DP 下各 rank 的有效 token 数不同（padding / packing），按 rank 平均再平均 ≠ 全局按 token 平均。torchtitan 关掉 FSDP 的梯度除法、改用全局 token 数，就是为了这个。

---

## 7. 自测题 + 动手练习

**自测题**

1. 一个 13B 模型用 bf16 + Adam 训练，DP=32。分别计算 DDP、ZeRO-1、ZeRO-2、ZeRO-3 的每卡模型状态显存。
2. 证明 ring all-reduce 每卡通信量为 `2Ψ(N−1)/N`，并说明为什么 ZeRO-3 是 3Ψ。
3. Megatron 的 `ColumnParallelLinear` 在 TP>1、不开 SP 时为什么不调用 `copy_to_tensor_model_parallel_region`？反向 all-reduce 去哪了？为什么要这样做？
4. 写出一个 Transformer 层（attention + MLP）在 TP=t、开 SP 时前向和反向各有哪些 collective，各自张量形状是什么。
5. 1F1B 中 stage i 最多缓存多少个 micro-batch 的激活？GPipe 呢？二者气泡比例是否相同？
6. DeepSpeed ZeRO-1 和 ZeRO-2 在代码上的关键差别是哪几行？ZeRO-2 的"reduce-scatter"在 `stage_1_and_2.py` 中实际用的是哪个原语？
7. torchtitan 为什么把 `cp` 轴也加入 FSDP 的分片轴？
8. Ulysses 和 ring attention 各自的通信量和约束是什么？什么情况下选哪个？
9. `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` 下，FSDP 分片里存的是什么 dtype？优化器在哪个 dtype 上更新？

**动手练习**

1. 运行 3.1 节代码；再把 W1 改成**行切**、W2 改成**列切**，看看需要哪些额外通信才能得到正确结果（提示：GeLU 之前就需要 all-reduce）。
2. 扩展 3.1 节为 attention：Q/K/V 投影按头列切，`W_o` 行切，验证每卡独立做 attention 后 all-reduce 的结果与单卡一致。
3. 把 3.3 节改成 causal：块 `j > r` 跳过、`j == r` 用下三角掩码；然后实现 zigzag 切分（每卡拿块 r 与 2cp−1−r），统计每卡实际计算的 `QKᵀ` 元素数是否均衡。
4. 用 `torch.multiprocessing` + `gloo` 后端（CPU 即可）把 3.1 节改成真正的多进程：用 `dist.all_reduce` 实现 f/g 两个 `autograd.Function`，与 Megatron `mappings.py` 对照。
5. 写一个 1F1B 调度模拟器：输入 p、m，打印每个 stage 的 F/B 序列（如 2.6 节的表），并统计每个 stage 的峰值在途 micro-batch 数，与 `num_warmup_microbatches + 1` 对照。
6. 阅读 `deepspeed/runtime/zero/stage_1_and_2.py` 的 `get_data_parallel_partitions`（L2308–L2326），手算 `numel=10, dp=4` 时每个分区的 `(start, size)`。
