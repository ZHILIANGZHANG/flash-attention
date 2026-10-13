# 07 · Mixture-of-Experts（MoE）源码导读

> 读完本篇你应该能回答：路由器（router/gate）到底输出了什么张量？top-k 之后 token 是怎样“分发”到专家、算完再“合并”回来的？为什么工业实现都要先排序（sort/permute）再做 grouped GEMM？DeepSeek-V3 的“无辅助损失负载均衡”是怎么用一个 bias 实现的？专家并行（EP）里的 all-to-all 发送了什么？

本篇引用的仓库与 commit（所有链接都是固定 commit 的 permalink）：

| 仓库 | 本地路径 | commit |
|---|---|---|
| deepseek-ai/DeepSeek-V3 | `repos/01-attention/DeepSeek-V3` | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` |
| huggingface/transformers | `repos/00-basics/transformers` | `7912d6045eafc60bbd0b2dc33f871105b463dea2` |
| databricks/megablocks | `repos/06-moe/megablocks` | `952db33d6eac334d22c61e47a0d5d41446298784` |
| deepseek-ai/DeepEP | `repos/06-moe/DeepEP` | `93eb6eb238127e96c6d7a4a625a6dad158348509` |
| NVIDIA/Megatron-LM | `repos/13-distributed/megatron-lm` | `3bb482c487cd447a2b40a3bf076ab14cad514e13` |
| vllm-project/vllm | `repos/03-inference/vllm` | `f5fefbffe51e1cc91e0d6507d272f337473989b3` |

---

## 1. 要解决的问题 / 一句话直觉

**问题**：Transformer 里 FFN 占了大部分参数和 FLOPs。想把模型参数做大（知识容量更多），但又不想让每个 token 的计算量同比例变大。

**一句话直觉**：把一个大 FFN 拆成 E 个小 FFN（“专家”），再加一个很小的线性层（“路由器”）给每个 token 打分，**每个 token 只走分数最高的 k 个专家**，输出是这 k 个专家输出的加权和。参数量 ∝ E，单 token 计算量 ∝ k。

工程上的难点几乎全部来自“每个 token 走不同的专家”这件事：

1. **负载均衡**：路由器很容易“偏心”，少数专家被挤爆，多数专家闲置（还学不到东西）。→ aux loss / bias 调节 / capacity。
2. **稀疏计算的效率**：E 个专家各自处理一个“大小不定”的 token 子集，朴素的 `for e in experts` 循环在 GPU 上效率很差。→ sort/permute + grouped GEMM / block-sparse。
3. **分布式**：专家多到一张卡放不下时，把专家分到不同 GPU（专家并行，EP），token 必须跨卡发送到专家所在的 GPU，再发回来。→ all-to-all dispatch / combine（DeepEP）。

---

## 2. 核心原理（带形状）

记号：`T` = 本批 token 数（batch×seq 展平后），`d` = hidden size，`E` = 路由专家数，`k` = 每个 token 激活的专家数，`h` = 单个专家的 intermediate size。

### 2.1 路由（router / gate）

```
x        : [T, d]
W_g      : [E, d]                         # 路由器权重，就是一个无 bias 的 Linear
logits   = x @ W_g^T            : [T, E]
```

**(a) softmax top-k（Mixtral / Qwen3-MoE / Switch）**

```
p        = softmax(logits, -1)  : [T, E]   # 每行和为 1
w, idx   = topk(p, k)           : [T, k], [T, k]
w        = w / w.sum(-1)  (可选，Mixtral 总是做；Qwen3 由 norm_topk_prob 控制)
```

**(b) sigmoid + bias（DeepSeek-V3）**

```
s        = sigmoid(logits)      : [T, E]   # 每个专家独立打分，不互相竞争
s_choice = s + b                : [T, E]   # b: [E]，只用于“选谁”，不参与梯度
idx      = topk(s_choice, k)    : [T, k]
w        = gather(s, idx)       : [T, k]   # 注意：权重用原始 s，不含 b
w        = w / w.sum(-1) * route_scale
```

**(c) group-limited routing（DeepSeek-V2/V3 的“节点受限路由”）**：把 E 个专家分成 G 组（一般对应 G 个节点/卡组），先给每组打分（V3 用组内 top-2 分数之和），只保留 top-`topk_groups` 个组，其余组的专家分数置为 `-inf`，再做 top-k。目的：限制一个 token 最多被发往几个节点，从而降低跨节点通信量。

```
s_choice.view(T, G, E/G)  → 组分 group_score : [T, G]
保留 topk_groups 个组 → mask : [T, G] → 展开成 [T, E] → masked_fill(-inf)
```

### 2.2 专家计算与合并

每个专家是一个 SwiGLU FFN：`Expert_e(x) = W2_e (silu(W1_e x) ⊙ W3_e x)`，`W1_e, W3_e: [h, d]`，`W2_e: [d, h]`。

```
y_t = Σ_{j=1..k} w[t, j] · Expert_{idx[t,j]}(x_t)            # 路由部分
out_t = y_t + SharedExpert(x_t)                               # DeepSeek/Qwen2-MoE 的共享专家
```

**共享专家（shared experts）**：所有 token 都走、不经过路由的普通 FFN，用来承载“通用知识”，让路由专家更专注于差异化知识。DeepSeek-V3 里共享专家就是 `MLP(dim, n_shared_experts * moe_inter_dim)`。

### 2.3 负载均衡辅助损失（Switch Transformer 风格）

```
f_i = (1/(T·k)) · #{(t,j): idx[t,j] == i}     # 专家 i 实际拿到的 token 比例（不可导，来自 topk）
P_i = (1/T) · Σ_t p[t, i]                     # 专家 i 的平均路由概率（可导）
L_aux = α · E · Σ_i f_i · P_i
```

直觉：`f` 是“谁被选得多”，梯度只能从 `P` 流过去；最小化 `Σ f_i P_i` 会压低那些已经被选多的专家的概率。完全均匀时 `f_i = P_i = 1/E`，`L_aux = α`（不同实现对 `k` 的归一化略有不同，详见源码导读）。

### 2.4 无辅助损失均衡（DeepSeek-V3 的 bias 调节）

不加 loss，而是在每个训练 step 结束后，根据每个专家实际接收的 token 数 `c_i` 调整 bias：

```
b_i ← b_i + γ · sign(mean(c) − c_i)
```

被选太多的专家 bias 减小，下一步更不容易被选；反之亦然。因为 b 只影响 `topk` 的选择、不影响输出权重 `w`，所以它不会扭曲模型输出的梯度。

### 2.5 Capacity 与 token dropping

为了让每个专家的输入是固定形状（便于静态内存分配 / 通信），设置容量：

```
C = ceil(capacity_factor · T · k / E)
```

专家 i 收到的 token 超过 C 的部分直接**丢弃**（这些 token 在该专家槽位上贡献为 0，只剩残差连接）。`capacity_factor` 越小越省显存，但丢 token 越多。现代实现（megablocks 的 dMoE、Megatron 默认、DeepSeek）多为 **dropless**。

### 2.6 dispatch → compute → combine 的三种写法

1. **朴素循环**：`for e in range(E): tok = where(idx == e); y[tok] += w * Expert_e(x[tok])`。E 次小 GEMM + E 次 gather/scatter，专家多时 kernel launch 和小矩阵效率很差。
2. **sort/permute + grouped GEMM**：把 `[T, k]` 的 `idx` 展平成 `T·k` 个 (token, expert) 对，**按 expert 排序** → 得到 permuted 输入 `[T·k, d]`，同一专家的行是连续的；用一个 grouped GEMM kernel（一个 launch 处理 E 个大小不同的矩阵乘）算完；再按逆排列 un-permute，乘权重，按 token 求和。
3. **block-sparse（megablocks）**：把 permute 后的 token 每专家补齐到 128 的倍数，整个专家层看成一个块稀疏矩阵乘（`[T_pad, d] × [d, E·h]` 的块对角结构），用 block-sparse kernel 计算。

### 2.7 专家并行（EP）与 all-to-all

EP 大小为 P 时，每张卡持有 `E/P` 个专家。每张卡上的 token 先本地按目标卡排序，然后：

```
dispatch : all-to-all（变长）  每卡把发往卡 r 的 token 块发给 r       → 每卡收到所有“需要本卡专家”的 token
compute  : 本地再按 local expert 排序 → grouped GEMM
combine  : all-to-all（反向）  把结果发回 token 的原卡 → un-permute → 加权求和
```

发送量（单方向）≈ `T · k · d` 个元素（每个 (token, expert) 对一份 hidden）。DeepEP 就是把这一对 all-to-all 做成高度优化的 NVLink/RDMA kernel。

---

## 3. 最小实现（PyTorch，可直接粘贴运行）

下面的 `ToyMoE` 同时给出：softmax top-k 路由 + Switch aux loss、sigmoid+bias（DeepSeek 风格）路由与 bias 更新、朴素循环版本、sort-based dispatch 版本（含可选的 capacity drop），并用断言验证两种 dispatch 数值等价。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):                       # SwiGLU FFN
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x):                      # x: [n, d] -> [n, d]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ToyMoE(nn.Module):
    def __init__(self, d=64, h=128, E=8, k=2, n_shared=1, score="softmax"):
        super().__init__()
        self.E, self.k, self.score = E, k, score
        self.router = nn.Linear(d, E, bias=False)                  # W_g: [E, d]
        self.experts = nn.ModuleList(Expert(d, h) for _ in range(E))
        self.shared = Expert(d, h * n_shared) if n_shared else None
        self.register_buffer("e_bias", torch.zeros(E))            # 仅 sigmoid 模式使用

    def route(self, x):                                           # x: [T, d]
        logits = self.router(x).float()                           # [T, E]
        if self.score == "softmax":
            probs = logits.softmax(-1)                            # [T, E]
            topw, topi = probs.topk(self.k, dim=-1)               # [T, k]
        else:                                                     # DeepSeek-V3 风格
            probs = logits.sigmoid()
            topi = (probs + self.e_bias).topk(self.k, dim=-1).indices   # bias 只影响“选谁”
            topw = probs.gather(1, topi)
        topw = topw / topw.sum(-1, keepdim=True)                  # 归一化
        counts = torch.bincount(topi.flatten(), minlength=self.E).float()   # [E]
        f = counts / (x.size(0) * self.k)                         # 实际分配比例（不可导）
        P = probs.mean(0)                                         # 平均概率（可导）
        aux = self.E * (f * P).sum()                              # Switch aux loss
        return topw.to(x.dtype), topi, aux, counts

    @torch.no_grad()
    def update_bias(self, counts, gamma=1e-3):                    # aux-loss-free 均衡
        self.e_bias += gamma * torch.sign(counts.mean() - counts)

    def forward_naive(self, x):
        topw, topi, aux, counts = self.route(x)
        y = torch.zeros_like(x)
        for e in range(self.E):
            tok, slot = torch.where(topi == e)                    # 选中专家 e 的 (token, 第几个槽)
            if tok.numel():
                y.index_add_(0, tok, self.experts[e](x[tok]) * topw[tok, slot, None])
        return self._finish(x, y), aux, counts

    def forward_sorted(self, x, capacity=None):
        T, d = x.shape
        topw, topi, aux, counts = self.route(x)
        flat_e = topi.reshape(-1)                                 # [T*k]  第 i 个 (token,slot) 对的专家
        order = torch.argsort(flat_e, stable=True)                # [T*k]  按专家排序（稳定 => 早到的 token 在前）
        tok_of = order // self.k                                  # 排序后每一行来自哪个 token
        xs = x[tok_of]                                            # permute: [T*k, d]
        ws = topw.reshape(-1)[order]                              # [T*k]
        n_per_e = torch.bincount(flat_e, minlength=self.E)        # [E]
        start = torch.cumsum(n_per_e, 0) - n_per_e                # 每个专家段的起点
        keep = torch.ones_like(ws, dtype=torch.bool)
        if capacity is not None:                                  # "position" 策略的 token dropping
            pos = torch.arange(T * self.k, device=x.device) - start[flat_e[order]]
            keep = pos < capacity
        out = torch.zeros_like(xs)
        for e, (s, n) in enumerate(zip(start.tolist(), n_per_e.tolist())):  # ← grouped GEMM 会把这个循环变成 1 个 kernel
            if n:
                out[s:s + n] = self.experts[e](xs[s:s + n])
        out = out * (ws * keep).unsqueeze(-1)                     # 乘路由权重；被丢弃的行置 0
        y = torch.zeros_like(x).index_add_(0, tok_of, out)        # un-permute + 按 token 求和（combine）
        return self._finish(x, y), aux, counts

    def _finish(self, x, y):
        return y + self.shared(x) if self.shared is not None else y

if __name__ == "__main__":
    torch.manual_seed(0)
    for score in ("softmax", "sigmoid"):
        m = ToyMoE(score=score); x = torch.randn(32, 64)
        y1, aux1, c = m.forward_naive(x); y2, aux2, _ = m.forward_sorted(x)
        assert torch.allclose(y1, y2, atol=1e-5), score
        m.update_bias(c)
        print(score, "aux=%.4f" % aux1.item(), "counts=", c.tolist())
    y3, _, _ = m.forward_sorted(x, capacity=2)                    # 小容量 -> 会丢 token
    print("dropped diff:", (y3 - y2).abs().max().item())
```

要点：

- `order // self.k`：`topi.reshape(-1)` 的第 `i` 个元素来自 token `i // k`，所以排序后的行号经 `// k` 就得到原 token 号。transformers 的 `grouped_mm_experts_forward` 用的是完全相同的技巧（见 4.3）。
- `index_add_` 做 combine：一个 token 的 k 个专家输出被加到同一行。
- `forward_sorted` 里的 for 循环就是 grouped GEMM 想消灭的东西——它变成“一个 kernel、E 个大小不同的矩阵乘”。

---

## 4. 源码导读（按阅读顺序）

阅读路线：先看最直白的 DeepSeek-V3 推理代码 → transformers 的统一 MoE 抽象（朴素循环 vs grouped_mm）→ megablocks 的 dropless/块稀疏 → Megatron-LM 的训练级路由与 all-to-all dispatcher → DeepEP 的通信 API → vLLM 推理侧 fused_moe Triton kernel。

### 4.1 DeepSeek-V3 `Gate`：sigmoid + bias + group-limited routing

- 文件：`repos/01-attention/DeepSeek-V3/inference/model.py`，`class Gate` L535–L598
- Permalink：https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L535-L598

先看 671B 配置（`inference/configs/config_671B.json`）：`n_routed_experts=256, n_shared_experts=1, n_activated_experts=8, n_expert_groups=8, n_limited_groups=4, route_scale=2.5, score_func="sigmoid", dim=7168`。

`__init__`（L549–L564）里最值得注意的一行：

```python
self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
```

只有 dim==7168（即 V3 671B）才有 bias——这就是论文里的 aux-loss-free 均衡 bias。推理代码里它只是被加载的常量；训练时它不靠梯度更新（推理代码中没有训练逻辑，训练侧更新规则见 4.4 Megatron）。

`forward`（L566–L598）逐行：

```python
scores = linear(x, self.weight)                       # [T, d] x [E, d]^T -> [T, E]=[T,256]
if self.score_func == "softmax":
    scores = scores.softmax(dim=-1, dtype=torch.float32)
else:
    scores = scores.sigmoid()                          # V3：每个专家独立打分
original_scores = scores                               # 保存“干净”分数，后面用来取权重
if self.bias is not None:
    scores = scores + self.bias                        # 只用于选择
if self.n_groups > 1:
    scores = scores.view(x.size(0), self.n_groups, -1)        # [T, 8, 32]
    if self.bias is None:
        group_scores = scores.amax(dim=-1)                    # V2：组内最大值
    else:
        group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)  # V3：组内 top-2 之和 -> [T, 8]
    indices = group_scores.topk(self.topk_groups, dim=-1)[1]  # 选 4 个组 -> [T, 4]
    mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)  # 未选中组=True
    scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)  # [T, 256]
indices = torch.topk(scores, self.topk, dim=-1)[1]    # [T, 8]
weights = original_scores.gather(1, indices)          # 用不含 bias 的分数作权重
if self.score_func == "sigmoid":
    weights /= weights.sum(dim=-1, keepdim=True)      # sigmoid 分数不归一，需手动归一
weights *= self.route_scale                           # ×2.5
return weights.type_as(x), indices
```

为什么 sigmoid 要手动归一？softmax 的输出天然和为 1，而 sigmoid 每个值在 (0,1) 独立，8 个加起来可能是任意值；归一化后再乘 `route_scale` 统一缩放路由输出的量级。

### 4.2 DeepSeek-V3 `Expert` / `MoE`：最朴素的循环 + 共享专家 + EP

- `class Expert` L601–L633：https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L601-L633
- `class MoE` L636–L693：https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L636-L693

`Expert.forward` L633：`return self.w2(F.silu(self.w1(x)) * self.w3(x))`——标准 SwiGLU。

`MoE.__init__`（L649–L667）体现了“最简单的专家并行”：

```python
self.n_local_experts = args.n_routed_experts // world_size
self.experts_start_idx = rank * self.n_local_experts
self.experts = nn.ModuleList([Expert(...) if self.experts_start_idx <= i < self.experts_end_idx else None
                              for i in range(self.n_routed_experts)])   # 不属于本卡的专家是 None
self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)  # 共享专家
```

`MoE.forward`（L669–L693）：

```python
x = x.view(-1, self.dim)                                   # [T, d]
weights, indices = self.gate(x)                            # [T, k], [T, k]
y = torch.zeros_like(x)
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # GPU->CPU 同步
for i in range(self.experts_start_idx, self.experts_end_idx):   # 只循环本卡专家
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)                   # idx: token 号, top: 第几个槽
    y[idx] += expert(x[idx]) * weights[idx, top, None]     # gather -> 专家 -> 加权 -> scatter
z = self.shared_experts(x)
if world_size > 1:
    dist.all_reduce(y)                                     # 各卡只算了自己专家的部分，求和
return (y + z).view(shape)
```

注意这里的“EP”并不是 all-to-all：**每张卡都拿到全部 token**（推理时 batch 小），只计算本卡专家的贡献，最后 `all_reduce` 求和。它简单但通信量是 `T·d` 的 all-reduce，且每卡要处理全部 token 的路由。训练/大批量推理会改用 all-to-all（4.4、4.5）。

`y[idx] += ...` 能用是因为同一个专家内 `idx` 不重复（一个 token 不会两次选同一专家）。

### 4.3 transformers：Router / Experts 解耦与 `grouped_mm` 实现

#### (1) Mixtral：softmax top-k 路由

- `MixtralTopKRouter` L95–L110：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/mixtral/modeling_mixtral.py#L95-L110

```python
router_logits = F.linear(hidden_states, self.weight)                         # (T, E)
router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (T, k)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)              # Mixtral 总是重新归一
return router_logits, router_scores, router_indices
```

返回 `router_logits` 是为了给 aux loss 用（需要整行 softmax，不只是 top-k）。

- `MixtralSparseMoeBlock` L113–L129：训练时可选 jitter 噪声（`hidden_states *= uniform(1-ε, 1+ε)`），然后 `gate → experts`，没有共享专家。

#### (2) `MixtralExperts`：3D 权重 + 朴素循环（默认 eager 实现）

- L55–L92：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/mixtral/modeling_mixtral.py#L55-L92

权重不再是 `ModuleList`，而是两个 3D 参数：`gate_up_proj: [E, 2h, d]`、`down_proj: [E, d, h]`——这正是 grouped GEMM 需要的布局。

```python
expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)  # [T, k, E+1]
expert_mask = expert_mask.permute(2, 1, 0)                                                # [E+1, k, T]
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()                   # 有 token 的专家
for expert_idx in expert_hit:
    expert_idx = expert_idx[0]
    if expert_idx == self.num_experts:      # 第 E 类是 EP 场景下“不属于本卡”的哨兵
        continue
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])     # [k, T] 上的非零位置
    current_state = hidden_states[token_idx]
    gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    current_hidden_states = self.act_fn(gate) * up
    current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
```

`num_classes = E + 1`：EP 时不属于本卡的专家 id 会被映射成哨兵值 `E`，one-hot 多留一类就不会越界。

#### (3) `@use_experts_implementation`：同一个 Experts 类，多种 forward

- `ExpertsInterface` L500–L525 与 `wrapper` L573–L602：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/integrations/moe.py#L500-L602

```python
_global_mapping = {
    "deepgemm": deepgemm_bf16_experts_forward,
    "batched_mm": batched_mm_experts_forward,
    "grouped_mm": grouped_mm_experts_forward,
    "sonicmoe": sonicmoe_experts_forward,
}
...
def forward(self, *args, **kwargs):
    experts_forward = experts_interface.get_interface(self.config._experts_implementation, original_forward)
    return experts_forward(self, *args, **kwargs)
```

装饰器把 `forward` 换成“查表分派”：`config._experts_implementation` 为 `"grouped_mm"` 时走 sort 版本，否则退回类里写的朴素循环。

#### (4) `grouped_mm_experts_forward`：sort → grouped GEMM → unsort

- L388–L497：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/integrations/moe.py#L388-L497

这段几乎就是第 3 节 `forward_sorted` 的工业版：

```python
sample_weights = top_k_weights.reshape(-1)        # (S,)  S = T*k
expert_ids = top_k_index.reshape(-1)              # (S,)
expert_ids_g, perm = torch.sort(expert_ids)       # 按专家排序
selected_hidden_states_g = hidden_states[perm // num_top_k]   # permute: (S, d)
sample_weights_g = sample_weights[perm]
tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)   # 每个专家段的结束位置
...
proj_out = _grouped_linear(selected_hidden_states_g, selected_weights, offsets, ...)  # (S, 2h)
proj_out = self._apply_gate(proj_out)                                                # (S, h)
proj_out = _grouped_linear(proj_out, self.down_proj, offsets, ...)                   # (S, d)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(perm.size(0), device=device)
weighted_out = weighted_out[inv_perm]                                                # unsort
final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
```

几个“为什么”：

- **为什么用 `histc` 不用 `bincount`**：注释写明 `bincount` 会破坏 CUDA graph（输出长度依赖数据）；`histc` 输出固定 `bins` 长度。
- **为什么 combine 用 `view(T, k, d).sum(1)` 而不是 `index_add_`**：unsort 后第 `t*k + j` 行正好是 token t 的第 j 个专家输出，reshape 即可；注释说明 `index_add_` 在 CUDA 上用 atomicAdd，不确定性且在 bf16 中累加。
- **EP 哨兵**：`sentinel_mask` 那段注释解释了不属于本卡的专家 id（≥E）经排序被推到尾部，`histc(max=E-1)` 不计数，grouped_mm 只读 `offsets[-1]` 之前的行，不付出任何 GEMM 计算。

`_grouped_mm`（L319 起）优先调用 `torch.nn.functional.grouped_mm` / `torch._grouped_mm`；不可用时走 `_grouped_mm_fallback`（L185–L208），后者就是把 for 循环写成 `torch.mm(input[start:end], weight[i], out=output[start:end])`——让你一眼看懂 grouped GEMM 的语义：**输入按行切成 E 段，第 i 段乘第 i 个专家的权重**。

#### (5) Qwen3-MoE 与 DeepSeek-V3 的 transformers 版本

- `Qwen3MoeTopKRouter` L248–L266：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L248-L266 ——与 Mixtral 一样，只是重新归一化受 `norm_topk_prob` 控制。
- `DeepseekV3TopkRouter` L120–L158：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L120-L158 ——bias 叫 `e_score_correction_bias`，是 `nn.Buffer`（不是 Parameter，不吃梯度）；`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`（L551）保证它始终 fp32。group mask 写法与官方推理代码等价（`scatter_(1, group_idx, 1)` 再 `~score_mask`）。
- `DeepseekV3MoE` L201–L222：`hidden_states = self.experts(...) + self.shared_experts(residuals)`，共享专家吃的是 MoE 输入（变量名 `residuals` 只是指“原始输入”）。

#### (6) aux loss：`load_balancing_loss_func`

- L474–L540：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/mixtral/modeling_mixtral.py#L474-L540

```python
routing_weights = torch.nn.functional.softmax(layer_gate.to(compute_device), dim=-1)    # [T, E]
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)                          # [T, k]
tokens_per_expert_sum += torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
router_prob_sum += routing_weights.float().sum(dim=0)
total_rows += routing_weights.shape[0]
...
tokens_per_expert = tokens_per_expert_sum / total_rows          # 注意：除以 T，不是 T*k -> Σ_i f_i = k
router_prob_per_expert = router_prob_sum / total_rows
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
return overall_loss * num_experts
```

它在 `MixtralForCausalLM.forward` 中通过 `loss += self.router_aux_loss_coef * aux_loss`（L629）加到主 loss。注意它把所有层的 `router_logits`（需要 `output_router_logits=True`）一起统计，并支持用 `attention_mask` 排除 padding。

### 4.4 megablocks：dMoE（dropless）、capacity 版 MoE、块稀疏与 grouped GEMM

#### (1) 有容量的 `ParallelMLP`

- `expert_capacity` L133–L136：https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/moe.py#L133-L136

```python
tokens_per_expert = (self.top_k * tokens * world_size / self.num_experts)
return int(self.args.moe_capacity_factor * tokens_per_expert)
```

- `indices_and_bins` L152–L183：`ops.sort(top_expert, ...)` 得到 `bin_ids`（排序后的专家号）与 `indices`（排序置换），`ops.histogram` 得 `tokens_per_expert`，`inclusive_cumsum` 得 `bins`（每个专家段的结束位置）。和 transformers 的 `sort/histc/cumsum` 一一对应。
- `permute_and_compute` L185–L207：`ops.binned_gather(x, indices, bins, expert_capacity, top_k)` 把 token 填进 `[E, C, d]` 的定长桶；`self.mlp(x)` 是批量矩阵乘（每专家一个 `[C, d]`）；`ops.binned_scatter` 带权重写回。**超过容量的 token 怎样被丢？** 看 Triton kernel `_binned_copy`（`megablocks/backend/kernels.py` L326 起，https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/backend/kernels.py#L314-L360）：grid 是 `(num_experts, expert_capacity)`，每个 program 处理一个槽位 `entry_idx`；序号 ≥ C 的 token 根本没有 program 去拷贝它——这就是 drop。排序是稳定的，所以被丢的是“位置靠后”的 token。
- `forward_once` L209–L235：`expert_capacity == 0` 时取 `max(tokens_per_expert)`，即“动态容量=不丢”。

#### (2) aux loss

- `load_balancing_loss` L138–L150 与 `batched_load_balancing_loss` L32–L89：https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/moe.py#L32-L150

```python
scale = self.num_experts / (tokens * self.top_k)
return scale * torch.dot(tokens_per_expert.to(expert_scores.dtype), expert_scores.mean(dim=0))
```

即 `E/(T·k) · Σ_i count_i · P_i = E · Σ_i f_i P_i`（这里 `f_i = count_i/(T·k)`，与 transformers 版本差一个因子 k）。`forward`（L425–L437）只是把 `(tokens_per_expert, scores)` 存进全局列表，`batched_load_balancing_loss` 在训练脚本里一次性按所有层计算。

#### (3) dMoE：dropless + 块稀疏

- `ParallelDroplessMLP` L18–L320，`dMoE` L323–L326：https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/dmoe.py#L18-L326

`indices_and_padded_bins`（L131–L154）多了一步：

```python
padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)   # blocking=128
padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
```

每个专家的 token 数补齐到 128 的倍数，这样 permute 后的矩阵 `[T_pad, d]` 能被切成 128×128 的块。`sparse_forward_once`（L156–L193）：

```python
x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.top_k)  # permute + 每段补零
topo = self.topology(x, padded_bins)                                       # 构造块稀疏拓扑
x = self.mlp(x, topo)                                                      # SparseGLU：stk.ops.sdd / dsd
x = ops.padded_scatter(x, indices, bin_ids, expert_weights, bins, padded_bins, self.top_k)
```

`topology`（L68–L129）构造一个 `[T_pad, h·E_local]` 的块稀疏矩阵：第 r 个 128 行块只在“它所属专家”的那 `h/128` 个列块上非零（`blocks_per_row = ffn_hidden_size // blocking`）。`SparseGLU.forward`（`megablocks/layers/glu.py` L47–L63，https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/glu.py#L47-L63）：

```python
x1 = stk.ops.sdd(x, w1.t(), topo)   # dense x dense -> 只算 topo 中的块（Sparse 输出）
x2 = stk.ops.sdd(x, v1.t(), topo)
x1 = stk.ops.mul(act_fn(x1), x2)
return stk.ops.dsd(x1, w2)          # sparse x dense -> dense
```

另一条路 `grouped_permute_and_compute`（dmoe.py L260–L280）+ `GroupedGLU.forward`（glu.py L175–L205）直接用 `grouped_gemm` 库：

```python
x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)    # batch_sizes = 每专家 token 数（CPU 张量）
x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)
x1 = self.args.activation_fn(x1) * x2
return gg.ops.gmm(x1, w2, batch_sizes)
```

grouped GEMM 不需要补齐到 128；块稀疏方案需要补齐但对 kernel 要求更统一。两者都是 dropless。

#### (4) megablocks 的专家并行：三次 permute

- `parallel_forward_once` L237–L423：https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/moe.py#L237-L423

函数开头的注释把 EP 讲得非常清楚：(1) 本地按专家排序，使发往同一远端卡的 token 连续；(2) all-to-all 跨卡置换；(3) 收到的 token 按“来源卡”分块，需要再本地按专家排序。关键代码：

```python
tpe_handle = dist.all_to_all_single(parallel_tokens_per_expert, repeated_tokens_per_expert, ...)  # 先交换“计数”
x = ops.gather(x, indices, bin_ids, bins, self.top_k)                    # 第 1 次 permute
send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)               # 发往每张卡多少行
recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)                 # 从每张卡收多少行
parallel_x, parallel_x_handle = all_to_all(x, recv_counts, send_counts, ...)   # 第 2 次：跨卡
parallel_bin_ids, parallel_indices = ops.sort(parallel_top_expert, ...)  # 第 3 次 permute 的索引
parallel_x = self.permute_and_compute(parallel_x, ..., top_k=1)          # 本地专家计算
x, _ = all_to_all(parallel_x, send_counts, recv_counts, ...)             # 反向 all-to-all
x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k)   # 本地 un-permute + 加权
```

为什么要先 all-to-all 一次“计数”？因为变长 all-to-all 需要每卡事先知道要收多少数据才能分配接收 buffer，这一步还需要 `.cpu()`（GPU→CPU 同步），是 EP 的经典开销点。

### 4.5 Megatron-LM：训练级路由、capacity、bias 更新与 all-to-all dispatcher

#### (1) `TopKRouter.routing`：一切路由策略的总入口

- L1013–L1195：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/router.py#L1013-L1195

主干（去掉诊断代码后）：

```python
logits = logits.view(-1, self.config.num_moe_experts)                 # [T, E]
logits = self.apply_z_loss(logits, padding_mask=padding_mask)          # z-loss：惩罚 logits 过大
...
probs, routing_map = topk_routing_with_score_function(
    logits, self.topk, use_pre_softmax=..., num_groups=self.config.moe_router_num_groups,
    group_topk=self.config.moe_router_group_topk, scaling_factor=...,
    score_function=self.score_function, expert_bias=self.expert_bias, ...)
if self.config.moe_expert_capacity_factor is not None:
    probs, routing_map = apply_router_token_dropping(probs, routing_map, ...)   # capacity
...
probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, ...)
...
self._apply_expert_bias(routing_map, padding_mask=padding_mask)        # 累计每个专家的 token 数
return probs, routing_map
```

注意 Megatron 的输出格式与前面不同：`probs` 与 `routing_map` 都是 `[T, E]`（稀疏的 dense 表示，`routing_map` 是 bool），而不是 `[T, k]` 的索引。

#### (2) `topk_routing_with_score_function` 与 `group_limited_topk`

- L787–L987：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L787-L987
- `group_limited_topk` L692–L747：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L692-L747

softmax 分支（L930 起）有两种顺序：`use_pre_softmax=True` 先对全部 E 做 softmax 再 top-k（Mixtral 语义），否则先 top-k logits 再对 k 个做 softmax（权重自动归一）。sigmoid 分支（L953–L959）：

```python
elif expert_bias is not None:
    scores_for_routing = scores + expert_bias.float()                  # 选择用带 bias 分数
    _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
    scores = torch.gather(scores, dim=1, index=top_indices)            # 权重用原始分数
...
probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
```

最后（L985–L987）把 `[T,k]` scatter 回 `[T,E]`：`routing_probs = zeros_like(logits).scatter(1, top_indices, probs)`。

`group_limited_topk` 的 docstring 把用途写得很直白：`num_groups = EP size` 即“设备受限路由”（DeepSeek-V2），`num_groups = 节点数` 即“节点受限路由”（DeepSeek-V3）。它用 `topk(topk // group_topk)` 的和作为组分（V3：8/4=2，正好是 top-2 之和）。

#### (3) aux loss 与“把 loss 挂到计算图上”的技巧

- `switch_load_balancing_loss_func` L65–L160：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L65-L160

```python
aggregated_probs_per_expert = probs.sum(dim=0)                                   # Σ_t p[t,i]
aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
    num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens))
```

= `coeff · E · Σ_i (count_i/(k·T)) · (Σ_t p_ti / T)`，和第 2.3 节公式一致。docstring 还说明了在序列/上下文并行下 `tokens_per_expert` 应取全局计数、`probs` 取本地（global-batch 级 aux loss）。

- `MoEAuxLossAutoScaler` L313–L365：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L313-L365

`forward(output, aux_loss)` 原样返回 `output`，`backward` 给 `aux_loss` 返回一个常数梯度 `ones * scale`。效果：不需要把 aux loss 一路 return 到训练循环里去加，只要 `probs` 参与了后续计算，反向时 aux loss 的梯度就会自动注入。这是 Megatron 的 `attach_and_log_load_balancing_loss` 用到的机制。

#### (4) capacity：`get_capacity` 与 `apply_router_token_dropping`

- L258–L276 / L1065–L1128：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L1065-L1128

```python
expert_capacity = get_capacity(num_tokens=num_tokens * router_topk, num_experts=num_experts,
                               capacity_factor=capacity_factor)              # ceil(T*k/E * cf)
if drop_policy == "probs":
    _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)  # 按列 top-C
elif drop_policy == "position":
    _, capacity_indices = torch.topk(routing_map.int(), k=expert_capacity, dim=0, sorted=False)
capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
final_map = torch.logical_and(routing_map, capacity_mask)
```

`dim=0` 很关键：在 `[T, E]` 上**按列**（每个专家）取 top-C。`"probs"` 策略保留该专家概率最高的 C 个 token；`"position"` 保留靠前的。`pad_to_capacity=True` 时直接用 `capacity_mask` 作为 map，即每个专家固定恰好 C 个槽（不足的用低分 token/空位填满），换来静态形状。

#### (5) aux-loss-free：bias 在哪里更新？

- `_apply_expert_bias` L909–L946（router.py）：训练前向中把 `routing_map.sum(dim=0)` 累加到 buffer `local_tokens_per_expert`（L946）。buffer 本身在 `__init__` L225–L248 注册：`expert_bias` 是 fp32 buffer。
- `get_updated_expert_bias` L1245–L1281（moe_utils.py）：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_utils.py#L1245-L1281

```python
torch.distributed.all_reduce(tokens_per_expert, group=tp_dp_cp_group)      # 全局计数
total_tokens = tokens_per_expert.sum(dim=-1, keepdim=True)
update_direction = torch.sign(total_tokens - tokens_per_expert * num_experts)   # = sign(mean - c_i)
updated_expert_bias = expert_bias + update_direction.to(dtype=expert_bias.dtype) * expert_bias_update_rate
```

调用点在 `megatron/core/distributed/finalize_model_grads.py` 的 `_update_router_expert_bias`（L377–L414，https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/distributed/finalize_model_grads.py#L377-L414），即**每个 global batch 结束、梯度同步之后**更新一次；默认 `moe_router_bias_update_rate = 1e-3`（`transformer_config.py` L919）。这正是 2.4 节公式的实现。

#### (6) `MoEAlltoAllTokenDispatcher`：dispatch / combine 的 6 个阶段

- 类定义 L381 起，关键方法 L507–L914：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/token_dispatcher.py#L507-L914

`MoELayer.forward` 内的 `custom_forward`（`moe_layer.py` L725–L768，https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/moe/moe_layer.py#L725-L768）把流程串起来：

```python
shared_expert_output = self.shared_experts_compute(hidden_states)
probs, routing_map = self.route(hidden_states, ...)
hidden_states, probs = self.preprocess(hidden_states, probs, routing_map, padding_mask)  # -> dispatch_preprocess
dispatched_input, probs = self.dispatch(hidden_states, probs)                           # -> token_dispatch (A2A)
output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)  # dispatch_postprocess -> experts -> combine_preprocess
output = self.combine(output)                                           # -> token_combine (A2A)
output = self.postprocess(output, shared_expert_output)                 # -> combine_postprocess + shared
```

逐阶段：

1. `preprocess`（L507–L628）：`num_local_tokens_per_expert = routing_map.sum(dim=0)`；`input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1)`——发往每个 EP rank 的行数；再 all-gather 全局计数得到 `output_splits`（从每个 rank 收多少行）和 `num_global_tokens_per_local_expert`。dropless 时 `num_out_tokens = T·k` 是静态的，不需要同步。
2. `dispatch_preprocess`（L630–L686）：调用 `permute(hidden_states, routing_map, probs=probs, ...)`。`permute`（moe_utils.py L366 起）的非 drop 分支（L483–L508）：

   ```python
   routing_map = routing_map.bool().T.contiguous()                          # [E, T]
   flat_sorted = routing_map.reshape(-1).argsort(descending=True, stable=True)
   flat_sorted = flat_sorted[:num_out_tokens]                               # 取所有 True 的位置（按专家顺序）
   sorted_indices = flat_sorted % num_tokens                                # 对应 token 号
   permuted_input = tokens.index_select(0, sorted_indices)                  # [T*k, d]
   ```

   转置后按行展平就是“按专家、专家内按 token 顺序”，`argsort(descending)` 把 True 排前面且是固定形状的（对 CUDA graph 友好）——这是不用 `nonzero` 的 permute 技巧。因为专家按 id 连续分布在各 rank，按专家排序 == 按目标 rank 排序。
3. `token_dispatch`（L688–L732）：`all_to_all(self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits)`，probs 同样发一份。注释说明了把共享专家的 fc1 插在 tokens A2A 和 probs A2A 之间以重叠通信。
4. `dispatch_postprocess`（L734–L800）：收到的数据按“来源 rank × 本地专家”排列，`sort_chunks_by_idxs` 重排成“本地专家 × 来源 rank”，使同一本地专家的行连续（对应 megablocks 的第 3 次 permute）。
5. 专家计算 `TEGroupedMLP.forward`（`experts.py` L931 起）：`fc1_output = self.linear_fc1(permuted_local_hidden_states, tokens_per_expert)`（L1023，TE 的 GroupedLinear）→ 激活 → **`intermediate_parallel * permuted_probs`**（L1052）→ `linear_fc2`（L1145）。注意 Megatron 在 fc2 之前乘路由权重：因为 fc2 是线性且无 bias，`W2(p·a) = p·W2(a)`，提前乘可以和激活函数融合（`weighted_bias_swiglu_impl`）。
6. `combine_preprocess`（L802–L841）逆重排；`token_combine`（L843–L880）反向 all-to-all（splits 互换）；`combine_postprocess`（L882–L914）调用 `unpermute` 按 token 求和并加上共享专家输出。

Megatron 还有 `MoEFlexTokenDispatcher`（L2022 起），其 `_DeepepManager`（L1306 起）用 DeepEP 的融合 dispatch/combine 取代上面 2–4 和 6 中的 permute + A2A。

### 4.6 DeepEP：dispatch / combine 在 API 层面做了什么

- `EPBuffer.dispatch` L562 起：https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/deep_ep/buffers/ep.py#L562-L700
- `EPBuffer.combine` L770–L844：https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/deep_ep/buffers/ep.py#L770-L844
- README 接口示例 L105–L290：https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/README.md#L105-L290

**版本提示**：本仓库是 DeepEP 2.5（`deep_ep/__init__.py` 中 `__version__ = '2.5.0'`）。README 的 release notes 写明 V2.5 “Fully remove V1”，并且 “High-throughput and low-latency APIs unified into a single `EPBuffer` interface”。所以你在本地 submodule 里**找不到**老的 `Buffer.low_latency_dispatch`；老 API 的调用方式可以在 vLLM 里看到（见下文）。

`dispatch` 的输入/输出（来自 docstring）：

- `x`: `[num_tokens, hidden]` bf16，或 `(fp8 数据, scale)` 二元组（FP8 dispatch，通信量减半）。
- `topk_idx`: `[num_tokens, num_topk]`，`-1` 表示不选；`topk_weights`: `[num_tokens, num_topk]` fp32。
- `do_expand=True`：“expanding layout（one slot per expert per token）”——输出直接是“按本地专家分段、每段对齐到 `expert_alignment`”的布局，可以直接喂 grouped GEMM（例如 DeepGEMM 的 contiguous layout），省掉 Megatron 那样的二次 permute。
- `do_cpu_sync`：训练/prefill 用 True（拿到精确的接收数量，按实际大小分配）；decode 用 False（输出按最大容量 `num_max_tokens_per_rank` 分配，用 GPU 上的 `psum_num_recv_tokens_per_expert` 判断有效范围，**不需要 GPU→CPU 同步，因而可以被 CUDA graph 捕获**）。
- 返回 `recv_x, recv_topk_idx, recv_topk_weights, handle, event`。`handle` 记录了路由布局，`combine` 和反向都要用。

`combine(x, handle, ...)`：把专家输出按 handle 发回原 rank 并**归约**（同一 token 的 k 份输出求和），可选 `bias` 在 epilogue 里顺便加上共享专家输出或残差。README 明确：“The backward pass of MoE dispatch is actually a combine”，“The backward pass of MoE combine is actually a dispatch”——dispatch 与 combine 互为转置。

**normal（high-throughput）vs low-latency** 的区别在概念上是：

| | normal / HT | low-latency / LL |
|---|---|---|
| 场景 | 训练、prefill（每卡 token 多） | decode（每卡 token 少，延迟敏感） |
| 布局 | 先 `get_dispatch_layout` 算出每 rank/每专家计数，按实际数量发送；节点内 NVLink、节点间 RDMA 分层转发 | 按固定最大 token 数预分配，纯 RDMA 点对点直写；输出 `[num_local_experts, max_tokens, hidden]` 的批量格式 |
| CPU 同步 | 需要（知道收多少） | 不需要，可 CUDA graph |
| 重叠 | event 异步 | recv hook，通信不占 SM |

在 vLLM 里两种模式分别对应两个 PrepareAndFinalize 类（使用 DeepEP V1 风格 API）：

- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`：`DeepEPHTPrepareAndFinalize`（L29），`self.buffer.get_dispatch_layout(...)`（L134）→ `self.buffer.dispatch(...)`（L153）→ `self.buffer.combine(...)`（L376）；`activation_format` 返回 `Standard`（L85–L86）。https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py#L125-L175
- `.../prepare_finalize/deepep_ll.py`：`DeepEPLLPrepareAndFinalize`（L50），`self.buffer.low_latency_dispatch(a1, topk_ids, self.max_tokens_per_rank, num_experts, use_fp8=..., return_recv_hook=True)`（L294）→ `self.buffer.low_latency_combine(...)`（L395）；`activation_format` 返回 `BatchedExperts`（L142–L143），即 `(num experts, max tokens per expert, hidden dim)`（定义见 `modular_kernel.py` L89–L96）。https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py#L280-L310

### 4.7 vLLM：推理侧 fused_moe

#### (1) 模块化结构

- `modular_kernel.py` 顶部注释 L52–L86：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/modular_kernel.py#L52-L86

```
[Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
```

`FusedMoEPrepareAndFinalize`（L183 起）负责量化 + dispatch / combine（单卡是 no-op，多卡可以是 DeepEP HT/LL 等），`FusedMoEExperts`（L475 起）负责“permute→GEMM→act→GEMM→unpermute”。`FusedMoEKernelModularImpl.apply`（L1426 起）就是 `_prepare → _fused_experts → _finalize` 三步。这种拆法让 “N 种通信 × M 种 GEMM 后端” 不需要写 N×M 份代码。

#### (2) 路由：`grouped_topk`

- `router/grouped_topk_router.py` L81–L162：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py#L81-L162

和 DeepSeek 官方 `Gate` 逐行对应：有 `e_score_correction_bias` 时组分用 top-2 之和，否则用 max；选择用带 bias 的分数，`topk_weights = original_scores.gather(1, topk_ids)`；有 CUDA 融合版本 `fused_grouped_topk` 时优先走融合 kernel。返回 `topk_weights: fp32 [T,k]`、`topk_ids: int32 [T,k]`。

#### (3) `moe_align_block_size`：推理版的 sort + pad

- `moe_align_block_size.py` L55 起，docstring 的例子 L112–L123：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/moe_align_block_size.py#L55-L123

输出三样东西：

- `sorted_token_ids`：把 `T·k` 个 (token,slot) 的**展平编号**按专家排序，每个专家段补齐到 `BLOCK_SIZE_M` 的倍数，补的位置填 `T·k`（一个越界的“假 token”）。
- `expert_ids`：每个 `BLOCK_SIZE_M` 行块属于哪个专家（EP 下不属于本卡的为 -1）。
- `num_tokens_post_padded`：补齐后的总长度。

docstring 的例子：`topk_ids = [[2,3,4],[1,2,4],[1,3,4],[1,2,3]]`，block_size=4 → 排序结果 `[3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]`，其中 12 是 padding。这与 megablocks 的 `padded_gather` 思想一致，但**不实际搬运数据**，只生成索引。

`_prepare_expert_assignment`（fused_moe.py L1541–L1587）里还有一个推理特有的优化：当 `num_tokens * top_k * 4 <= global_num_experts`（decode 时 token 很少、专家很多）就跳过对齐，走 `naive_block_assignment`——每个 block 只放一个 token。

#### (4) Triton kernel：`fused_moe_kernel`

- `fused_moe.py` L297–L608：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/fused_moe.py#L297-L608

docstring：A `(*, K)` 是 token，B `(E, N, K)` 是堆叠的专家权重，C `(M, topk, N)` 是输出。核心几行：

```python
if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:   # L404：超出补齐长度的 program 直接退出
    return
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)   # L408：本块要处理的 BLOCK_M 个“展平编号”
token_mask = offs_token < num_valid_tokens                   # L419：padding(=T*k) 被 mask 掉
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)   # L421：整块共享一个专家
...
a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)  # L471：// top_k 得原 token 行
b_ptrs = (b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))  # 选专家权重
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    ...
    accumulator += tl.dot(a, b)                               # L561
if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]                        # L597：在 fp32 中乘路由权重
c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
tl.store(c_ptrs, accumulator, mask=c_mask)                    # L608：写回 C 的第 offs_token 行
```

这就是“**gather 融合进 GEMM 的 A 读取、scatter 融合进 C 的写回**”：A 不需要被 permute 成新张量（直接按 `offs_token // top_k` 读原始行），C 按展平编号写到 `(T, k, N)` 的对应位置。`expert_ids == -1`（EP 下非本卡专家）时写零后返回（L421 之后的分支）。

#### (5) `fused_experts_impl`：两次 kernel + act + moe_sum

- L1652–L1855：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/fused_moe/fused_moe.py#L1652-L1855

```python
intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)   # W1 输出（gate+up, N=2h）
intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)   # W2 输出，复用同一块内存
sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(...)
dispatch_fused_moe_kernel(qhidden_states, w1, intermediate_cache1, ..., apply_router_weight_on_input, top_k_num, ...)
apply_moe_activation(activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N))   # silu(gate)*up
dispatch_fused_moe_kernel(qintermediate_cache2, w2, intermediate_cache3, ..., not apply_router_weight_on_input, 1, ...)
ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()), out_hidden_states)       # (M,k,K) -> (M,K)
```

注意两次调用的区别：第一次 `top_k=top_k_num`（A 的行要 `// top_k` 映射回原 token），第二次 `top_k=1`（中间结果已经是 `M*k` 行，每行独立）；路由权重只在其中一次乘（默认第二次，`MUL_ROUTED_WEIGHT = not apply_router_weight_on_input`）。cache1 与 cache3 共用内存是因为算 cache3 时 cache1 已经用完——推理侧对显存的抠门。

---

## 5. 不同实现对比

| 实现 | 路由 | dispatch 方式 | 专家计算 | 丢 token? | 均衡手段 | EP 通信 |
|---|---|---|---|---|---|---|
| DeepSeek-V3 `inference/model.py` | sigmoid + bias + group-limited | `where(indices==i)` 循环 | 每专家一个 `Expert` 模块 | 否 | bias（推理只读） | 每卡全量 token + `all_reduce` |
| transformers eager | softmax top-k / DS 风格 | `one_hot` + 循环 + `index_add_` | 3D 权重切片 `F.linear` | 否 | `load_balancing_loss_func` | 哨兵 id 跳过非本卡专家 |
| transformers `grouped_mm` | 同上 | `sort` + `histc` + `cumsum` | `torch._grouped_mm` | 否 | 同上 | 同上 |
| megablocks `MoE` | softmax top-k | `binned_gather` 到 `[E,C,d]` | bmm | 是（capacity） | aux loss | 三次 permute + `all_to_all` |
| megablocks `dMoE` | 同上 | `padded_gather`（补到 128） | 块稀疏 `sdd/dsd` 或 `gg.ops.gmm` | 否 | aux loss | 同上 |
| Megatron-LM | softmax/sigmoid/sqrtsoftplus，group-limited，bias，sinkhorn… | `argsort` 固定形状 permute | TE GroupedLinear | 可选（probs/position，可 pad） | aux / seq-aux / global-aux / z-loss / expert bias | AlltoAll / AllGather / Flex(DeepEP, HybridEP) |
| DeepEP | （不管路由） | 融合 permute+A2A，expand layout | （交给 DeepGEMM 等） | 否 | — | NVLink + RDMA 定制 kernel；HT vs LL |
| vLLM fused_moe | `fused_topk` / `grouped_topk` | `moe_align_block_size` 只生成索引 | Triton kernel 内 gather/scatter | 否 | —（推理） | PrepareAndFinalize 插件（DeepEP HT/LL 等） |

---

## 6. 常见困惑 / 坑

1. **“top-k 不可导，路由器怎么学？”** 梯度通过权重 `w = gather(p, idx)` 流回 router：输出是 `Σ w_j · Expert(x)`，`w_j` 对 logits 可导。被选中的专家的概率会被推高或压低；没被选中的专家只能通过 softmax 的归一化（或 aux loss 的 `P_i`）间接得到梯度。
2. **DeepSeek 的 bias 为什么不能参与权重？** 若权重也用 `s + b`，bias 会直接改变输出、模型就会把 bias 当作可利用的参数，均衡调节就失去了“只影响选择”的中立性。所以所有实现都是 “用 `s+b` 选，用 `s` 加权”。
3. **aux loss 的 `f_i` 到底除以 T 还是 T·k？** transformers 除以 T（`Σ f_i = k`），megablocks 与 Megatron 除以 T·k（`Σ f_i = 1`）。二者相差常数 k，只影响 `aux_loss_coef` 的合适取值；换框架复现时要换算。
4. **`index_add_` vs `reshape().sum()`**：前者在 CUDA 上非确定性、在低精度中累加；后者确定性。transformers 的 grouped_mm 版本特地选了后者。
5. **capacity 下被丢弃的 token 去哪了？** MoE 输出对它是 0（或少了某个专家的贡献），但残差连接 `x + MoE(x)` 仍然把 x 传下去。所以 drop 不会让 token “消失”，只是这一层没有 FFN 更新。
6. **推理时 drop token 很危险**：推理 batch 组成与训练不同，capacity 可能导致同一 prompt 在不同 batch 中结果不同；vLLM 等推理框架都是 dropless。
7. **`.tolist()` / `.cpu()` 的隐藏同步**：DeepSeek 推理代码的 `bincount(...).tolist()`、megablocks 的 `send_counts.cpu()`、transformers fallback 的 `offs.tolist()` 都会让 CPU 等 GPU。Megatron 专门有 `_maybe_dtoh_and_synchronize` 和 `cuda_dtoh_point` 管理同步点；DeepEP 的 `do_cpu_sync=False` 也是为了绕开它。
8. **共享专家与路由专家的输入**：在 transformers DeepSeek-V3 中共享专家吃的是 MoE 层输入（不是残差流之前的东西），变量名 `residuals` 容易误导。
9. **EP 下的 group-limited routing 是通信优化，不是精度优化**：限制每个 token 最多去 `topk_groups` 个节点，直接限制了 all-to-all 跨节点流量的上界。
10. **路由权重在 fc2 之前乘还是之后乘？** 数学上等价（fc2 无 bias 时），Megatron 在激活后乘以便融合；vLLM 在 GEMM epilogue 里乘；有 bias 的专家（如 GPT-OSS）就要小心，不能随意交换。

---

## 7. 自测题 + 动手练习

**自测题**

1. 设 `T=4096, E=64, k=6, capacity_factor=1.25`，Megatron 的 `get_capacity` 给出每专家容量是多少？megablocks（world_size=1）的 `expert_capacity` 给出多少？把 T 换成 4097 时两者还相等吗？（提示：一个是 `math.ceil`，一个是 `int` 截断）
2. DeepSeek-V3 671B 中，一个 token 最多会被发往几个专家组？如果 8 个组恰好对应 8 个节点，每个 token 最多跨几个节点？
3. 解释 transformers `grouped_mm_experts_forward` 中 `hidden_states[perm // num_top_k]` 为什么是对的。如果 `top_k_index` 是 `[T, k]` 按列优先展平的，这个公式还对吗？
4. vLLM `fused_experts_impl` 两次调用 `dispatch_fused_moe_kernel` 的 `top_k` 参数分别是多少？为什么？
5. Megatron 的 `MoEAuxLossAutoScaler.backward` 返回的 aux loss 梯度为什么是常数 `ones * scale`？它和“把 aux loss 加到总 loss 上再 backward”等价吗？
6. DeepEP 的 `do_cpu_sync=False` 时输出 buffer 有多大？专家 GEMM 如何知道哪些行是有效的？
7. 为什么 megablocks dMoE 的块稀疏路径要求 `ffn_hidden_size % 128 == 0`？

**动手练习**

1. 运行第 3 节代码，把 `capacity` 从 1 增加到 16，画出“被丢弃的 (token, expert) 对比例 vs capacity”的曲线；再把 `forward_sorted` 改成 `"probs"` 丢弃策略（每专家保留权重最大的 C 个）。
2. 在 ToyMoE 上训练一个玩具任务（例如拟合随机 MLP 的输出），分别用 (a) 无均衡、(b) aux loss 系数 0.01、(c) sigmoid + `update_bias`，记录每步 `counts` 的标准差，观察负载均衡效果。
3. 把 `forward_sorted` 中的专家 for 循环替换为 `torch._grouped_mm`（按 transformers 注释：torch≤2.8 的 `torch._grouped_mm` 只支持 Hopper，torch≥2.9 支持 SM80+；参考 transformers `_grouped_linear` 对 `offs` 与权重转置的处理），并与原版对比数值和速度。
4. 阅读 `repos/03-inference/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py` 的 docstring 例子，用纯 PyTorch 写一个 `moe_align_block_size_ref(topk_ids, block_size, num_experts)`，复现 `[3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]`（注意：例子中的专家编号是 1..4，需要先减 1 或把 num_experts 设为 5）。
5. 用 `torch.distributed`（gloo 后端、CPU、2 进程）实现一个最小 EP：每个进程持有 E/2 个专家，按第 2.7 节的流程用 `all_to_all_single` 做 dispatch/combine，并与单进程 `ToyMoE.forward_sorted` 结果对齐。
