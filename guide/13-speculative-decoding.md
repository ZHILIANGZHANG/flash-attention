# 13 · 投机解码（Speculative Decoding）源码导读

> 读完本篇你应该能回答：为什么"小模型先猜、大模型再验"能**一字不差地**保持大模型的输出分布？Medusa / EAGLE / MTP 的"猜"分别是怎么做的？树形注意力掩码长什么样？vLLM 的 rejection sampler 在 GPU 上是怎么把这件事批量化的？

本文引用的代码版本（全部来自 `repos/` 下的子模块，行号均已用 `grep -n` / `sed -n` 核对过）：

| 子模块 | 仓库 | commit |
|---|---|---|
| `repos/12-speculative-decoding/Medusa` | FasterDecoding/Medusa | `e2a5d20c048a9b0a4092e6933c34313687422518` |
| `repos/12-speculative-decoding/EAGLE` | SafeAILab/EAGLE | `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` |
| `repos/03-inference/vllm` | vllm-project/vllm | `f5fefbffe51e1cc91e0d6507d272f337473989b3` |
| `repos/01-attention/DeepSeek-V3` | deepseek-ai/DeepSeek-V3 | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` |
| `repos/13-distributed/megatron-lm` | NVIDIA/Megatron-LM | `3bb482c487cd447a2b40a3bf076ab14cad514e13` |

---

## 1. 要解决的问题 / 一句话直觉

**问题**：自回归解码每步只出 1 个 token，而每步都要把整个大模型的权重从 HBM 读一遍。batch 小的时候，GPU 算力几乎闲着，瓶颈是**显存带宽**（memory-bound）。于是"跑 1 个 token"和"跑 5 个 token"的耗时几乎一样——前向一次能并行算多个位置的 logits（就像 prefill 一样）。

**一句话直觉**：先用一个便宜的"草稿器"（draft）一口气猜出 γ 个 token，再让大模型（target）**一次前向**并行地给这 γ 个位置打分；按一个精心设计的"接受/拒绝"规则保留前缀，第一个被拒绝的位置用"修正分布"重采一个 token。只要规则对，**输出分布与直接用大模型逐 token 采样完全相同**，只是更快。

```
草稿器:   x1  x2  x3  x4         (γ=4 次便宜前向)
目标模型: p1  p2  p3  p4  p5     (1 次昂贵前向，对 5 个位置同时出分布)
验证:     ✓   ✓   ✗→重采 y3     → 本轮产出 x1 x2 y3（3 个 token）
若全部 ✓ → 额外从 p5 采一个 "bonus token"，本轮产出 γ+1 个
```

草稿器有很多种：

| 草稿来源 | 代表 | 是否需要额外模型 |
|---|---|---|
| 独立小模型 | Leviathan et al. 2023 / Chen et al. 2023（原始论文） | 是（同词表的小 LM） |
| 目标模型上加几个预测头 | **Medusa** | 只加几个 head |
| 目标模型特征 + 1 层小 Transformer | **EAGLE / EAGLE-2 / EAGLE-3** | 加 1 层 decoder |
| 模型训练时自带 | **DeepSeek-V3 MTP** | 训练时就有 |
| 从 prompt 里查 n-gram | vLLM `ngram` | 不需要 |

---

## 2. 核心原理

### 2.1 记号

- 词表大小 `V`。目标模型在位置 i 的分布 `p_i ∈ R^V`，草稿器的分布 `q_i ∈ R^V`。
- 草稿 token `x_i ~ q_i`（i = 1..γ）。
- 目标模型对拼接后的序列 `[prefix, x_1, …, x_γ]` 做**一次**前向，得到 `γ+1` 个位置的 logits，形状 `[γ+1, V]`：第 i 行是"已知 prefix 和 x_1..x_{i-1} 时下一个 token 的分布" `p_i`，最后一行 `p_{γ+1}` 用于 bonus token。

### 2.2 接受/拒绝规则（Speculative Sampling）

对 i = 1..γ 依次：

1. 采 `r ~ U(0,1)`；
2. 若 `r < min(1, p_i(x_i) / q_i(x_i))`：**接受** `x_i`，继续下一个；
3. 否则**拒绝**：从修正分布 `p'_i = normalize( max(p_i − q_i, 0) )` 采一个 token 作为本位置的输出，**结束本轮**。

若 γ 个全部接受，再从 `p_{γ+1}` 采一个 bonus token。

### 2.3 证明（为什么输出分布严格等于 p）

只看一个位置（后续位置在"前面都被接受"的条件下同理，用归纳法即可）。令 β = Σ_x min(p(x), q(x))（接受概率）。

输出为 x 有两条路：

- 草稿采到 x 且被接受：`q(x) · min(1, p(x)/q(x)) = min(p(x), q(x))`
- 草稿被拒（总概率 `1 − β`），再从 p' 采到 x：
  `(1 − β) · max(p(x) − q(x), 0) / Σ_y max(p(y) − q(y), 0)`

关键恒等式：`Σ_y max(p(y) − q(y), 0) = Σ_y (p(y) − min(p(y),q(y))) = 1 − β`。所以第二项就是 `max(p(x) − q(x), 0)`。

两项相加：`min(p, q) + max(p − q, 0) = p(x)`。∎

几个推论，后面读代码时会反复用到：

- **草稿是确定性的（q 是 one-hot δ_x）**：接受概率 = `p(x)`；修正分布 = "把 x 从 p 里抹掉再归一化"。EAGLE 的采样验证和 vLLM 的 ngram 路径都是这个特例。
- **贪心解码（temperature=0）**：p 是 one-hot（argmax），规则退化成"草稿 token == target argmax 才接受，否则输出 target argmax"。
- 如果有多个候选（树），对**互不相同**的兄弟候选反复做"point-mass 拒绝采样 + 抹掉被拒 token"，依然是精确的（SpecInfer 的多候选验证思路），EAGLE 的 `evaluate_posterior` 就是这么做的。

### 2.4 期望加速比

设每个位置独立地以概率 α 被接受（α = E[Σ min(p,q)]），一轮产出的 token 数是截断几何分布：

```
E[#tokens / 轮] = 1 + α + α² + … + α^γ = (1 − α^{γ+1}) / (1 − α)
```

设草稿器一次前向的耗时是目标模型的 c 倍，则一轮耗时 ≈ `γ·c + 1`（单位：一次目标前向），墙钟加速比：

```
speedup ≈ (1 − α^{γ+1}) / ((1 − α)(γ·c + 1))
```

例：α=0.8，γ=4，c=0.05 → E[tokens]=3.36，speedup≈2.8×。α 越高、草稿越便宜越好；γ 太大时 α^k 衰减，收益递减而草稿开销线性增长。树形草稿（Medusa/EAGLE）的思路是：与其把预算花在一条长链上，不如在每一层多放几个候选，提高"至少有一条路被接受"的概率。

### 2.5 树形注意力（tree attention）

把多条候选路径合并成一棵前缀树，展平成一个长度 `T` 的序列一次喂给目标模型。为了让每个节点只"看见"自己的祖先（而不是展平后排在前面的兄弟节点），需要一个 `[T, T]` 的掩码：

```
树:        root
          /    \
        a1      a2
       /  \      \
     b1    b2     b3

展平顺序: [root, a1, a2, b1, b2, b3]
mask[i][j] = 1 iff j 是 i 的祖先或 i 自身
        root a1 a2 b1 b2 b3
root  [  1   0  0  0  0  0 ]
a1    [  1   1  0  0  0  0 ]
a2    [  1   0  1  0  0  0 ]
b1    [  1   1  0  1  0  0 ]
b2    [  1   1  0  0  1  0 ]
b3    [  1   0  1  0  0  1 ]
position_id = 深度 = [0,1,1,2,2,2]（不是 0..5！）
```

这个 `[T,T]` 块拼在已有 KV cache 的 causal mask 右下角；KV cache 部分全部可见。验证之后，只有被接受那条路径的 KV 需要保留，其余要从 cache 里"挪走"。

---

## 3. 最小实现（可直接运行的 PyTorch 代码）

下面的代码用两个"玩具模型"（下一个 token 的分布只依赖上一个 token，用随机表表示）验证：投机采样产生的前两个 token 的联合分布与目标模型精确一致。我用等价的 NumPy 版本跑过 20 万次，最大绝对误差 ~7e-4（纯采样噪声），每轮平均产出 ~1.9 个 token。

```python
import torch

torch.manual_seed(0)
V, GAMMA = 5, 3
P_TABLE = torch.distributions.Dirichlet(torch.ones(V)).sample((V,))        # target: [V, V]
Q_TABLE = torch.distributions.Dirichlet(0.5 * torch.ones(V)).sample((V,))  # draft:  [V, V]

def target_probs(prefix):  # 真实场景: 一次前向返回 [len, V]，这里逐位置模拟
    return P_TABLE[prefix[-1]]

def draft_probs(prefix):
    return Q_TABLE[prefix[-1]]

def speculative_step(prefix, gamma=GAMMA):
    # 1) 草稿器自回归猜 gamma 个 token，并记下每步的 q 分布
    ctx, draft_tokens, qs = list(prefix), [], []
    for _ in range(gamma):
        q = draft_probs(ctx)
        x = torch.multinomial(q, 1).item()
        draft_tokens.append(x); qs.append(q); ctx.append(x)
    # 2) 目标模型"一次前向"得到 gamma+1 个位置的分布 p_1..p_{gamma+1}
    ps = [target_probs(list(prefix) + draft_tokens[:i]) for i in range(gamma + 1)]
    # 3) 逐位置接受/拒绝
    out = []
    for i, x in enumerate(draft_tokens):
        r = torch.rand(())
        if r < torch.clamp(ps[i][x] / qs[i][x], max=1.0):
            out.append(x)                                   # 接受
        else:
            residual = torch.clamp(ps[i] - qs[i], min=0.0)  # max(p - q, 0)
            residual = residual / residual.sum()
            out.append(torch.multinomial(residual, 1).item())  # 修正分布重采
            return out                                       # 第一个拒绝即结束
    out.append(torch.multinomial(ps[gamma], 1).item())       # 全部接受 -> bonus token
    return out

# 经验验证：前两个生成 token 的联合分布应等于 P[0, a] * P[a, b]
N, counts, total = 50_000, torch.zeros(V, V), 0
for _ in range(N):
    o = speculative_step([0]); total += len(o)
    if len(o) < 2:
        o = o + speculative_step([0] + o)
    counts[o[0], o[1]] += 1
exact = P_TABLE[0][:, None] * P_TABLE
print("max |emp - exact| =", (counts / N - exact).abs().max().item())
print("avg tokens per target forward =", total / N)
print("per-context acceptance alpha =", torch.minimum(P_TABLE, Q_TABLE).sum(-1))
```

要点：

- `torch.clamp(ps[i][x] / qs[i][x], max=1.0)` 就是 `min(1, p/q)`；
- `residual` 那两行就是修正分布 `normalize(max(p − q, 0))`；
- 第一个拒绝就 `return`：拒绝之后的草稿 token 是在**错误前缀**上生成的，作废；
- 每轮至少产出 1 个 token（最坏情况等于普通解码），最多 γ+1 个。

---

## 4. 源码导读（按阅读顺序）

建议的阅读顺序：**Medusa（最直观的树 + 掩码）→ EAGLE（特征级草稿 + 动态树 + 精确采样验证）→ vLLM（工业级批量化 rejection sampler）→ MTP**。

### 4.1 Medusa：在目标模型上长出几个"预测未来"的头

#### (1) `ResBlock` 与 `medusa_head`

文件：`medusa/model/medusa_model.py`，`ResBlock` L43–L72，`MedusaModelABC.__init__` 中构造 head 的部分 L111–L119
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/medusa_model.py#L43-L119

```python
class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)   # 初始化为恒等映射
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))
...
self.medusa_head = nn.ModuleList(
    [
        nn.Sequential(
            *([ResBlock(self.hidden_size)] * medusa_num_layers),
            nn.Linear(self.hidden_size, self.vocab_size, bias=False),
        )
        for _ in range(medusa_num_heads)
    ]
)
```

逐行解释：

- 第 k 个 head 输入的是目标模型最后一层隐藏状态 `h_t`（形状 `[B, L, D]`），预测的是 **t+k+1** 位置的 token（原 LM head 预测 t+1）。所以 Medusa 是"一次性并行猜未来 K 个位置"，各 head 之间**没有依赖**——这是它简单的地方，也是它接受率不如 EAGLE 的根本原因（head k 不知道 head k−1 猜了什么）。
- `zeros_(self.linear.weight)` 使 ResBlock 初始时变成 `x + SiLU(b)`：权重项为 0，只剩一个与输入无关的常数偏移（bias 仍是 `nn.Linear` 的默认初始化，数值很小），即"近似恒等"。所以每个 head 起点约等于"把原特征直接送进一个新的 LM head"，训练更稳。
- 注意 `[ResBlock(...)] * medusa_num_layers` 是把**同一个对象**重复 n 次（Python 列表乘法），多层时这些层共享参数。默认 `medusa_num_layers=1` 时无影响。

#### (2) `forward(..., medusa_forward=True)`：一次前向得到原 logits + K 组 Medusa logits

文件：`medusa_model.py` L170–L221
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/medusa_model.py#L170-L221

```python
outputs = self.base_model.model(input_ids=..., past_key_values=..., position_ids=...)
if output_orig:
    orig = self.base_model.lm_head(outputs[0])          # [B, L, V]
hidden_states = outputs[0].clone()
medusa_logits = []
for i in range(self.medusa):
    medusa_logits.append(self.medusa_head[i](hidden_states))
...
return torch.stack(medusa_logits, dim=0), outputs, orig  # [K, B, L, V]
```

`orig` 是原模型的 next-token logits，**既用来验证上一轮的候选，也提供本轮候选的第 1 个 token**；`medusa_logits[k]` 提供第 k+2 个 token 的候选。

#### (3) 主循环 `medusa_generate`

文件：`medusa_model.py` L235–L371（循环体 L314–L371）
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/medusa_model.py#L307-L371

```python
medusa_logits, logits = initialize_medusa(input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values)
for idx in range(max_steps):
    candidates, tree_candidates = generate_candidates(medusa_logits, logits, tree_indices, retrieve_indices, ...)
    medusa_logits, logits, outputs = tree_decoding(self, tree_candidates, past_key_values, medusa_position_ids, input_ids, retrieve_indices)
    best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, ...)
    input_ids, logits, medusa_logits, new_token = update_inference_inputs(...)
```

四步，正好对应"出候选 → 树形验证 → 选最长接受前缀 → 更新 KV cache"。注意 `assert input_ids.shape[0] == 1`：参考实现只支持 batch=1。

#### (4) `generate_medusa_buffers`：把树结构编译成 4 个张量

文件：`medusa/model/utils.py` L32–L125
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py#L32-L125

树用 `medusa_choices` 描述（`medusa/model/medusa_choices.py` 第 1 行 `mc_sim_7b_63`）：每个元素是一条路径，如 `[0, 1]` 表示"head0 的 top-1，再接 head1 的 top-2"。共 63 个节点 + 1 个根 = 64。

```python
sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
medusa_len = len(sorted_medusa_choices) + 1           # +1 是根（原 LM head 的 token）
...
medusa_attn_mask = torch.eye(medusa_len, medusa_len)  # L59: 自己看自己
medusa_attn_mask[:, 0] = 1                             # 所有节点都看根
...   # 对每个节点，把它所有祖先的列置 1
    medusa_attn_mask[j + start + 1, ancestor_idx] = 1
```

- **`medusa_attn_mask`** `[T,T]`：就是 2.5 节画的那个"只看祖先"的掩码。
- **`tree_indices`**（L81）：`cur_medusa_choice[-1] + TOPK * i + 1`。候选池是一个扁平向量 `candidates = [root_token, head0 的 top10, head1 的 top10, ...]`（长度 `1 + K·TOPK`），节点 `[.., c]` 在第 i 层取第 c 名，所以下标是 `1 + i·TOPK + c`。
- **`medusa_position_ids`**（L88）：每个节点的 position = 深度（根 0，第一层 1……），这样 RoPE 位置对得上。
- **`retrieve_indices`**（L93–L108）：形状 `[num_leaves, max_depth+1]`，每行是一条从根到叶子的路径在展平序列中的下标（不足补 -1，再整体 +1 后前面拼一列 0 作为根）。验证时用它把树形输出"取回"成若干条线性候选。

#### (5) 掩码如何注入到 attention

文件：`medusa/model/modeling_llama_kv.py` L815–L821
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/modeling_llama_kv.py#L815-L821

```python
# [MODIFIED] add medusa mask
if hasattr(self, "medusa_mask") and self.medusa_mask is not None:
    medusa_mask = self.medusa_mask
    medusa_len = medusa_mask.size(-1)
    combined_attention_mask[:, :, -medusa_len:, -medusa_len:][
        medusa_mask == 0
    ] = combined_attention_mask.min()
```

`combined_attention_mask` 是 HF 风格的加性掩码 `[B, 1, q_len, kv_len]`（可见 = 0，不可见 = 极小值）。只改右下角 `[T, T]` 块：树里不是祖先的位置填成最小值。KV cache 部分（左边）保持全可见。`initialize_medusa`（utils.py L128–L150）在 prefill **之后**才把 `medusa_mask` 挂上去，所以 prefill 仍是普通 causal。

#### (6) `generate_candidates` / `tree_decoding`

文件：`utils.py` L258–L307、L309–L348
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py#L258-L348

```python
candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)               # 根：原 LM head（贪心/fast 时）
candidates_medusa_logits = torch.topk(medusa_logits[:, 0, -1], TOPK, dim=-1).indices   # [K, TOPK]
candidates = torch.cat([candidates_logit, candidates_medusa_logits.view(-1)], dim=-1)   # [1+K*TOPK]
tree_candidates = candidates[tree_indices]                                 # [T] 展平的树
tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), ...)], dim=0)
cart_candidates = tree_candidates_ext[retrieve_indices]                    # [num_leaves, depth+1]
```

`retrieve_indices` 中的 -1 会索引到 `tree_candidates_ext` 末尾补的那个 0（Python 负下标），这是一个小技巧：填充位置随便给个 token，反正后面 `cumprod` 会截断。

`tree_decoding` 把 `tree_candidates`（`[1, T]`）作为输入、`position_ids = medusa_position_ids + input_ids.shape[1]`（L333）一次前向，然后 `logits = tree_logits[0, retrieve_indices]`（L346）把输出 `[T, V]` 重排成 `[num_leaves, depth+1, V]`，与 `cart_candidates` 一一对齐。

#### (7) `evaluate_posterior`：贪心精确匹配 vs. typical acceptance

文件：`utils.py` L436–L529
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py#L436-L529

贪心分支（L459–L472）：

```python
posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()   # [leaves, depth]
candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)        # 最长全 1 前缀
accept_length = candidates_accept_length.max()
best_candidate = torch.argmax(candidates_accept_length)
```

- `logits[:, i]` 是"看到候选第 0..i 个 token 后"的分布，所以它验证的是 `candidates[:, i+1]`——这就是 `[:, 1:]` 和 `[:, :-1]` 错一位的原因。
- `cumprod` 把 `[1,1,0,1]` 变成 `[1,1,0,0]`，`sum` 得到最长接受前缀长度 2。这是"按行找第一个 0"的向量化写法，后面 EAGLE 也一模一样。

**typical acceptance**（fast 分支 L474–L498）：

```python
posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
candidates_prob = torch.gather(posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)).squeeze(-1)
posterior_entropy = -torch.sum(posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1)
threshold = torch.minimum(torch.ones_like(posterior_entropy) * posterior_threshold,
                          torch.exp(-posterior_entropy) * posterior_alpha)
posterior_mask = candidates_prob > threshold
```

规则：候选 token 在目标分布下的概率 `p(x)` 只要超过 `min(ε, δ·exp(−H(p)))` 就接受。熵 H 高（目标模型自己也不确定）时阈值低，更宽松。**注意：这条规则不是 2.2 节的 rejection sampling，不保证输出分布等于 p**——它换来了更高的接受率，牺牲了严格无损。Medusa 论文的主张是"typical 的输出质量与原模型相当"，这是经验结论，不是数学保证。若接受长度相同，多条候选按 `log p` 之和选 likelihood 最大的。

#### (8) `update_inference_inputs`：把被接受路径的 KV 挪到连续位置

文件：`utils.py` L531–L593
链接：https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py#L531-L593

```python
select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
input_ids = torch.cat([input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1)
tgt = past_key_values_data[..., select_indices, :]
dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
dst.copy_(tgt, non_blocking=True)
current_length_data.fill_(prev_input_len + tgt.shape[-2])
```

树里的 T 个节点在 KV cache 中占了 `[prev_len, prev_len+T)`，但被接受的只是其中一条路径上的 `accept_length+1` 个节点（下标不连续）。这几行把它们 gather 出来再写回到 `[prev_len, prev_len+accept_length+1)`，然后把"当前长度"指针设到新末尾——其余节点的 KV 下一轮直接被覆盖，相当于"丢弃"。

---

### 4.2 EAGLE：在特征层面做自回归草稿

EAGLE 的核心观察：**在 hidden state（特征）层面做自回归比在 token 层面容易**；而且草稿器要知道"上一步实际采样的 token"，否则特征的不确定性无法消除。所以 EAGLE 的草稿器输入是 `(目标模型的特征 f_t, 下一个 token 的 embedding e_{t+1})`，输出预测的 `f_{t+1}`，再过目标模型（或自己的）LM head 得到 token 分布。草稿器是**一层** decoder layer，比 Medusa head 重一点，但是自回归的，接受率高很多。

本仓库同时包含 EAGLE-1/2（`cnets1.py`）与 EAGLE-3（`cnets.py`），由 `EaModel.__init__` 中的 `use_eagle3` 选择：

文件：`eagle/model/ea_model.py` L20–L21、L55–L60
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py#L55-L60

```python
from .cnets import Model
from .cnets1 import Model as Model1
...
if use_eagle3:
    self.ea_layer = Model(config, ..., total_tokens=total_token, depth=depth, top_k=top_k, ...)
else:
    self.ea_layer = Model1(config, ..., total_tokens=total_token, depth=depth, top_k=top_k, ...)
```

#### (1) EAGLE-1/2 的输入融合：`fc(cat(embedding, feature))`

文件：`eagle/model/cnets1.py` L524、L623
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/cnets1.py#L623

```python
self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)        # L524
hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))       # L623
```

形状：`inputs_embeds [B, L, D]`（token 往后错一位的 embedding）与 `hidden_states [B, L, D]`（目标模型最后一层特征）拼成 `[B, L, 2D]`，投影回 `D`，然后过 decoder layer。

#### (2) EAGLE-3：融合低/中/高三层特征

目标模型侧：`eagle/model/modeling_llama_kv.py` L1137–L1139
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/modeling_llama_kv.py#L1137-L1139

```python
for idx, decoder_layer in enumerate(self.layers):
    if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:
        all_hidden_states += (hidden_states,)
```

只收集第 2 层、中间层、倒数第 3 层的输入隐藏状态。`utils.py` L252 / L328 中 `torch.cat(outputs["hidden_states"], dim=-1)` 把它们拼成 `[B, L, 3D]`。

草稿器侧：`eagle/model/cnets.py` L478–L545（`Model.__init__`）
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/cnets.py#L478-L545

```python
self.lm_head=nn.Linear(config.hidden_size,config.draft_vocab_size,bias=False)    # L487
...
self.midlayer = LlamaDecoderLayeremb(config)                                      # L530
if hasattr(config, "target_hidden_size"):
    self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)   # L532
...
d2t=torch.zeros((config.draft_vocab_size),dtype=torch.long)                       # L538
```

- `fc: 3D → D` 把三层特征压成一个；`forward` 中 `if hidden_states.shape[-1]!=inputs_embeds.shape[-1]: hidden_states = self.fc(hidden_states)`（L639–L640）——第一步输入是 3D 的目标特征要投影，后续步骤输入的是草稿器自己输出的 D 维特征，不需要投影。
- `draft_vocab_size` + `d2t`：EAGLE-3 的草稿 LM head 只在一个**缩小的词表**上出分布（高频 token），`d2t[i]` 是"草稿词表下标 → 目标词表下标"的偏移，`topK_genrate` 里 `topk_index + self.d2t[topk_index]` 就是做这个映射。
- `LlamaDecoderLayeremb`（L387–L461）的特别之处：`torch.cat((input_emb, hidden_states), dim=-1)`（L430）——embedding 和特征**分别 norm 之后拼接**作为 attention 的输入（所以这层 attention 的 q/k/v 投影输入维度是 2D），残差仍走特征那一路。

#### (3) `topK_genrate`：EAGLE-2 的动态草稿树

文件：`cnets.py` L670–L827
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/cnets.py#L670-L827

Medusa 的树是**静态**的（`medusa_choices` 写死）。EAGLE-2 发现草稿器的置信度（累计 log 概率）和真实接受率高度相关，所以每轮**按置信度动态长树**：

```python
last_p = self.logsoftmax(last_headout)
top = torch.topk(last_p, top_k, dim=-1)                   # 第一层: top_k 个孩子
scores = topk_p[0]
...
for i in range(depth):                                    # L719: 逐层扩展
    out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                       position_ids=position_ids, use_cache=True)
    last_p = self.logsoftmax(self.lm_head(self.norm(out_hidden[0])))   # [top_k, V']
    top = torch.topk(last_p, top_k, dim=-1)                            # 每个当前节点再取 top_k
    cu_scores = topk_p + scores[:, None]                               # L740: 累计 log 概率 [top_k, top_k]
    topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)            # L742: 全局只保留 top_k 个继续扩展
    scores = topk_cs_p
    out_ids = topk_cs_index // top_k                                   # 每个保留节点的父亲
    input_hidden = out_hidden[:, out_ids]
    ...
    tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
```

- **扩展（expand）阶段**：每层有 `top_k` 个"活跃"节点，各自展开 `top_k` 个孩子得到 `top_k²` 个候选，按**累计 log 概率**（即路径概率的对数）全局选 `top_k` 个继续。每层只需一次草稿前向（这 top_k 个节点作为一个 batch 的序列位置，用 `self.tree_mask` 控制可见性）。
- **重排（rerank）阶段**（L760–L770）：

```python
scores_list = torch.cat(scores_list, dim=0).view(-1)
top_scores = torch.topk(scores_list, total_tokens, dim=-1)   # L762: 所有层的所有候选中选 total_tokens 个
top_scores_index = torch.sort(top_scores_index).values
draft_tokens = ss_token_list[top_scores_index]
```

  从所有展开过的节点里选累计分数最高的 `total_tokens` 个组成最终草稿树（构造函数默认 `total_tokens=63`，L522 `self.total_tokens = total_tokens - 1` 减掉的 1 是根节点，即上一轮已采样的 token）。因为子节点分数 ≤ 父节点分数（log 概率累加只会变小），选出来的集合（不考虑分数并列的边界情况）天然是一棵**连通的树**。
- **建掩码**（L776–L782）：

```python
tree_mask = torch.eye(total_tokens + 1).bool()
tree_mask[:, 0] = True
for i in range(total_tokens):
    tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])   # 自己的行 |= 父亲的行 => 继承所有祖先
tree_position_ids = torch.sum(tree_mask, dim=1) - 1        # 祖先个数 = 深度
```

  非常优雅：因为节点按下标排序后父亲总在儿子前面，"我的可见集 = 我自己 ∪ 父亲的可见集"一遍扫描就得到完整的祖先掩码；深度直接等于掩码行和减 1。
- 之后生成 `retrieve_indices`（L796–L809，每个叶子回溯到根）——和 Medusa 同一个概念。

#### (4) `evaluate_posterior`：贪心与"精确"的采样验证

文件：`eagle/model/utils.py` L337–L415
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py#L337-L415

贪心分支（L360–L372）与 Medusa 完全相同（`cumprod` 找最长前缀）。采样分支是本篇最值得细读的一段：

```python
accept_length = 1
accept_cand = candidates[0][:1]
best_candidate = 0
for i in range(1, candidates.shape[1]):              # 逐深度
    if i != accept_length: break                     # 上一层没接受任何孩子 -> 停
    adjustflag = False
    is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)   # 与已接受前缀一致的所有路径
    fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
    gt_logits = logits[fi, i - 1][None]
    gt_logits = logits_processor(None, gt_logits)[0]                   # 温度/top-p/top-k
    gtp = torch.softmax(gt_logits, dim=0)                              # 目标分布 p
    candidates_set = []
    for j in range(candidates.shape[0]):
        if is_eq[j]:
            x = candidates[j, i]; xi = x.item()
            if xi in candidates_set or xi == -1: continue              # 兄弟去重
            candidates_set.append(xi)
            r = random.random()
            px = gtp[xi]
            qx = 1.0                                                   # L398: 草稿视为确定性 (point mass)
            acp = px / qx
            if r <= acp:                                               # 以 p(x) 概率接受
                accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                accept_length += 1; best_candidate = j
                break
            else:
                gtp[xi] = 0                                            # L406: 抹掉被拒的 x
                gtp = gtp / gtp.sum()                                  # 归一化 = point-mass 的修正分布
                adjustflag = True
if adjustflag and accept_length != candidates.shape[1]:
    sample_p = gtp                                                     # 用修正分布采下一个 token
else:
    sample_p = torch.softmax(logits_processor(None, logits[best_candidate, accept_length - 1][None])[0], dim=0)
```

把它和 2.3 节对照：

- 草稿树的孩子是 top-k 选出来的（确定性），所以取 `q = δ_x`，接受概率 `min(1, p(x)/1) = p(x)`。
- 被拒后修正分布 `normalize(max(p − δ_x, 0))` 就是"把 x 的概率置 0 再归一化"——正是 `gtp[xi] = 0; gtp /= gtp.sum()`。
- 然后在**修正后的分布**上继续尝试下一个（不同的）兄弟。这相当于连续多次 rejection sampling，每次的"目标"是上一次的修正分布，所以最终无论在哪个兄弟处接受、或全部被拒后从 `gtp` 采样，边缘分布都恰好是原始 p。这就是 EAGLE 说"lossless"的依据。
- 全部被拒时 `sample_p = gtp`；全部接受（或某层接受后无更深节点）时从被接受节点处的 p 采样，相当于 bonus token。返回 `accept_length - 1` 是因为第 0 个元素是根（上一轮已采样的 token）。

#### (5) 主循环与"下一轮草稿复用目标特征"

文件：`ea_model.py` L199–L304，`utils.py` L419–L479
链接：https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py#L419-L479

```python
retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
...
draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
    accept_hidden_state_new, input_ids=torch.cat((input_ids, token...), dim=1), ...)
```

验证时目标模型已经为树上所有节点算出了特征；被接受路径上那 `accept_length+1` 个节点的特征直接作为下一轮草稿的输入（草稿器自己的 KV cache `stable_kv` 增量追加），**不需要额外跑目标模型**。这是 EAGLE 系列每轮只需"1 次目标前向 + depth 次草稿前向"的原因。

---

### 4.3 vLLM：批量化的 Rejection Sampler

vLLM v1 的投机解码分两半：**proposer**（`vllm/v1/spec_decode/`，负责出草稿）和 **rejection sampler**（`vllm/v1/sample/rejection_sampler.py`，负责验证）。和上面两个研究代码最大的不同：

1. 支持 batch，每个请求的草稿长度可以不同（被 flatten 成一维，用 `cu_num_draft_tokens` 前缀和定位）；
2. 草稿是**链**而不是树（每个请求 γ 个线性 token）；
3. 严格按 2.2 节的算法实现，用 Triton kernel 一个请求一个 program。

#### (1) 选择 proposer

文件：`vllm/v1/worker/gpu_model_runner.py` L643–L651、L663–L665；`vllm/config/speculative.py` L1944–L1948
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/worker/gpu_model_runner.py#L643-L665

```python
elif self.speculative_config.use_eagle():
    self.drafter = EagleProposer(self.vllm_config, self.device, self)
...
elif self.speculative_config.method == "medusa":
    self.drafter = MedusaProposer(vllm_config=self.vllm_config, device=self.device)
...
self.rejection_sampler = RejectionSampler(self.sampler, self.speculative_config, self.device)
```

`use_eagle()` 的定义：`return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")`——**MTP 在 vLLM 里复用 EAGLE 的 proposer 框架**（都是"用目标模型 hidden state 作输入的草稿模块"）。

#### (2) `SpecDecodeMetadata` 与 `_calc_spec_decode_metadata`：flatten 的下标体操

文件：`vllm/v1/spec_decode/metadata.py` L9–L27；`gpu_model_runner.py` L2625–L2698
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/worker/gpu_model_runner.py#L2625-L2698

注释里给的例子非常好懂（5 个请求，草稿数 `[3,0,2,0,1]`）：

```
# cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
# num_draft_tokens:         [  3,   0,   2,   0,   1]
# cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
# logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106, 206, 207, 208]
# target_logits_indices:    [  0,   1,   2,   5,   6,   9]
# bonus_logits_indices:     [  3,   4,   7,   8,  10]
```

- 请求 0 本轮输入是"上一个 token + 3 个草稿" = 4 个位置（0..3），它们的 logits 都要算：前 3 个验证草稿（target），第 4 个产生 bonus。
- 请求 1 是 prefill（100 个 token），只要最后一个位置（103）的 logits，它只有 bonus、没有 target。
- `logits_indices` 告诉模型 "只在这 11 个位置上算 LM head"（省掉大部分位置的 `[*, V]` 矩阵乘）；`target_logits_indices`/`bonus_logits_indices` 是在这 11 行里的下标。
- `draft_token_ids = self.input_ids.gpu[logits_indices][target_logits_indices + 1]`：位置 i 的 logits 验证的是位置 i+1 的输入 token，又是"错一位"。

#### (3) `RejectionSampler.forward`

文件：`vllm/v1/sample/rejection_sampler.py` L45–L216
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/sample/rejection_sampler.py#L45-L216

类注释定义了三个术语：**accepted**（通过检验的草稿）、**recovered**（被拒位置从修正分布重采的 token）、**bonus**（全部接受时额外采的 token）。`output = accepted + recovered + bonus`。

```python
bonus_logits = logits[bonus_logits_indices]                       # [batch, V]
bonus_sampler_output = self.sampler(logits=bonus_logits, ..., predict_bonus_token=True, ...)
bonus_token_ids = bonus_sampler_output.sampled_token_ids          # L162: bonus 用普通 sampler 采（支持 top-k/top-p）
raw_target_logits = logits[target_logits_indices].to(torch.float32)   # [num_tokens, V]
target_logits = self.apply_logits_processors(target_logits, ...)
target_logits = apply_sampling_constraints(target_logits, metadata.cu_num_draft_tokens, sampling_metadata)
output_token_ids = rejection_sample(metadata.draft_token_ids, metadata.num_draft_tokens, metadata.max_spec_len,
                                    metadata.cu_num_draft_tokens, draft_probs, target_logits, bonus_token_ids, ...)
```

- bonus token 提前用普通 sampler 采好再传进去，这样 bonus 能享受所有采样选项。
- `apply_sampling_constraints`（L522 起）对 target logits 做温度/top-k/top-p，得到真正要匹配的 p；注意草稿侧 `draft_probs` 也必须是**同样处理后的**分布，否则 `p/q` 比较的不是同一个东西。
- 精度：`.to(torch.float32)`，softmax 也在 fp32 做（L487）。

#### (4) `rejection_sample`：先贪心 kernel，再随机 kernel

文件：`rejection_sampler.py` L410–L519
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/sample/rejection_sampler.py#L410-L519

```python
output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
...
if not sampling_metadata.all_random:
    target_argmax = target_logits.argmax(dim=-1)
    _rejection_greedy_sample(output_token_ids, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids, is_greedy, ...)
    if sampling_metadata.all_greedy:
        return output_token_ids
target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
recovered_token_ids = sample_recovered_tokens(max_spec_len, num_draft_tokens, cu_num_draft_tokens,
                                              draft_token_ids, draft_probs, target_probs, sampling_metadata, device, ...)
_rejection_random_sample(output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs,
                         bonus_token_ids, recovered_token_ids, uniform_probs, is_greedy, max_spec_len, ...)
```

- 输出是 `[batch, max_spec_len+1]`，没用到的位置是 `PLACEHOLDER_TOKEN_ID = -1`（L38），由 `parse_output` 过滤。
- 一个 batch 里贪心请求和随机请求可以混合，两个 kernel 各自用 `is_greedy` 早退，互不干扰。
- recovered token **提前对所有位置并行采好**，随机 kernel 里只在第一个拒绝处取用——把"串行的接受判断"和"昂贵的 V 维采样"拆开，前者只是 γ 次标量比较。

#### (5) 贪心 kernel

文件：`rejection_sampler.py` L726–L780
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/sample/rejection_sampler.py#L726-L780

```python
req_idx = tl.program_id(0)                                  # 一个 program 处理一个请求
start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
rejected = False
for pos in range(num_draft_tokens):
    if not rejected:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
        ...
            token_id = target_argmax_id                     # 无论是否接受，输出都是 argmax
            rejected = draft_token_id != target_argmax_id
        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
if not rejected:
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)
```

贪心时 p 是 one-hot：接受就是"相等"，被拒时的 recovered token 就是 argmax 本身，所以干脆"一律写 argmax"，再用 `rejected` 标志截断。

#### (6) 随机 kernel：`p/q >= u`

文件：`rejection_sampler.py` L829–L900
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/sample/rejection_sampler.py#L829-L900

```python
if draft_token_id < 0:
    accepted = False                       # -1 是 padding 草稿，必拒
...
    if NO_DRAFT_PROBS:
        draft_prob = 1                     # ngram 等没有分布的草稿：当作 point mass
    else:
        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
    target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
    accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob    # L884
if accepted:
    token_id = draft_token_id
else:
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

L884 就是 2.2 节的规则（`p/q >= u` 等价于 `u <= min(1, p/q)`）。`uniform_probs` 在 `generate_uniform_probs`（L622–L675）里用 **float64** 生成，注释解释：fp32 的 `torch.rand` 有不可忽略的概率恰好取到 0.0，而 `p/q >= 0` 永远成立会造成偏差。每个请求可以有自己的 `torch.Generator`（seed 可复现）。

#### (7) recovered token：用指数分布做"免归一化"采样

文件：`rejection_sampler.py` L678–L722（host 侧）、L1004–L1084（kernel）
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/sample/rejection_sampler.py#L1004-L1084

host 侧：

```python
# NOTE(woosuk): Create only one distribution for each request.
q = torch.empty((batch_size, vocab_size), dtype=q_dtype, device=device)
q.exponential_()                     # E_v ~ Exp(1)
inv_q = q.reciprocal()               # 1/E_v
```

kernel 侧：

```python
if NO_DRAFT_PROBS:
    prob = tl.load(target_probs_ptr + ..., mask=(vocab_mask & (vocab_offset != draft_token_id)), other=0.0)
else:
    prob = tl.maximum(target_prob - draft_prob, 0.0)      # L1061: max(p - q, 0)，不归一化
...
score = prob * inv_q                                       # L1075: p'_v / E_v
local_max, local_id = tl.max(score, axis=0, return_indices=True)
```

两个技巧：

1. **Exponential-race（Gumbel-max 的等价形式）**：若 `E_v ~ Exp(1)` 独立，则 `argmax_v w_v / E_v` 服从 `Categorical(w / Σw)`。证明：`E_v / w_v ~ Exp(rate=w_v)`，一组独立指数变量中最小者是第 k 个的概率为 `w_k / Σw`。所以**不需要归一化** `max(p−q,0)`，也不需要前缀和，按 BLOCK 扫一遍词表求 argmax 即可（注释 "We don't need `prob = prob / tl.sum(prob)`"）。
2. **NO_DRAFT_PROBS 分支**：q 是 one-hot δ_x 时，`max(p − δ_x, 0)` 就是"p 中把 x 置 0"，用 `vocab_offset != draft_token_id` 的 mask 实现——与 EAGLE 的 `gtp[xi] = 0` 是同一回事。
3. "每个请求只造一份噪声 `q[req]`、所有位置共用"为什么没问题？因为一个请求最多只会用到**一个** recovered token（第一个被拒位置），其余位置的 recovered token 全被丢弃，所以共享噪声不会引入相关性。

#### (8) proposer 侧：EAGLE / Medusa / MTP

`EagleProposer`（`vllm/v1/spec_decode/eagle.py` L10–L22）只是把 `pass_hidden_states_to_model=True` 传给 `SpecDecodeBaseProposer`。核心在 `llm_base_proposer.py` 的 `propose`（L512–L774）：

链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/spec_decode/llm_base_proposer.py#L512-L774

```python
if self.method in ("eagle3", "dflash"):
    target_hidden_states = self.model.combine_hidden_states(target_hidden_states)   # L550: EAGLE-3 多层特征融合
...
ret_hidden_states = self.model(**model_kwargs)          # 第一次草稿前向（对本轮所有新 token）
sample_hidden_states = last_hidden_states[token_indices_to_sample]
draft_token_ids, draft_probs = self._sample_draft_tokens(sample_hidden_states, sampling_metadata)
for token_index in range(self.num_speculative_tokens - 1):   # L689: 链式自回归，每步 1 个 token/请求
    input_ids = draft_token_ids_list[-1].int()
    ...
    self.hidden_states[:batch_size] = hidden_states          # 把上一步草稿器输出的特征喂回去（EAGLE 的特征自回归）
    ret_hidden_states = self.model(**model_kwargs)
    draft_token_ids, draft_probs = self._sample_draft_tokens(last_hidden_states[:batch_size], sampling_metadata)
draft_token_ids = torch.stack(draft_token_ids_list, dim=1)   # L771: [batch, num_speculative_tokens]
```

- 草稿默认是**贪心**：`_greedy_sample`（L438–L448）`return self.model.compute_logits(hidden_states).argmax(dim=-1)`，`draft_probs=None`，于是 rejection sampler 走 `NO_DRAFT_PROBS` 的 point-mass 分支——依然无损。开启概率草稿时 `_sample_from_logits` 返回 `(token, probs)`，走一般分支。
- 与 EAGLE 原始实现相比：vLLM 用**链**（`num_speculative_tokens` 个），不做动态树，换取对 batch、CUDA graph、paged KV 的友好。

`MedusaProposer.propose`（`medusa.py` L38–L56）就更简单了：

```python
blocks = self.model(target_hidden_states)
logits = self.model.compute_logits(blocks)
draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)   # [batch, num_heads]
```

每个 head 取 argmax 组成一条链——没有树、没有 typical acceptance，验证仍走无损的 rejection sampler。

---

### 4.4 MTP（DeepSeek-V3 Multi-Token Prediction）

**DeepSeek-V3 仓库本身**只在 `README_WEIGHTS.md` 中描述了 MTP 权重（`inference/model.py` 里没有 MTP 推理代码）：

文件：`repos/01-attention/DeepSeek-V3/README_WEIGHTS.md` L33–L56
链接：https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md#L33-L56

要点：`num_nextn_predict_layers = 1`；MTP 模块 = 共享的 `embed_tokens` + `enorm`/`hnorm` 两个 RMSNorm + `eh_proj`（降维投影）+ 一个与主干结构相同的 Transformer 层（层号 61）+ 共享的输出头 `shared_head`。

**vLLM 中的推理实现**：`vllm/model_executor/models/deepseek_mtp.py` `DeepSeekMultiTokenPredictorLayer.forward` L113–L145
链接：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/models/deepseek_mtp.py#L113-L144

```python
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)   # 位置 0 没有"前一个特征"，置零
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))   # [T, 2D] -> [T, D]
hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
hidden_states = residual + hidden_states
return hidden_states, self.shared_head(hidden_states)
```

你会发现它和 EAGLE-1 的 `fc(cat(emb, feature))` + 一层 decoder **几乎同构**：区别是 MTP 在**预训练**时就作为辅助损失一起训练（DeepSeek-V3 技术报告），而 EAGLE 是在冻结的目标模型上事后训练的。正因为同构，vLLM 让 `method="mtp"` 走 `EagleProposer`。

**训练侧**：Megatron-LM 的 `megatron/core/transformer/multi_token_prediction.py`，`MultiTokenPredictionLayer`（L1312 起，docstring 引用 DeepSeek-V3 报告），拼接与投影在 L1753–L1758：

链接：https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/transformer/multi_token_prediction.py#L1753-L1758

```python
hidden_states = apply_module(self.hnorm)(hidden_states)
...
hidden_states = torch.cat((decoder_input, hidden_states), -1)
hidden_states, _ = self.eh_proj(hidden_states)
```

---

## 5. 方法对比表

| | 原始投机采样 | Medusa | EAGLE-1 | EAGLE-2 | EAGLE-3 | MTP (DeepSeek-V3) | vLLM ngram |
|---|---|---|---|---|---|---|---|
| 草稿器 | 独立小 LM | K 个 MLP head | 1 层 decoder | 同 EAGLE-1 | 1 层 decoder + 3 层特征融合 + 小词表 | 1 层 decoder（预训练自带） | prompt 中的 n-gram 匹配 |
| 草稿输入 | token | 目标最后层特征 | 特征 + 下一 token emb | 同左 | 低/中/高层特征 + emb | 特征 + 下一 token emb | token |
| 草稿间依赖 | 自回归 | 无（并行） | 自回归 | 自回归 | 自回归 | 自回归（可串多个 MTP 模块） | — |
| 候选结构 | 链 | 静态树 | 静态树 | **动态树**（按置信度） | 动态树 | 链（vLLM） | 链 |
| 验证规则 | rejection sampling（无损） | 贪心匹配 / typical（**有损**） | 多候选 point-mass rejection（无损） | 同左 | 同左 | vLLM rejection sampler（无损） | point-mass（无损） |
| 是否需训练 | 需要一个对齐的小模型 | 训 head（可冻结主干） | 训 1 层 | 同 EAGLE-1 | 训练时用 training-time test | 预训练时联合训练 | 不需要 |
| 本文代码 | 第 3 节 | `medusa/model/utils.py` | `eagle/model/cnets1.py` | `cnets.py: topK_genrate` | `cnets.py` + `modeling_llama_kv.py` | `deepseek_mtp.py` | `ngram_proposer.py` |

---

## 6. 常见困惑 / 坑

1. **"投机解码会改变输出质量吗？"** 用 2.2 节的规则（vLLM、EAGLE、原始论文）：**不会**，分布严格相同；Medusa 的 typical acceptance：**会**（有损，换速度）。贪心模式下所有方法都等价于"和 target argmax 精确匹配"。注意：浮点非确定性（batch 不同导致 kernel 选择不同）可能让 greedy 输出与非投机版本有极少差异，这是数值问题，不是算法问题。
2. **"错一位"**：位置 i 的 logits 验证的是位置 i+1 的 token。Medusa/EAGLE 的 `candidates[:, 1:] == argmax(logits[:, :-1])`、vLLM 的 `target_logits_indices + 1`，全部是这个。
3. **树的 position_id 是深度，不是展平下标**。若用展平下标，兄弟节点的 RoPE 位置不同，结果就错了。
4. **KV cache 回滚**：验证时整棵树/整条链的 KV 都写进了 cache，被拒部分必须作废。Medusa/EAGLE 用 gather+copy 把接受路径挪到连续位置；vLLM 的 paged KV 中只需要不推进 `seq_len`（拒绝的 slot 下一轮被覆盖）。
5. **采样参数必须两边一致**：如果 target 用了 temperature/top-p，`q` 也得是同样变换后的分布，否则 `p/q` 不再对应 2.3 节的证明。vLLM 中 `apply_sampling_constraints` 处理 target，草稿侧在 `compute_probs_and_sample_next_token` 里用同样的 `sampling_metadata`。
6. **"为什么 vLLM 草稿默认用 argmax 还是无损？"** 因为 point-mass 也是一个合法的 q，接受概率 = p(x)。代价是接受率比"按 q 采样"略低一些（取决于草稿与 target 的匹配程度）。
7. **batch 大时收益下降**：投机解码利用的是"小 batch 时算力闲置"。batch 很大时目标模型已接近 compute-bound，多算 γ 个位置不再免费，被拒 token 的计算纯属浪费。这也是 vLLM 有 "Dynamic SD"（代码中 `num_speculative_tokens == 0` 的分支）的原因。
8. **EAGLE-3 的小词表**：`draft_vocab_size < vocab_size` 时草稿只能提出高频 token；低频 token 只能通过"拒绝后从 target 修正分布采样"得到。正确性不受影响，只影响接受率。

---

## 7. 自测题 + 动手练习

**自测题**

1. 写出 2.3 节证明中 `Σ_y max(p(y) − q(y), 0) = 1 − β` 的推导。为什么 q 的支撑集比 p 小时（某些 p(x)>0 但 q(x)=0）算法依然正确？
2. 设 α=0.6、γ=5，计算每轮期望产出 token 数。若 c=0.1，加速比是多少？把 γ 改为 3 呢？
3. 在 Medusa 的 `generate_medusa_buffers` 里，`medusa_choices = [[0],[1],[0,0],[0,1],[1,0]]` 时，写出 `medusa_attn_mask`（6×6）、`medusa_position_ids` 和 `tree_indices`（TOPK=10）。
4. EAGLE `topK_genrate` 中为什么"按累计分数选 top total_tokens 个节点"选出的集合一定是一棵连通的树？
5. vLLM 的 recovered token 采样为什么能不归一化 `max(p − q, 0)`？`inv_q` 每个请求只有一份，为什么不会出问题？
6. Medusa 的 head k 和 EAGLE 的草稿器在"知道前面猜了什么"这件事上的区别是什么？这如何影响接受率？
7. DeepSeek-V3 MTP 模块和 EAGLE-1 草稿器在结构和训练方式上分别有什么异同？

**动手练习**

1. 运行第 3 节的代码；把 `Q_TABLE` 改为与 `P_TABLE` 完全相同，验证接受率为 1、每轮产出恰好 γ+1 个 token。再把 `Q_TABLE` 改为均匀分布，观察 α。
2. 把第 3 节改成"q 为 point mass"（草稿取 `argmax(q)`），接受概率改为 `p(x)`、修正分布改为"抹掉 x"，再跑一遍联合分布检验。
3. 把第 3 节的"接受/拒绝"改为 Medusa 的 typical acceptance（阈值 `min(ε, δ·exp(−H))`），用同样的经验检验观察分布偏差。
4. 用 `torch` 实现 2.5 节的树掩码生成：输入 `parents = [-1, 0, 0, 1, 1, 2]`，用 EAGLE 的 `tree_mask[i] |= tree_mask[parent[i]]` 技巧生成 `[T, T]` 掩码和深度，并与手画结果对照。
5. 阅读 `vllm/v1/sample/rejection_sampler.py` 的 `_get_logprobs_tensors`（L218 起），回答：当某个位置被拒时，返回给用户的 logprob 是谁的？
