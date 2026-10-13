# 05 · KV Cache 压缩与 Token 剪枝：源码导读

> 本章涉及的参考仓库（均为 `repos/04-kv-compression-token-pruning/` 下的浅 submodule）：
>
> | 仓库 | commit | 本章读什么 |
> |---|---|---|
> | NVIDIA/kvpress | `7136c224105ba7b046a6e538742c3589698680b8` | press 抽象、hook 机制、SnapKV / StreamingLLM / TOVA / H2O(Observed) / KNorm / Expected Attention / AdaKV |
> | mit-han-lab/streaming-llm | `2e5042606d69933d88fbf909bd77907456b9b4dd` | attention sink + 最近窗口，位置平移（pos shift） |
> | pkunlp-icler/FastV | `d1659729b5bf1be225e99ee15783deeea80f63b1` | 在第 K 层按注意力分数剪掉图像 token |
> | facebookresearch/ToMe | `af95e4b1befa172dadccd8c81e223b10090f9579` | 二分软匹配（bipartite soft matching）、proportional attention |
> | microsoft/LLMLingua | `5a4c78ae18ab17a98cf997e8259354e546081d64` | 基于困惑度的 prompt token 丢弃、budget controller |
>
> 文中所有行号都按上述 commit 核对过；永久链接格式为 `https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<a>-L<b>`。

---

## 0. 要解决的问题 / 一句话直觉

**问题。** 自回归推理时，每层、每个 KV head 都要为每个历史 token 存一份 `K, V`：

```
KV cache 字节数 = 2(K和V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_elem
```

以 Llama-3-8B（32 层、8 个 KV head、head_dim=128、bf16）为例，每个 token 占 `2×32×8×128×2 = 128 KiB`，128K 上下文就是 16 GiB，比模型权重还大。解码阶段每生成一个 token 都要把整个 KV cache 从显存读一遍，所以 **KV cache 既吃显存又决定解码速度**。

**一句话直觉。** 注意力分布是高度稀疏的：大部分 token 拿到的注意力权重接近 0。既然如此，就可以：

1. **(a) KV cache 淘汰/压缩**：只保留"会被关注"的那部分 `K,V`（StreamingLLM、H2O、SnapKV、TOVA、Expected Attention……），或者把每个数存得更省（KV 量化）。
2. **(b) 模型内部的 token 剪枝/合并**：干脆让不重要的 token 不再往后面的层走（FastV 剪图像 token），或者把相似的 token 合并成一个（ToMe）。这不仅省 KV，还省 FLOPs。
3. **(c) Prompt 压缩**：在文本层面就把 prompt 删短（LLMLingua），对任何模型（包括闭源 API）都适用。

三类方法作用在不同的"层级"上：

```
   文本层面                     模型内部（hidden states）             KV cache
 ┌────────────┐  tokenizer   ┌────────────────────────────┐   写入   ┌──────────┐
 │  prompt    │ ───────────▶ │ layer0 … layerK … layerN   │ ───────▶ │  K, V    │
 └────────────┘              └────────────────────────────┘          └──────────┘
   (c) LLMLingua               (b) FastV / ToMe                       (a) kvpress / StreamingLLM / 量化
```

---

## 1. 核心原理（带 shape）

### 1.1 记号

- `B` batch，`T` 序列长度（prefill 长度），`H` query head 数，`H_kv` KV head 数（GQA 时 `H = H_kv × n_rep`），`d` head_dim。
- 某层 key/value：`K, V ∈ R^{B×H_kv×T×d}`。
- 注意力：`A = softmax(Q Kᵀ / √d + M) ∈ R^{B×H×T_q×T}`，`M` 是因果掩码。
- **压缩比** `r`（kvpress 里叫 `compression_ratio`）：删掉 `r` 比例的 KV，保留 `n_kept = ⌊T(1−r)⌋` 个。

### 1.2 "打分 + Top-k" 统一框架

绝大多数 KV 淘汰方法都可以写成：

```
score = f(Q, K, V, A, hidden)          # shape [B, H_kv, T]，越大越重要
idx   = topk(score, n_kept, dim=-1)    # [B, H_kv, n_kept]
K'    = gather(K, dim=2, idx)          # [B, H_kv, n_kept, d]
V'    = gather(V, dim=2, idx)
```

不同方法只是 `f` 不同：

| 方法 | 打分 `f` | 需要什么 |
|---|---|---|
| StreamingLLM | 前 `n_sink` 个 + 最近窗口 = 1，其余 = 0（与内容无关） | 只要位置 |
| H2O / ObservedAttention | 每个 key 收到的**累积**注意力 `Σ_i A[i, j]`（再除以能看到它的 query 数） | 完整注意力矩阵（eager） |
| TOVA | 最后一个 query 的注意力 `A[-1, j]`，跨 head 平均 | 一行注意力 |
| SnapKV | 最后 `w` 个 query（观察窗口）的注意力平均，再做 1D 平滑池化 | `w` 行注意力 |
| KNorm | `−‖k_j‖₂`（key 范数越小越重要，经验规律） | 只要 K |
| Expected Attention | 把未来 query 建模成高斯 `q ~ N(μ, Σ)`，算 `E[exp(qᵀk/√d)]` 的闭式解，再乘 `‖v_j‖` | Q 的统计量 + RoPE |

**SnapKV 的公式**（单个 head）：

```
Q_obs = Q[T−w : T]                          # [w, d]   观察窗口里的 query
A_obs = softmax(Q_obs Kᵀ/√d + M)[:, :T−w]   # [w, T−w] 只看窗口之前的 key
s     = mean_i A_obs[i, :]                  # [T−w]
s     = avg_pool1d(s, kernel)               # 平滑，让被选中的位置"成簇"，保留局部上下文
GQA:  s 在同组 n_rep 个 query head 上取平均 → [H_kv, T−w]
保留: topk(s) ∪ {窗口内全部 w 个 token}
```

**Expected Attention 的公式**：若 `q ~ N(μ, Σ)`，则对数正态的期望

```
E[exp(qᵀk/√d)] = exp( μᵀk/√d + ½ · kᵀ Σ k / d )
```

再考虑 RoPE：未来位置 `t+Δ` 的 query 要乘旋转矩阵 `R_{t+Δ}`，把 `n_future_positions` 个位置的 `R` 取平均得 `R̄`，用 `μ' = R̄μ, Σ' = R̄ΣR̄ᵀ` 代替。

### 1.3 Attention sink 与位置平移（StreamingLLM）

观察：无论内容如何，模型都会把大量注意力分给**最开始的几个 token**（softmax 必须把概率质量"倒"在某处，开头 token 对所有后续 query 都可见，于是被训练成"垃圾桶"）。如果直接用滑动窗口把它们踢掉，困惑度会爆炸。StreamingLLM 的做法：

```
cache = K[0 : start_size] ⊕ K[T − recent_size : T]      # 4 个 sink + 最近若干个
```

此外，为了让模型永远不会看到"超出训练长度"的相对位置，**RoPE 位置按 token 在 cache 中的下标重新分配**（而不是它在原文中的绝对位置）。这要求 cache 里存**未旋转**的 K，在每次 attention 时再按 `0..kv_len−1` 旋转。

### 1.4 FastV

LVLM（如 LLaVA）里一张图 = 576 个视觉 token，占了大部分序列。FastV 发现：第 2 层之后，图像 token 得到的注意力极低。于是在第 `K` 层，用第 `K−1` 层的注意力（**最后一个 token 对每个图像 token 的注意力，跨 head 平均**）给图像 token 排序，只保留 top `R` 个：

```
a = mean_h A^{(K−1)}[h, −1, sys : sys+576]     # [576]
keep = [0..sys) ∪ (topR(a)+sys) ∪ [sys+576 .. T)
hidden = hidden[:, keep, :]                    # 之后的所有层序列都变短
```

### 1.5 ToMe

ViT 里每层把 token 数减少 `r` 个：把 token 交替分成 A（偶数位）/B（奇数位）两组，A 中每个 token 找 B 里最相似的（余弦相似度，度量用 attention 的 key），选出相似度最高的 `r` 条边，把这 `r` 个 A token **加权平均**合并进对应的 B token。被合并的 token 记一个 `size`（代表了多少个原始 patch），后续注意力用 **proportional attention**：

```
A = softmax( QKᵀ/√d + log(size)ᵀ )
```

这相当于"一个代表了 s 个 token 的 key 在 softmax 里被复制了 s 次"，保证合并前后注意力分布尽量一致。

### 1.6 LLMLingua

用一个小 LM（如 LLaMA-7B / GPT-2）计算每个 token 的自信息 `−log p(x_i | x_<i)`。**信息量低（容易预测）的 token 删掉对语义影响小**。为了满足总预算，又分三层控制：

1. **context 级**（budget controller）：多个文档/示例先按困惑度排序，按预算挑选；
2. **sentence 级**（可选）；
3. **token 级**：按 `iterative_size` 分段迭代，在每段里求阈值，只保留 loss（自信息）高于阈值的 token；已经压缩好的前缀作为 KV cache 条件，后面段落在"压缩后的上下文"下重新算困惑度（这就是 "iterative"）。

### 1.7 KV 量化（概念）

不删 token，而是把每个 `K,V` 元素从 16bit 降到 8/4/2bit：

```
x_q = round((x − z) / s),   x̂ = s · x_q + z     # 按 group 求 scale s / zero z
```

要点：K 有明显的**按通道（channel）离群值**，适合 per-channel 量化；V 更适合 per-token 量化（KIVI 的结论）。量化与淘汰正交，可以叠加：kvpress 在写回 cache 时就专门处理了 `QuantizedCache`（见 §3.1）。

---

## 2. 最小实现

### 2.1 SnapKV 打分 + 选择（含 GQA）

```python
import torch
import torch.nn.functional as F

def snapkv_select(q, k, v, window=32, kernel=5, keep_ratio=0.5):
    """
    q: [B, Hq, T, D]   已经加过 RoPE 的 query（prefill 时的全部 query）
    k, v: [B, Hkv, T, D]  这一层的 KV cache
    返回压缩后的 k, v: [B, Hkv, n_keep, D] 以及保留下标
    """
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    n_rep = Hq // Hkv
    k_rep = k.repeat_interleave(n_rep, dim=1)               # [B, Hq, T, D]   GQA 展开

    q_obs = q[:, :, -window:]                                # [B, Hq, W, D]   观察窗口
    logits = q_obs @ k_rep.transpose(-1, -2) / D ** 0.5      # [B, Hq, W, T]
    qpos = torch.arange(T - window, T)[:, None]              # 窗口里第 i 个 query 的绝对位置
    kpos = torch.arange(T)[None, :]
    logits = logits.masked_fill(kpos > qpos, float("-inf"))  # 因果掩码
    attn = logits.softmax(-1)[..., : T - window]             # [B, Hq, W, T-W]  只给窗口外的 key 打分

    score = attn.mean(dim=-2)                                # [B, Hq, T-W]     对窗口内 query 平均
    score = F.avg_pool1d(score, kernel, stride=1, padding=kernel // 2)  # 平滑
    score = score.view(B, Hkv, n_rep, T - window).mean(2)    # [B, Hkv, T-W]    同组 query head 平均

    n_keep = int(T * keep_ratio) - window
    idx = score.topk(n_keep, dim=-1).indices.sort(-1).values # [B, Hkv, n_keep]
    win = torch.arange(T - window, T).expand(B, Hkv, window) # 窗口本身永远保留
    idx = torch.cat([idx, win], dim=-1)                      # [B, Hkv, n_keep+W]

    gidx = idx[..., None].expand(-1, -1, -1, D)
    return k.gather(2, gidx), v.gather(2, gidx), idx

if __name__ == "__main__":
    torch.manual_seed(0)
    B, Hq, Hkv, T, D = 1, 8, 2, 256, 64
    q, k, v = torch.randn(B, Hq, T, D), torch.randn(B, Hkv, T, D), torch.randn(B, Hkv, T, D)
    k2, v2, idx = snapkv_select(q, k, v)
    print(k2.shape)   # torch.Size([1, 2, 128, 64])
```

> 把 `score` 换成 `attn[..., -1:, :].mean(1)` 就是 TOVA；换成整个注意力矩阵按列求和就是 H2O；换成"前 4 个 + 最后 n 个"就是 StreamingLLM。kvpress 正是用这种"只换 `score`"的方式组织几十种方法的。

### 2.2 ToMe 二分软匹配 + 加权合并 + proportional attention

```python
import torch

def bipartite_merge(x, metric, r, size=None):
    """
    x:      [B, N, C]  token 表示
    metric: [B, N, Ck] 相似度度量（ToMe 用多头 key 的均值）
    r:      本层要减少的 token 数（<= N//2）
    size:   [B, N, 1]  每个 token 代表的原始 patch 数
    """
    B, N, C = x.shape
    if size is None:
        size = x.new_ones(B, N, 1)
    m = metric / metric.norm(dim=-1, keepdim=True)
    a, b = m[:, ::2], m[:, 1::2]                    # A=偶数位, B=奇数位
    scores = a @ b.transpose(-1, -2)                # [B, Na, Nb] 余弦相似度
    node_max, node_idx = scores.max(-1)             # 每个 A token 的最佳 B 伙伴及相似度
    order = node_max.argsort(-1, descending=True)   # 相似度从高到低
    src, unm = order[:, :r], order[:, r:]           # 前 r 条边被合并；其余 A token 原样保留
    dst = node_idx.gather(1, src)                   # 被合并 A token 的目标 B 下标

    def merge(t, mode):
        ta, tb = t[:, ::2], t[:, 1::2]
        c = t.shape[-1]
        t_unm = ta.gather(1, unm[..., None].expand(-1, -1, c))
        t_src = ta.gather(1, src[..., None].expand(-1, -1, c))
        tb = tb.scatter_reduce(1, dst[..., None].expand(-1, -1, c), t_src, reduce=mode)
        return torch.cat([t_unm, tb], dim=1)        # [B, N-r, c]

    x = merge(x * size, "sum")                      # 按 size 加权求和
    size = merge(size, "sum")
    return x / size, size                           # 加权平均

def proportional_attn(q, k, v, size):
    # q,k,v: [B, h, N, d]; size: [B, N, 1]
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    logits = logits + size.log()[:, None, None, :, 0]   # 一个 size=s 的 key 相当于被复制 s 次
    return logits.softmax(-1) @ v

if __name__ == "__main__":
    x = torch.randn(2, 197, 768)
    y, s = bipartite_merge(x, x, r=16)
    print(y.shape, s.sum(1).squeeze())   # [2, 181, 768]，size 之和仍为 197
```

---

## 3. 源码导读（按阅读顺序）

### 3.1 kvpress：press 抽象 + forward hook

**阅读顺序**：`BasePress`（hook 如何挂到 attention 上）→ `ScorerPress`（打分 + topk 的公共逻辑）→ 各个 `score()` 实现 → `AdaKVPress` 与 `attention_patch`（head-wise 不等长压缩）→ `pipeline._forward`（压缩发生在什么时刻）。

#### Stop 1 · `BasePress.__call__`：把 press 挂到每个 attention 层

- 文件：`kvpress/presses/base_press.py`，`BasePress.__call__`，L164–L207
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/base_press.py#L164-L207

```python
194        self.post_init_from_model(model)
195        hooks = []
196        try:
197            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
198            for layer in language_model.layers:
...
202                layer.self_attn.rotary_emb = language_model.rotary_emb
203                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
204            yield
205        finally:
206            for forward_hook in hooks:
207                forward_hook.remove()
```

逐行解释：

- press 是一个 **context manager**：`with press(model): model(...)`。进入时给**每个解码层的 `self_attn` 模块**注册一个 PyTorch forward hook，退出时移除。所以 kvpress **不改任何模型代码**，完全靠 hook。
- `with_kwargs=True` 让 hook 能拿到 attention 的关键字参数（`hidden_states`、`past_key_values`、`cache_position`、`position_embeddings`），这些是打分要用的。
- L202 把模型级的 `rotary_emb` 塞进每个 attention 模块，是给 Expected Attention 之类需要自己算 RoPE 的 press 用的（见 Stop 6 的 `module.rotary_emb(...)`）。

#### Stop 2 · `BasePress.forward_hook`：attention 跑完之后，原地改 cache

- 文件：`kvpress/presses/base_press.py`，`forward_hook` L101–L162；`is_prefilling` L37–L40
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/base_press.py#L101-L162

```python
 37 def is_prefilling(cache_position: torch.Tensor, q_len: int) -> bool:
 39     prefilling = cache_position[-1] + 1 == q_len
...
139        hidden_states = kwargs["hidden_states"]
140        cache = kwargs["past_key_values"]
141        cache_layer = cache.layers[module.layer_idx]
142        q_len = hidden_states.shape[1]
144        # Don't compress after pre-filling
145        if not is_prefilling(kwargs["cache_position"], q_len):
146            return output
148        keys, values = extract_keys_and_values(cache, module.layer_idx)
150        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
152        if isinstance(cache, QuantizedCache):
153            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
154            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
...
158        else:
159            cache_layer.keys = keys
160            cache_layer.values = values
```

- **时机**：forward hook 在 attention 的 `forward` **返回之后**执行。此时这层的 `K,V` 已经被 HF 写进 `DynamicCache`，attention 输出也已用**完整** KV 算完——所以压缩不影响 prefill 本身的结果，只影响**之后**解码阶段看到的 cache（每层只压缩自己那一层的 cache，也不影响后面层的 prefill）。
- `is_prefilling`：`cache_position` 是本次输入 token 在整个序列中的位置。若最后一个位置 +1 等于本次输入长度，说明本次从 0 开始——就是 prefill。解码阶段（`q_len=1`、`cache_position=[t]`）直接返回不压缩。
- `keys/values` 的 shape 是 `[B, H_kv, T, d]`；`output[1]` 是注意力权重，只有 `attn_implementation="eager"` 时才不是 `None`。
- L152–L157：如果用的是量化 cache，压缩后要**重新量化**写回。这就是"淘汰 + 量化"可叠加的落点。

#### Stop 3 · `ScorerPress.compress`：打分 → topk → gather

- 文件：`kvpress/presses/scorer_press.py`，`ScorerPress` L17–L103，`compress` L77–L103
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/scorer_press.py#L77-L103

```python
 90        # Compute scores
 91        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
 93        # Get indices of KV pairs with the lowest scores
 94        k_len = keys.shape[2]
 95        n_kept = compute_n_kept(k_len, self.compression_ratio)
 96        indices = scores.topk(n_kept, dim=-1).indices
 97        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
 99        # Prune keys and values
100        keys = keys.gather(2, indices).contiguous()
101        values = values.gather(2, indices).contiguous()
```

- `scores`: `[B, H_kv, T]`。`topk(..., dim=-1)` 对**每个 KV head 独立**选 `n_kept` 个——不同 head 可以保留不同 token，但每个 head 保留的**数量**相同（否则张量不规则）。
- `indices` 从 `[B,H_kv,n_kept]` 扩成 `[B,H_kv,n_kept,d]` 才能对 4 维的 `keys` 做 `gather(dim=2)`。
- 注意 `topk` 返回的下标**不排序**。这没关系：cache 里的 K 已经带了 RoPE（位置信息"烤"进了向量），注意力对 key 的排列顺序不敏感（softmax 加权和是置换不变的）。
- `compute_n_kept`（`kvpress/utils.py` L137–L147，https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/utils.py#L137-L147）：`max(1, int(k_len * (1 - compression_ratio)))`，`ratio>=1` 时返回 0。

#### Stop 4 · `SnapKVPress`：观察窗口打分

- 文件：`kvpress/presses/snapkv_press.py`，`compute_window_attention` L41–L67，`score` L69–L103
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/snapkv_press.py#L41-L103

```python
 53        cos, sin = position_embeddings
 54        query_states = get_query_states(
 55            module, hidden_states[:, -window_size:], (cos[:, -window_size:], sin[:, -window_size:])
 56        )
 59        key_states = repeat_kv(keys, num_key_value_groups)
 60        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
 61        attention_mask = torch.ones_like(attn_weights) * float("-inf")
 62        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
 63        attn_weights += attention_mask
 64        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
 65        attn_weights = attn_weights[..., :-window_size]
```

- **为什么要重算 query？** HF 的 KV cache 只存 K/V，不存 Q；而 FlashAttention/SDPA 也不返回注意力矩阵。所以 SnapKV 从 hook 拿到的 `hidden_states`（这一层 attention 的输入，`[B,T,hidden]`）里取最后 `w` 个，用该层的 `q_proj`（`get_query_states`，`kvpress/utils.py` L57–L88）重新投影并加 RoPE，得到 `[B, H, w, d]`。只算 `w` 行，代价是 `O(w·T)` 而非 `O(T²)`。
- L61–L62：构造因果掩码。窗口第 `i` 行 query 的绝对位置是 `k_len − w + i`，它不能看见 `> k_len − w + i` 的 key，`triu(diagonal=k_len−w+1)` 正好把这些位置置为 `-inf`。
- L65：只保留窗口**之前**的列 `[..., :T−w]`——窗口内的 token 会被无条件保留，不需要打分。

```python
 86        if attentions is not None:
 87            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
 88        else:
 89            attn_weights = self.compute_window_attention(...)
 93        scores = attn_weights.mean(dim=-2)
 94        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
 97        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
 98        scores = scores.mean(2)
101        scores = F.pad(scores, (0, self.window_size), value=scores.max().item() + 1)
```

- L86–L87：如果是 eager 模式、注意力矩阵现成，就直接切片复用。
- L93：`[B,H,w,T−w] → [B,H,T−w]`，窗口内 query 的平均关注度。
- L94：`avg_pool1d` 把分数在序列维上平滑（kernel=5）。直觉：被关注的往往是一个"短语"，平滑后被选中的 token 成簇，保留局部上下文，避免只剩孤立的单个 token。
- L97–L98：GQA 时 `H = H_kv × n_rep`，同一组的 `n_rep` 个 query head 共享一份 K/V，所以只能按组平均成 `[B, H_kv, T−w]`。
- L101：给窗口补上"比所有分数都大"的分数，确保 topk 一定选中它们。

#### Stop 5 · 其它几个 `score()`：一眼看懂方法之间的差异

**StreamingLLM**：`kvpress/presses/streaming_llm_press.py` L38–L54
https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/streaming_llm_press.py#L38-L54

```python
 50        n_pruned = k_len - int(k_len * (1 - self.compression_ratio))
 51        scores = torch.ones_like(keys[..., 0])
 52        scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0
```

分数与内容无关：sink 之后紧跟的 `n_pruned` 个 token 记 0，其它记 1 → topk 留下"前 `n_sink` 个 + 最后若干个"。docstring（L22）提醒：要完全复现论文的位置平移需要外面再套 `KeyRerotationPress`。

**TOVA**：`kvpress/presses/tova_press.py` L35–L61
https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/tova_press.py#L35-L61

```python
 45        if attentions is not None:
 46            attn_weights = attentions[..., -1:, :-1]
 47        else:
 48            attn_weights = SnapKVPress.compute_window_attention(
 49                module, hidden_states, keys, 1, kwargs["position_embeddings"]
 50            )
 53        scores = attn_weights.mean(1)
 54        scores = scores.repeat(1, keys.shape[1], 1)
```

TOVA = window_size=1 的 SnapKV，但**跨所有 head 平均**（L53，`[B,H,1,T−1] → [B,1,T−1]`）后复制给每个 KV head——所有 head 保留同一组 token。

**H2O（ObservedAttentionPress）**：`kvpress/presses/observed_attention_press.py` L34–L49
https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/observed_attention_press.py#L34-L49

```python
 43        assert attentions is not None, 'Set attn_implementation="eager" to use this hook'
 44        scores = attentions.sum(2)
 46        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(attentions.device, attentions.dtype)
 47        scores = scores / n_tokens_in_sum
 48        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
```

- L44：`[B,H,T,T]` 沿 query 维求和 → 每个 key 收到的**累积注意力**（H2O 的 "heavy hitter"）。
- L46–L47：**为什么要除？** 因果掩码下，第 `j` 个 key 只能被 `T−j` 个 query 看见，越早的 token 累积和天然越大。除以 `T, T−1, …, 1` 得到平均值，去掉这种位置偏置。
- 需要 eager 注意力（完整 `T×T` 矩阵），长上下文时显存是 `O(T²)`，这是 H2O 类方法的工程痛点。

**KNorm**：`kvpress/presses/knorm_press.py` L32–L46
https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/knorm_press.py#L32-L46

```python
 45        score_dtype = torch.promote_types(keys.dtype, torch.float32)
 46        return -torch.linalg.vector_norm(keys, dim=-1, dtype=score_dtype)
```

完全不看注意力：key 范数**小**的反而更重要（取负号）。L41–L44 的注释很有教育意义：bf16 下算范数会大量"打平"（并列），于是到底删谁变成由 topk 的并列规则决定，所以要提升到 fp32。

#### Stop 6 · `ExpectedAttentionPress`：用 query 的统计量预测"未来注意力"

- 文件：`kvpress/presses/expected_attention_press.py`，`get_query_statistics` L62–L86、`apply_avg_rope` L88–L124、`score` L126–L165
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/expected_attention_press.py#L62-L165

```python
 70        h = hidden_states[:, self.n_sink :]
 71        query_states = get_prerope_query_states(module, h)
 74        mu = query_states.mean(dim=2, keepdim=True)
 79            centered_states = query_states - mu
 80            cov = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / h.shape[1]
 84        mu, cov = self.apply_avg_rope(module, mu, cov, q_len)
```

- 用**未加 RoPE** 的 query（`get_prerope_query_states`）估计均值 `μ: [B,H,d]` 和协方差 `Σ: [B,H,d,d]`；去掉前 `n_sink` 个 token，因为 sink 位置的 hidden state 常有离群值。

```python
110        position_ids = torch.arange(q_len, q_len + self.n_future_positions).unsqueeze(0).to(mu.device)
112        cos, sin = module.rotary_emb(mu, position_ids)
116        P[head_dim // 2 :, : head_dim // 2], P[: head_dim // 2, head_dim // 2 :] = torch.eye(head_dim // 2), -torch.eye(head_dim // 2)
119        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
120        R = R.mean(dim=0).to(mu.device)
121        mu = torch.matmul(mu, R.T)
123            cov = torch.matmul(R, torch.matmul(cov, R.T))
```

- 把 HF 的 `rotate_half` 写成矩阵形式：`RoPE(x) = cos⊙x + sin⊙rotate_half(x) = (diag(cos) + diag(sin)·P) x`，`P` 就是 `rotate_half` 对应的置换+取负矩阵。对未来 `n_future_positions`（默认 512）个位置的 `R` 取平均 `R̄`，再变换 `μ, Σ`。

```python
148        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
149        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
151            scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2
152        scores = F.softmax(scores, dim=-1)
160            scores = (scores + self.epsilon) * values.norm(dim=-1)
163        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item() + 1)
```

- L149 + L151 正是 `μᵀk/√d + ½kᵀΣk/d`（对数正态期望的指数部分），L152 softmax 把它归一化成"期望注意力"。
- L160：注意力只是权重，真正影响输出的是 `a_j · v_j`，所以再乘上 `‖v_j‖`。
- L163：sink 永远保留。

#### Stop 7 · `AdaKVPress` + `attention_patch`：每个 head 保留不同数量的 token

- `kvpress/presses/adakv_press.py` `compress` L54–L81：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/adakv_press.py#L54-L81
- `kvpress/attention_patch.py` `attention_patch` L51–L95，`search_hyperplane` L8–L48：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/attention_patch.py#L8-L95

```python
 66        n_safe = int(n_kept * self.alpha_safeguard)
 69        top_indices = torch.topk(scores, n_safe, dim=-1).indices
 70        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)
 73        n_pruned = num_key_value_heads * (k_len - n_kept)
 74        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()
 77        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
 78        head_indices = indices // k_len
 79        seq_indices = indices % k_len
 80        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
 81        return keys, values
```

- AdaKV 的思想：不同 head 的注意力集中程度不同，应该**跨 head 统一排名**（L74 把 `[B,H_kv,T]` 拉平成 `[B,H_kv·T]` 再取最小的那些），但每个 head 至少保留 `alpha_safeguard` 比例（L66–L70 先把每个 head 的 top-`n_safe` 设成最大值）。
- 关键工程问题：每个 head 保留的数量不同 → 张量不规则。kvpress 的解法是**不真的删**（L81 原样返回 keys/values），而是记下被删的位置，在解码时把那些 key 替换成"假 key"：

```python
 82            k = search_hyperplane(q)
 86            batch_indices, head_indices, seq_indices = module.masked_key_indices
 87            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]
```

- `search_hyperplane` 找一个向量 `Y` 使所有当前 query 满足 `⟨q, Y⟩ ≤ 0`，再返回 `−1e5·Y/‖Y‖²`，于是 `⟨q, k_fake⟩` 是极大的负数，`exp` 后下溢为 0——等价于 mask 掉。docstring（L54–L55）坦白：这**不省显存**，只是模拟效果，用于研究。
- 这个 patch 在 `import kvpress` 时全局生效：`kvpress/__init__.py` L55 调用 `patch_attention_functions()`。

#### Stop 8 · `pipeline._forward`：压缩发生在"context prefill"阶段

- 文件：`kvpress/pipeline.py`，`_forward` L173–L254
- 链接：https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/pipeline.py#L173-L254

```python
215        perform_prefill_compression = press is not None and not isinstance(press, DecodingPress)
216        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
217            # We run the model without the lm head for pre-filling.
218            self.model.model(
219                input_ids=context_ids,
220                past_key_values=cache,
221            )
```

- 只对 **context**（长文档）做 prefill + 压缩，**问题（question）不在压缩范围内**——这就是 kvpress 的评测设定（"query-agnostic"压缩）：压缩时不知道将来问什么。SnapKV 原论文是把问题放在观察窗口里的（query-aware），所以在 kvpress 设定下 SnapKV 表现会比论文差，这是读 benchmark 时要注意的点。
- L236–L251：每个问题解码完后 `_remove_answer_from_cache` 把答案的 KV 切掉，下一个问题复用同一份压缩后的 context cache。

### 3.2 streaming-llm：sink + 最近窗口 + 位置平移

#### Stop 9 · `StartRecentKVCache`

- 文件：`streaming_llm/kv_cache.py`，类 L23–L119；`__call__` L40–L64；`evict_for_space` L66–L94
- 链接：https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/streaming_llm/kv_cache.py#L23-L94

```python
 66    def evict_for_space(self, past_key_values, num_coming):
 69        seq_len = past_key_values[0][0].size(self.k_seq_dim)
 70        if seq_len + num_coming <= self.cache_size:
 71            return past_key_values
 72        return [
 73            [
 74                torch.cat(
 75                    [
 76                        self.k_slice(k, 0, self.start_size),
 77                        self.k_slice(
 78                            k, seq_len - self.recent_size + num_coming, seq_len
 79                        ),
 80                    ],
```

- `past_key_values` 是老式 HF 格式：`[(k, v) for each layer]`，Llama 的 `k: [B, H_kv, seq, d]`，所以 `k_seq_dim=2`；MPT 的 K 是 `[B, H, d, seq]`，所以 `enable_streaming_llm` 里 MPT 设 `k_seq_dim=3`（`streaming_llm/enable_streaming_llm.py` L12–L14）。
- `evict_for_space(…, num_coming)`：**预先腾位置**。新来 `num_coming` 个 token 之前，保留前 `start_size`（默认 4）个 sink 和最近 `recent_size − num_coming` 个，保证加入新 token 后总长不超过 `cache_size`。
- 调用处：`examples/run_streaming_llama.py` L60–L76（https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/examples/run_streaming_llama.py#L60-L76），每轮对话前 `space_needed = seq_len + max_gen_len`。

#### Stop 10 · `llama_pos_shift_attention_forward`：cache 内位置重编号

- 文件：`streaming_llm/pos_shift/modify_llama.py`，L85–L104（核心），`apply_rotary_pos_emb_single` L21–L28，`enable_llama_pos_shift_attention` L164–L174
- 链接：https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/streaming_llm/pos_shift/modify_llama.py#L85-L104

```python
 85    kv_seq_len = key_states.shape[-2]
 86    if past_key_value is not None:
 87        kv_seq_len += past_key_value[0].shape[-2]
 88    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
 89    ### Shift Pos: query pos is min(cache_size, idx)
 90    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
 91    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
 94    if past_key_value is not None:
 96        key_states = torch.cat([past_key_value[0], key_states], dim=2)
 97        value_states = torch.cat([past_key_value[1], value_states], dim=2)
 99    past_key_value = (key_states, value_states) if use_cache else None
101    ### Shift Pos: key pos is the pos in cache
102    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
103    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
```

和原版 Llama attention 的区别只有三处：

1. L90 注释掉了原来"q 和 k 一起旋转"的调用；
2. L99：**存进 cache 的是未旋转的 K**（先 concat，再存，再旋转）；
3. L102–L103：每次前向都把 cache 里的全部 key 按 `0..kv_seq_len−1` 重新旋转。

为什么？淘汰中间 token 后，若 K 保留原始绝对位置，sink 在位置 0、最近 token 在位置 10 万，相对距离会超出训练长度。重编号后 cache 里的相对距离永远 ≤ `cache_size`。query 的 `position_ids` 由 HF 根据 `past_key_values` 长度生成（即 cache 长度），所以 query 位置也是 "min(cache_size, idx)"（L89 注释）。代价：每步都要对整个 cache 重做 RoPE。

### 3.3 FastV：第 K 层按注意力剪图像 token

#### Stop 11 · `LlamaModel.forward` 中的 FastV 分支

- 文件：`src/transformers/src/transformers/models/llama/modeling_llama.py`（FastV 仓库里魔改的 transformers），L717–L822
- 链接：https://github.com/pkunlp-icler/FastV/blob/d1659729b5bf1be225e99ee15783deeea80f63b1/src/transformers/src/transformers/models/llama/modeling_llama.py#L717-L822
- 配置字段在同目录 `configuration_llama.py` L125–L162；同目录 `fastv_forward.py` 是带中文注释的同一段逻辑（L94–L100 解释了 `SYS_LENGTH=36`、`IMAGE_TOKEN_LENGTH=576` 等含义）。

```python
717                USE_FAST_V = self.use_fast_v
718                SYS_LENGTH= self.fast_v_sys_length
719                IMAGE_TOKEN_LENGTH = self.fast_v_image_token_length
720                ATTENTION_RANK = self.fast_v_attention_rank
721                AGG_LAYER = self.fast_v_agg_layer
...
736                    elif idx==AGG_LAYER:
738                        last_layer_attention = layer_outputs[1]
740                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
742                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
744                        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
746                        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
748                        keep_indexs = torch.cat( (torch.arange(SYS_LENGTH,device=device), top_attention_rank_index, torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH,seq_length_with_past,device=device)))
750                        keep_indexs = keep_indexs.sort().values
752                        new_seq_length = keep_indexs.shape[0]
754                        hidden_states = hidden_states[:,keep_indexs,:]
756                        position_ids = keep_indexs.unsqueeze(0)
758                        new_attention_mask = self._prepare_decoder_attention_mask(
759                            None, (batch_size, new_seq_length), inputs_embeds, 0
760                        )
```

逐行（这是 `FASTV_INPLACE=True` 的"真删"版本）：

- 这段代码位于"逐层循环"里、**调用第 `idx` 层之前**。当 `idx == AGG_LAYER` 时，`layer_outputs` 还是**上一层**（第 `K−1` 层）的输出，`layer_outputs[1]` 是它的注意力 `[B, H, T, T]`——因此要求 `output_attentions=True`（eager attention）。
- L740：`mean(dim=1)` 对 head 平均，`[0]` 取 batch 第 0 个（**只支持 batch=1**）→ `[T, T]`。
- L742：取最后一行 = 最后一个 token（通常是指令末尾）对所有 token 的注意力 `[T]`。
- L744–L746：只看图像段 `[SYS, SYS+576)`，选 top `ATTENTION_RANK`，加回偏移得到全局下标。
- L748–L750：保留 = 系统提示 + 被选中的图像 token + 图像之后的文本；排序以保持原顺序。
- L754：**真正把序列变短**，第 K 层及之后所有层都只处理 `new_seq_length` 个 token，FLOPs 随之下降。
- L756：`position_ids` 用**原始位置**（`keep_indexs`），保证 RoPE 位置与剪枝前一致——和 StreamingLLM 的"重编号"策略恰好相反。

L764–L808 是另一种实现（`FASTV_INPLACE=False`）：不删 hidden states，而是生成一个把未选中图像 token 屏蔽掉的 attention mask（L785–L787）。这个版本序列长度不变，方便与 KV cache 配合（仓库里另有 `modeling_llama_fastv_kvcache.py`），但不省 FLOPs，用于验证精度。L794–L804 是 `AGG_LAYER==0` 时没有上一层注意力可用，只能随机丢。

### 3.4 ToMe：二分软匹配

#### Stop 12 · `bipartite_soft_matching`

- 文件：`tome/merge.py`，L18–L97
- 链接：https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/merge.py#L18-L97

```python
 43    t = metric.shape[1]
 44    r = min(r, (t - protected) // 2)
 49    with torch.no_grad():
 50        metric = metric / metric.norm(dim=-1, keepdim=True)
 51        a, b = metric[..., ::2, :], metric[..., 1::2, :]
 52        scores = a @ b.transpose(-1, -2)
 54        if class_token:
 55            scores[..., 0, :] = -math.inf
 59        node_max, node_idx = scores.max(dim=-1)
 60        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
 62        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
 63        src_idx = edge_idx[..., :r, :]  # Merged Tokens
 64        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
```

- 输入 `metric: [B, N, C]`（ToMe 用多头 key 的平均，见 Stop 13）。
- L44：每层最多减半（A 组最多全部合并进 B 组）。
- L50–L52：L2 归一化后点积 = 余弦相似度；`a: [B, ⌈N/2⌉, C]`（偶数位）、`b: [B, ⌊N/2⌋, C]`（奇数位），`scores: [B, Na, Nb]`。**为什么二分？** 若在全部 token 间做匹配，会出现链式合并（a→b→c），需要迭代/聚类；二分图上每个 A 只连一个 B，一次 `max` + `argsort` 就完成，完全并行、无需迭代，这是 ToMe 快的关键。
- L54–L55：CLS token 在 A 组第 0 个，把它的整行设成 `-inf`，它永远排在最后、不会被合并。
- L59–L64：每个 A token 找最佳 B 伙伴（`node_idx`），按相似度降序排序，前 `r` 个 A token（`src_idx`）合并进它们的伙伴（`dst_idx`），其余（`unm_idx`）保留。
- 整段在 `no_grad` 下：匹配是离散决策，不需要梯度；梯度通过后面的 `merge`（gather/scatter）流过。

```python
 70    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
 71        src, dst = x[..., ::2, :], x[..., 1::2, :]
 72        n, t1, c = src.shape
 73        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
 74        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
 75        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
 80            return torch.cat([unm, dst], dim=1)
```

- `merge` 是一个**闭包**：匹配只算一次，但可以作用于任意与 token 对齐的张量（`x`、`size`、`source`）。
- L75：`scatter_reduce` 把多个 src 聚到同一个 dst（多个 A 可能选中同一个 B），`include_self` 默认为 True，所以 dst 自己也参与平均/求和。
- 输出 `[B, N−r, C]`；`unmerge`（L82–L95）是逆操作，用于 dense 预测任务把 token 还原回原位置。

#### Stop 13 · `ToMeBlock` / `ToMeAttention` / `merge_wavg`

- `tome/patch/timm.py` `ToMeBlock.forward` L34–L56、`ToMeAttention.forward` L66–L96、`apply_patch` L116–L150：https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/patch/timm.py#L21-L150
- `tome/merge.py` `merge_wavg` L210–L224：https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/merge.py#L210-L224

```python
 36        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
 37        x_attn, metric = self.attn(self.norm1(x), attn_size)
 38        x = x + self._drop_path1(x_attn)
 40        r = self._tome_info["r"].pop(0)
 41        if r > 0:
 43            merge, _ = bipartite_soft_matching(metric, r, self._tome_info["class_token"], self._tome_info["distill_token"])
 53            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
 55        x = x + self._drop_path2(self.mlp(self.norm2(x)))
```

- 合并发生在 **attention 之后、MLP 之前**：attention 已经让 token 交换过信息，合并损失更小；MLP 是逐 token 的，放在合并后正好少算 `r` 个 token。
- L40：`r` 是每层一个值的列表（`tome/utils.py` `parse_r` L80–L105 支持常数、递减/递增 schedule）。

```python
 85        if size is not None:
 86            attn = attn + size.log()[:, None, None, :, 0]
 96        return x, k.mean(1)
```

- L86：proportional attention。`size: [B, N, 1]` → `[B, 1, 1, N]` 加在每个 key 列上，`exp(logit + log s) = s·exp(logit)`。
- L96：返回 key 在 head 维上的平均作为相似度度量 `[B, N, d]`——key 本来就是为了"被比较"而学出来的表示，比直接用 `x` 更适合度量相似性。

```python
220    x = merge(x * size, mode="sum")
221    size = merge(size, mode="sum")
223    x = x / size
```

- 加权平均：一个代表了 10 个 patch 的 token 与一个代表 1 个 patch 的 token 合并时，权重 10:1。
- `apply_patch`（L116–L150）通过直接改写 `module.__class__` 把 timm 的 `Block`/`Attention` 换成 ToMe 版本，**不需要重新加载权重**，所以 ToMe 可以"即插即用"到已训练好的 ViT 上。

### 3.5 LLMLingua：基于困惑度的 prompt 压缩

#### Stop 14 · `compress_prompt`：三级流程入口

- 文件：`llmlingua/prompt_compressor.py`，`PromptCompressor.compress_prompt` L426–L725
- 链接：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L426-L725

```python
587        if target_token == -1:
588            target_token = (
589                (
590                    instruction_tokens_length
591                    + question_tokens_length
592                    + sum(context_tokens_length)
593                )
594                * rate
595                - instruction_tokens_length
596                - (question_tokens_length if concate_question else 0)
597            )
...
601        if len(context) > 1 and use_context_level_filter:
602            context, dynamic_ratio, context_used = self.control_context_budget(
...
683        if use_token_level_filter:
684            context = self.iterative_compress_prompt(
```

- L587–L597：总预算 = 原长 × `rate`；instruction 和 question 不压缩，所以从预算里扣掉，剩下的都给 context。
- L601：多个 context（few-shot 示例 / 检索文档）时先做 context 级筛选（Stop 15）。
- L629：sentence 级（可选）。
- L683：token 级迭代压缩（Stop 16）。
- L714–L725：用 tiktoken（`oai_tokenizer`）统计压缩前后长度、压缩比，返回字典。

#### Stop 15 · `control_context_budget`：budget controller（文档级）

- L1173–L1241：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L1173-L1241

```python
1191        demostrations_sort = self.get_rank_results(
1192            context, question, rank_method, condition_in_question, context_tokens_length,
1197        )
1201        target_token = eval("target_token" + context_budget)
1210        for idx, _ in demostrations_sort:
1213            target_token -= context_tokens_length[idx]
1214            if idx not in used:
1215                used.append(idx)
1216            if target_token < 0 or (
1217                force_context_number is not None and len(res) >= force_context_number
1218            ):
1219                break
1221        if reorder_context == "original":
1222            used = sorted(used)
```

- 先把文档排序（`get_rank_results`，L1818 起；`rank_method` 为 `"llmlingua"`/`"longllmlingua"` 时都走 `get_distance_longllmlingua`，L2032–L2045，用小模型算"文档 + 问题"的条件困惑度）。
- L1201：`context_budget="+100"` 是一个字符串，`eval` 后给预算加 100 个 token 的余量——因为后面 token 级还会再删。
- L1210–L1219：贪心地按排名装文档直到预算用完。
- L1221–L1227：恢复原顺序（或 LongLLMLingua 的 `two_stage` 重排，把重要文档放到首尾，缓解 "lost in the middle"）。

#### Stop 16 · `get_ppl` + `iterative_compress_prompt` + 阈值

- `get_ppl` L165–L210：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L165-L210

```python
189            response = self.model(
190                input_ids[:, past_length:end],
191                attention_mask=attention_mask[:, :end],
192                past_key_values=past_key_values,
193                use_cache=True,
194            )
197        shift_logits = response.logits[..., :-1, :].contiguous()
198        shift_labels = input_ids[..., past_length + 1 : end].contiguous()
203        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
204        loss = loss_fct(active_logits, active_labels)
209        res = loss.mean() if granularity == "sentence" else loss
```

- 标准 next-token 交叉熵，但 `reduction="none"`：`loss[i] = −log p(x_{i+1} | x_{≤i})`，即每个 token 的**自信息**。`granularity="token"` 时返回逐 token 向量。
- `past_key_values`：只对新段（`past_length:end`）做前向，前面已压缩好的部分走 KV cache。

- `iterative_compress_prompt` L1523–L1749：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L1523-L1749

```python
1536            iterative_ratios = self.get_dynamic_compression_ratio(
1537                context, target_token, iterative_size, dynamic_ratio, start
1538            )
1587        while end <= compressed_input_ids.shape[1]:
1636            loss, past_key_values = self.get_ppl(
1637                "", "token", compressed_input_ids, compressed_attention_mask,
1641                past_key_values=past_key_values, return_kv=True,
1643                end=end if idx else None,
1644            )
1657                past_key_values = [
1658                    [k[:, :, : end - iterative_size], v[:, :, : end - iterative_size]]
1659                    for k, v in past_key_values
1660                ]
1704            for delta_end, ratio in iterative_ratios[idx]:
1712                    threshold = self.get_estimate_threshold_base_distribution(
1713                        loss, ratio, False
1714                    )
1725                ) = self.get_compressed_input(
1726                    loss, compressed_input_ids, compressed_attention_mask,
1729                    end - iterative_size + delta_end,
1730                    iterative_size=delta_end,
1731                    threshold=threshold,
1743            end += iterative_size
```

- 把整个 context 切成长度 `iterative_size`（默认 200）的段，`get_dynamic_compression_ratio`（L1023–L1065）为每段算保留比例 `tau = target_token / 总长`（再叠加每个文档的动态偏移，`get_ratio` 把它夹到 [0,1]）。
- 每轮：对当前段算逐 token loss（L1636），然后在该段里按比例求阈值（L1712），删掉低于阈值的 token（L1725），推进 `end`。
- L1657–L1660：KV cache 只保留到"本段开始之前"——因为本段马上要被删 token，它的 KV 已失效；下一轮会以**压缩后的**本段为条件重新算。这正是 LLMLingua 相对 Selective Context 的改进：后面 token 的困惑度是在"已压缩的前文"条件下估计的，考虑了 token 之间的依赖。
- L1588–L1634：当长度超过模型最大位置数时，对 LM 自己的 KV cache 做"保留开头 `cache_bos_num` + 丢弃中间"——又是一个 StreamingLLM 式的 sink 技巧。

- 阈值 `get_estimate_threshold_base_distribution` L1508–L1521 与筛选 `get_compressed_input` L1402–L1506：

```python
1513        ppl = ppl[ppl != 10000]
1514        target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
1515        return (
1516            ppl.sort(descending=not condition_flag)
1517            .values[target_token]
...
1426            need_idx = torch.concat([loss > threshold, loss[:1] > 0])
1427        need_idx[end:] = 1
1428        need_idx[: end - iterative_size] = 1
```

- L1516：loss 降序排列，取第 `len·ratio` 个值作阈值 → 恰好 `ratio` 比例的 token loss 高于阈值。
- L1426：**保留 loss > 阈值的 token**（难预测 = 信息量大）。`loss[:1] > 0` 是在末尾补一位，因为 loss 比 token 少一个（第一个 token 没有预测）。
- L1427–L1428：只在当前段 `[end−iterative_size, end)` 内删除，其它位置强制保留。

#### Stop 17（扩展）· LLMLingua-2：token 分类

- `__get_context_prob` 中 L2178–L2183：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L2178-L2183
- `__compress` 中阈值 L2404–L2417：https://github.com/microsoft/LLMLingua/blob/5a4c78ae18ab17a98cf997e8259354e546081d64/llmlingua/prompt_compressor.py#L2404-L2417

```python
2180                probs = F.softmax(logits, dim=-1)
2183                    _probs = probs[j, :, 1]
...
2407                    threshold = np.percentile(
2408                        new_token_probs, int(100 * reduce_rate + 1)
2409                    )
2415                if word_prob > threshold or (
```

LLMLingua-2 不再用因果 LM 的困惑度，而是用一个双向编码器（XLM-RoBERTa 等）做**二分类**（label 1 = 保留），用 GPT-4 蒸馏出的标注训练。`probs[..., 1]` 就是"保留概率"，按百分位求阈值，保留概率高于阈值的词。双向模型能看到整句上下文，且更快。

---

## 4. 不同方法对比

| 方法 | 作用层级 | 选择依据 | 何时决定 | 需要注意力矩阵? | 省显存 | 省计算 | 需要训练? | 典型压缩率 |
|---|---|---|---|---|---|---|---|---|
| StreamingLLM | KV cache | 位置（sink+最近） | 解码中持续淘汰 | 否 | ✔ 恒定 | ✔ 解码 | 否 | 任意长度→固定窗口 |
| H2O (Observed) | KV cache | 累积注意力 | prefill（原文也在解码中持续） | 是（eager, O(T²)） | ✔ | ✔ 解码 | 否 | 50–80% |
| TOVA | KV cache | 最后 query 的注意力 | prefill/解码 | 一行 | ✔ | ✔ 解码 | 否 | 50–80% |
| SnapKV | KV cache | 观察窗口注意力+池化 | prefill 末 | `w` 行 | ✔ | ✔ 解码 | 否 | 50–90% |
| KNorm | KV cache | `−‖k‖` | prefill | 否 | ✔ | ✔ 解码 | 否 | ~50% |
| Expected Attention | KV cache | query 高斯模型的期望注意力 × ‖v‖ | prefill | 否 | ✔ | ✔ 解码 | 否 | 50–90% |
| AdaKV | KV cache（包装器） | 跨 head 统一排名 | prefill | 取决于内层 | kvpress 实现中✘（仅模拟） | ✘ | 否 | — |
| KV 量化 | KV cache 数值精度 | — | 写 cache 时 | 否 | ✔ 2–8× | 带宽 ✔ | 否 | 16→4/2 bit |
| FastV | 模型内 hidden states（LVLM 图像 token） | 第 K−1 层最后 token 的注意力 | 第 K 层 | 是（一层） | ✔ | ✔ prefill+解码 | 否 | 图像 token 50–75% |
| ToMe | 模型内 hidden states（ViT） | key 余弦相似度，合并而非删除 | 每层 | 否 | ✔ | ✔ | 否（训练可更好） | 每层 −r |
| LLMLingua | 文本 prompt | 小 LM 自信息 + 预算控制 | 推理前 | 否 | ✔ | ✔（含 API 费用） | 否（v2 需要） | 2–20× |

---

## 5. 常见困惑 / 坑

1. **"prefill 时压缩，会不会让 prefill 的输出变差？"** 不会。kvpress 的 hook 在 attention 返回**之后**才改 cache（Stop 2），当前这一层用的是完整 KV。只有后续解码 token 看到的是压缩后的 cache。
2. **"topk 后下标没排序，位置会乱吗？"** 不会。K 已经加过 RoPE，位置信息在向量里；softmax 加权和对 key 的排列是置换不变的。**但**如果你想做 StreamingLLM 式"按 cache 下标重编号位置"，就必须存未旋转的 K 或者做反旋转再旋转（kvpress 的 `KeyRerotationPress`）。
3. **两种位置策略相反**：StreamingLLM 把位置**压紧**（重编号，避免超出训练长度）；FastV 保留**原始位置**（`position_ids = keep_indexs`，因为它不会超长，保持一致更重要）。没有统一答案，取决于是否会超出训练长度。
4. **注意力矩阵哪来的？** FlashAttention / SDPA 不返回注意力权重。H2O（ObservedAttention）、FastV 需要 eager 注意力（显存 `O(T²)`）；SnapKV / TOVA 通过重算少量 query 绕过（Stop 4）；KNorm / Expected Attention / StreamingLLM 根本不需要。
5. **GQA 下的打分要按组聚合**：一个 KV head 服务 `n_rep` 个 query head，删 KV 是按 KV head 删的，所以分数必须先在组内平均（SnapKV L97–L98）或全 head 平均（TOVA L53）。
6. **query-aware vs query-agnostic**：SnapKV 论文里观察窗口包含用户问题，所以能"按问题挑上下文"；kvpress 的 pipeline 只压缩 context、问题在之后才来（Stop 8），这种设定对 SnapKV 不友好。读论文数字时要看清评测设定。
7. **H2O 的位置偏置**：因果掩码下早期 token 能被更多 query 看到，累积和天然大。kvpress 除以 `T−j`（Stop 5）。原版 H2O 在解码中持续累积，偏置更明显。
8. **Sink 必须保留**：StreamingLLM、Expected Attention 都显式保护前 4 个 token；直接用纯滑动窗口困惑度会爆炸。
9. **AdaKV 在 kvpress 里不省显存**：它通过"假 key"屏蔽（Stop 7），是研究用的模拟实现；真正省显存需要支持变长 head 的 paged KV kernel。
10. **ToMe 每层最多减半**，`r` 过大会被截断（`r = min(r, (t-protected)//2)`）；CLS token 必须保护。MAE 等模型评估时作者建议 `prop_attn=False`（`apply_patch` docstring L125–L126）。
11. **FastV 的 inplace 版本只支持 batch=1、不支持 KV cache**（L729 注释 "KV-Cache not supported"，L740 的 `[0]`）。
12. **LLMLingua 删的是"可预测"的 token**：对模型来说冗余，对人来说可能难读；压缩后的 prompt 语法往往是破碎的，但大模型仍能理解。数字、代码等关键 token 可用 `force_tokens` / `force_reserve_digit` 保护。
13. **压缩比定义不统一**：kvpress 的 `compression_ratio=0.7` 表示**删掉** 70%；LLMLingua 的 `rate=0.3` 表示**保留** 30%；ToMe 的 `r` 是每层减少的**个数**。

---

## 6. 自测题 + 动手练习

### 自测题

1. 写出 KV cache 显存公式，并计算 Qwen2-7B（28 层、4 个 KV head、head_dim 128、bf16）在 32K 上下文下单条序列的 KV cache 大小。
2. kvpress 的 `forward_hook` 用什么条件判断是否处于 prefill？解码阶段为什么不压缩？
3. SnapKV 为什么对分数做 `avg_pool1d`？如果去掉会出现什么现象？
4. ObservedAttentionPress 中 `scores / torch.arange(n_tokens, 0, -1)` 的作用是什么？
5. Expected Attention 中 `½kᵀΣk/d` 这一项来自哪里？为什么还要乘 `‖v‖`？
6. StreamingLLM 为什么要在 cache 中存**未加 RoPE** 的 key？代价是什么？
7. FastV 在 `idx == AGG_LAYER` 时用的 `layer_outputs[1]` 是哪一层的注意力？为什么 `AGG_LAYER == 0` 时只能随机丢弃？
8. ToMe 为什么用二分图而不是全图匹配？proportional attention 中 `+log(size)` 的数学含义是什么？
9. LLMLingua 中阈值是怎么从 `ratio` 得到的？保留的是 loss 高还是低的 token？为什么？
10. AdaKV 在 kvpress 里是如何"删除"不同 head 中不同数量的 token 的？

### 动手练习

1. **把 §2.1 的 SnapKV 改成 TOVA 和 H2O**：只改 `score` 的计算，比较三者在随机输入上选出的下标重合度（Jaccard）。
2. **测"sink 现象"**：加载任意小 LLM（如 Qwen2.5-0.5B，`attn_implementation="eager"`），对一段长文本输出 `output_attentions=True`，画出每层"所有 query 对前 4 个 token 的平均注意力"，验证 sink 存在。
3. **kvpress 实战**：用 `KVPressTextGenerationPipeline`（`kvpress/pipeline.py`）分别跑 `StreamingLLMPress`、`SnapKVPress`、`ExpectedAttentionPress`，`compression_ratio` 取 0.25/0.5/0.75，在同一篇长文 + 若干问题上比较回答质量。
4. **ToMe 守恒性检查**：用 §2.2 的 `bipartite_merge`，验证 `(x*size).sum(1)` 在合并前后不变；再用 `trace_source` 的思路（`tome/merge.py` `merge_source`，L227 起）记录每个合并 token 来自哪些原始 patch。
5. **玩具版 LLMLingua**：用 GPT-2 对一段文本算逐 token 自信息，保留自信息最高的 50% token，比较"一次性打分"与"分段迭代、以压缩前文为条件重新打分"两种方式的结果差异。
6. **KV 量化**：对一层真实 K/V 分别做 per-token 和 per-channel 的 4bit 量化，测量反量化误差，验证 "K 适合 per-channel、V 适合 per-token"。
