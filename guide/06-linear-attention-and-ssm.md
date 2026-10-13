# 06 · 线性注意力与状态空间模型（SSM）：源码导读

> 本章涉及的参考仓库（均为 `repos/05-linear-attention/` 下的浅 submodule）：
>
> | 仓库 | commit | 本章读什么 |
> |---|---|---|
> | fla-org/flash-linear-attention（下称 FLA） | `1fe52b5d05c53601885610fab3d39157647d836d` | `fla/ops/*/naive.py` 参考实现 → chunk / fused_recurrent Triton kernel → `fla/layers` → `fla/models` |
> | state-spaces/mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | `selective_scan_ref`、`Mamba`（mamba_simple）、`Mamba2`、`ssd_minimal` |
> | BlinkDL/RWKV-LM | `20be0f8cf0bdf506d8c0030c87a9a3416f95b143` | RWKV-4 的 WKV（RNN 形式 + CUDA kernel），RWKV-7 的 time-mix |
>
> 文中行号均按上述 commit 核对；永久链接格式 `https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<a>-L<b>`。

---

## 0. 要解决的问题 / 一句话直觉

**问题。** softmax 注意力的两个"二次方/线性增长"：

- **训练/prefill**：`QKᵀ` 是 `T×T`，计算量 `O(T²d)`；FlashAttention 省掉了 `T×T` 的显存，但 FLOPs 仍是二次。
- **解码**：每生成一个 token 都要读完整个 KV cache，**每步代价 `O(Td)`，显存 `O(T)` 随长度线性增长**（上一章的全部动机）。

**一句话直觉。** 把 softmax 里的 `exp(qᵀk)` 换成可分解的核 `φ(q)ᵀφ(k)`，利用矩阵乘法结合律 `(φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV)`，就可以先把所有历史 `kᵢvᵢᵀ` **累加成一个固定大小的矩阵 `S ∈ R^{d_k×d_v}`**。于是：

- 解码时 `S_t = S_{t−1} + k_t v_tᵀ`，`o_t = S_tᵀ q_t`——**一个 RNN**，每步 `O(d_k d_v)`，状态大小恒定，**不再需要 KV cache**；
- 训练时用 **chunkwise 并行形式**：块内用（掩码）矩阵乘、块间用状态传递，既能吃满 Tensor Core，又是 `O(T)`。

这条主线的演化就是不断改进"状态怎么更新"：

```
线性注意力   S_t = S_{t-1}                 + k_t v_tᵀ                      (只加不减，会"记满")
RetNet      S_t = γ   · S_{t-1}           + k_t v_tᵀ                      (固定标量衰减)
Mamba2/SSD  S_t = α_t · S_{t-1}           + k_t v_tᵀ                      (数据依赖的标量衰减)
GLA         S_t = diag(α_t) · S_{t-1}     + k_t v_tᵀ                      (数据依赖的逐通道衰减)
Mamba1      逐通道、逐状态维的对角衰减 exp(Δ·A)，没有 head 共享
DeltaNet    S_t = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ               (先擦除旧值再写入：delta rule)
Gated DeltaNet S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ        (衰减 + delta rule)
RWKV-7      S_t = S_{t-1}(diag(w_t) + a_t b_tᵀ) + v_t k_tᵀ                 (广义 delta rule)
```

---

## 1. 核心原理（带 shape）

### 1.1 记号

单个 head，`T` 序列长度，`d_k`（FLA 里叫 `K`）key 维度，`d_v`（`V`）value 维度。

- `Q, K ∈ R^{T×d_k}`，`V ∈ R^{T×d_v}`，`O ∈ R^{T×d_v}`。
- 状态 `S ∈ R^{d_k×d_v}`（FLA 默认布局 `[B, H, K, V]`；`state_v_first=True` 时转置为 `[B, H, V, K]`）。
- 带 batch/head：FLA 的 kernel 接口张量是 `[B, T, H, K]`（"head 在第三维"），而 naive 参考实现里大多先转成 `[B, H, T, K]`。读代码时一定先确认是哪种布局。

### 1.2 从 softmax 到线性注意力

因果 softmax 注意力：

```
o_t = Σ_{i≤t} exp(q_tᵀk_i/√d) v_i  /  Σ_{i≤t} exp(q_tᵀk_i/√d)
```

用特征映射 `φ: R^{d_k} → R^{d'}` 近似 `exp(qᵀk) ≈ φ(q)ᵀφ(k)`：

```
o_t = φ(q_t)ᵀ [Σ_{i≤t} φ(k_i) v_iᵀ]  /  φ(q_t)ᵀ [Σ_{i≤t} φ(k_i)]
         = φ(q_t)ᵀ S_t / φ(q_t)ᵀ z_t
S_t = S_{t−1} + φ(k_t) v_tᵀ   ∈ R^{d'×d_v}
z_t = z_{t−1} + φ(k_t)        ∈ R^{d'}
```

- **结合律**：并行形式 `O = (φ(Q)φ(K)ᵀ ⊙ M) V`（`M` 下三角掩码）是 `O(T²)`；不带掩码时 `φ(Q)(φ(K)ᵀV)` 是 `O(T d' d_v)`。**因果掩码破坏了直接的结合律**，所以才需要"递归形式"或"chunk 形式"。
- 现代做法（GLA、DeltaNet、Mamba2…）通常**去掉分母 `z`**，改用输出端的 RMSNorm/GroupNorm 保持数值尺度；`φ` 也常常就是恒等/SiLU + L2 归一化。FLA 的 `naive_recurrent_linear_attn` 保留了 `normalize` 选项以对照经典形式。

### 1.3 三种等价形式（以带标量衰减的版本为例）

令 `g_t = log α_t ≤ 0`（log 空间的衰减），`G_t = Σ_{j≤t} g_j`（前缀和）。

**递归形式（recurrent）**，`O(T d_k d_v)`，适合解码：

```
S_t = exp(g_t) · S_{t−1} + k_t v_tᵀ          S: [d_k, d_v]
o_t = S_tᵀ q_t                               o: [d_v]
```

**并行形式（parallel）**，`O(T² d)`，类似 attention：

```
D[t, s] = exp(G_t − G_s)  (s ≤ t),  0 (s > t)        D: [T, T]   衰减掩码
O = ((Q Kᵀ) ⊙ D) V                                    O: [T, d_v]
```

（展开递归即得：`o_t = Σ_{s≤t} (Π_{s<j≤t} α_j)(q_tᵀk_s) v_s`。）

**chunkwise 形式**，块长 `C`（通常 64），`T/C` 个块。对第 `n` 块，记块内局部前缀和 `Γ_i = Σ_{j∈块, j≤i} g_j`，`S_{[n]}` 为进入该块时的状态：

```
块内（intra）: O_intra = ((Q_n K_nᵀ) ⊙ D_local) V_n                [C, d_v]   一个 C×C 的小"attention"
块间（inter）: O_inter = (Q_n ⊙ exp(Γ)) S_{[n]}                      [C, d_v]   用状态回答"块之前的历史"
O_n = O_intra + O_inter
状态传递:    S_{[n+1]} = exp(Γ_C) · S_{[n]} + (K_n ⊙ exp(Γ_C − Γ))ᵀ V_n   [d_k, d_v]
```

- 总复杂度 `O(T C d + T d_k d_v)`，对 `T` 线性；所有操作都是 `C×d` 级矩阵乘，能吃满 Tensor Core。
- 这正是 FLA 中 `chunk_*` kernel 的结构：`chunk_fwd_h` 算每块的起始状态 `h`，`chunk_fwd_o` 算 `O_intra + O_inter`（见 §3 Stop 6、7）。

### 1.4 门控/衰减的变体

| 模型 | 衰减 `α_t` 的形状 | 参数化 |
|---|---|---|
| RetNet | 每 head 一个常数 `γ_h = 1 − 2^{−5−h}` | 不依赖输入 |
| Mamba2 / SSD | 每 head、每 token 一个标量 `exp(Δ_t A_h)` | `Δ_t = softplus(W x_t + b)`，`A_h = −exp(A_log)` |
| GLA | 每 token 一个 `d_k` 维向量 `α_t ∈ (0,1)^{d_k}` | `logsigmoid(W x_t)/16` |
| Mamba1 | 每个通道 `d`、每个状态维 `n` 各不相同：`exp(Δ_{t,d} A_{d,n})` | 同上，但 `Δ` 逐通道 |
| Gated DeltaNet | 每 head 一个标量（同 Mamba2）+ delta rule 的 `β_t` | `g = −exp(A_log)·softplus(a_proj(x)+dt_bias)`，`β = sigmoid(b_proj(x))` |

### 1.5 Delta rule

纯线性注意力只会"往状态里加"，`d_k` 维空间里最多存 `d_k` 个正交的 key，之后就开始互相干扰（"记满"）。Delta rule 把状态看作一个**联想记忆**，每步做一次在线回归：

```
预测:   v̂_t = S_{t−1}ᵀ k_t                               [d_v]
误差:   u_t = β_t (v_t − v̂_t)                            [d_v]   β_t ∈ (0,1) 是"学习率"
更新:   S_t = S_{t−1} + k_t u_tᵀ = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
```

等价于对损失 `½‖S_{t−1}ᵀk_t − v_t‖²` 做一步梯度下降。若 `‖k_t‖=1, β_t=1`，更新后 `S_tᵀk_t = v_t` **严格成立**——旧的 `k_t` 关联被替换掉了。所以 FLA 里 DeltaNet 系列总是对 q/k 做 L2 归一化（`use_qk_l2norm_in_kernel=True`）。

**Gated DeltaNet**：`S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`，FLA naive 实现中顺序是"先衰减、再预测/写入"。

**Delta rule 的 chunk 形式（WY 表示）**：块内 `u_t` 依赖于同块之前所有的 `u_s`：

```
u_t = β_t v_t − β_t S_0ᵀ k_t − β_t Σ_{s<t} (k_tᵀ k_s) u_s
⇒ (I + tril(diag(β) K Kᵀ, −1)) U = diag(β) V − diag(β) K S_0
⇒ U = T·(βV) − T·(βK) S_0 ≡ u − w S_0,     T = (I + tril(diag(β)KKᵀ, −1))⁻¹   [C×C]
O_n = Q S_0 + tril(Q Kᵀ) U,     S_{next} = S_0 + Kᵀ U
```

`T` 是 `C×C` 下三角矩阵的逆，用前代法（forward substitution）求解——FLA 里的 `solve_tril` / naive 代码里的那个 `for i in range(1, chunk_size)` 循环。

### 1.6 SSM 视角

Mamba 的离散 SSM（逐通道 `d`）：

```
h_t = exp(Δ_t A) ⊙ h_{t−1} + Δ_t B_t x_t        h: [d_inner, N]
y_t = C_tᵀ h_t + D x_t
```

把一个 head 内 `P` 个通道并起来、并令所有通道共享标量 `A`（Mamba2），就变成：

```
H_t = exp(Δ_t A) · H_{t−1} + (Δ_t x_t) B_tᵀ      H: [P, N]
y_t = H_t C_t
```

对照 §1.3：`q ↔ C`，`k ↔ B`，`v ↔ Δ·x`，`g ↔ Δ·A`，`d_k ↔ N (d_state)`，`d_v ↔ P (headdim)`。**Mamba2 = 带数据依赖标量衰减的线性注意力**——这就是 SSD（state space duality）的核心结论；`ssd_minimal.py` 的四步就是 §1.3 的 chunk 形式。

### 1.7 推理状态 vs KV cache

| | 每层缓存 | 随 T 增长? | 每步解码代价 |
|---|---|---|---|
| softmax attention | `K, V: 2 × H_kv × T × d` | 是 | `O(T d)` |
| 线性注意力 / GLA / DeltaNet | `S: H × d_k × d_v`（+ short conv 状态 `conv_size−1` 个 token） | 否 | `O(d_k d_v)` |
| Mamba1 | `ssm_state: d_inner × d_state` + `conv_state: d_inner × d_conv` | 否 | `O(d_inner · N)` |
| Mamba2 | `ssm_state: nheads × headdim × d_state` + `conv_state` | 否 | 同上 |
| RWKV-4 | 每通道 3 个标量 `aa, bb, pp` + 上一个 token 的 `x` | 否 | `O(C)` |

代价：状态是"有损压缩"的历史，精确召回（如 needle-in-a-haystack、复制长串）不如 softmax attention，所以工业界常用**混合架构**（少数几层 full/sliding attention + 多数线性层，FLA 的 `GatedDeltaNetBlock` 就支持按层混合，见 Stop 11）。

---

## 2. 最小实现

### 2.1 带衰减的线性注意力：parallel = recurrent = chunk

```python
import torch
import torch.nn.functional as F
torch.manual_seed(0)
B, H, T, K, V, C = 2, 3, 256, 16, 32, 64
q = torch.randn(B, H, T, K, dtype=torch.float64)
k = torch.randn(B, H, T, K, dtype=torch.float64) * K ** -0.5
v = torch.randn(B, H, T, V, dtype=torch.float64)
g = F.logsigmoid(torch.randn(B, H, T, dtype=torch.float64)) * 0.1  # log 衰减 ≤ 0；g=0 即纯线性注意力

def parallel(q, k, v, g):
    G = g.cumsum(-1)                                          # [B,H,T]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    D = torch.where(mask, G[..., :, None] - G[..., None, :], float("-inf")).exp()  # [B,H,T,T]
    return ((q @ k.transpose(-1, -2)) * D) @ v                 # [B,H,T,V]

def recurrent(q, k, v, g):
    S = q.new_zeros(B, H, K, V)                               # 恒定大小的状态
    o = q.new_empty(B, H, T, V)
    for t in range(T):
        S = S * g[..., t, None, None].exp() + k[..., t, :, None] * v[..., t, None, :]  # 衰减 + 外积写入
        o[..., t, :] = (q[..., t, :, None] * S).sum(-2)                                # S^T q
    return o, S

def chunk(q, k, v, g, C=64):
    S = q.new_zeros(B, H, K, V)
    o = q.new_empty(B, H, T, V)
    mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    for i in range(0, T, C):
        qi, ki, vi, gi = q[..., i:i+C, :], k[..., i:i+C, :], v[..., i:i+C, :], g[..., i:i+C]
        G = gi.cumsum(-1)                                     # [B,H,C] 块内局部前缀和
        D = torch.where(mask, G[..., :, None] - G[..., None, :], float("-inf")).exp()
        o_intra = ((qi @ ki.transpose(-1, -2)) * D) @ vi       # [B,H,C,V] 块内
        o_inter = (qi * G[..., :, None].exp()) @ S             # [B,H,C,V] 块间：用状态
        o[..., i:i+C, :] = o_intra + o_inter
        S = S * G[..., -1, None, None].exp() + \
            (ki * (G[..., -1:] - G).exp()[..., None]).transpose(-1, -2) @ vi   # 状态传给下一块
    return o, S

o_p = parallel(q, k, v, g)
o_r, S_r = recurrent(q, k, v, g)
o_c, S_c = chunk(q, k, v, g)
print((o_p - o_r).abs().max(), (o_p - o_c).abs().max(), (S_r - S_c).abs().max())  # 均 ~1e-14
```

（本文写作时用逐行等价的 NumPy 版本验证过三者误差在 `1e-14` 量级。）

### 2.2 Delta rule：递归形式 vs WY chunk 形式

```python
def delta_recurrent(q, k, v, beta):
    # q,k: [B,H,T,K]（k 已 L2 归一化）; v: [B,H,T,V]; beta: [B,H,T]
    S = q.new_zeros(B, H, K, V); o = q.new_empty(B, H, T, V)
    for t in range(T):
        kt = k[..., t, :]
        pred = (S * kt[..., :, None]).sum(-2)                 # S^T k_t     [B,H,V]
        u = beta[..., t, None] * (v[..., t, :] - pred)         # β(v − v̂)
        S = S + kt[..., :, None] * u[..., None, :]             # S += k u^T
        o[..., t, :] = (q[..., t, :, None] * S).sum(-2)
    return o, S

def delta_chunk(q, k, v, beta, C=16):
    S = q.new_zeros(B, H, K, V); o = q.new_empty(B, H, T, V)
    I = torch.eye(C, dtype=q.dtype)
    for i in range(0, T, C):
        qi, ki, vi, bi = q[..., i:i+C, :], k[..., i:i+C, :], v[..., i:i+C, :], beta[..., i:i+C]
        A = torch.tril(bi[..., :, None] * (ki @ ki.transpose(-1, -2)), -1)   # tril(diag(β)KK^T, -1)
        Tm = torch.linalg.inv(I + A)                                         # C×C 下三角求逆（kernel 中用前代法）
        u = Tm @ (bi[..., None] * vi)                                         # [B,H,C,V]
        w = Tm @ (bi[..., None] * ki)                                         # [B,H,C,K]
        ui = u - w @ S                                                        # 真正写入状态的 "new v"
        o[..., i:i+C, :] = qi @ S + torch.tril(qi @ ki.transpose(-1, -2)) @ ui
        S = S + ki.transpose(-1, -2) @ ui
    return o, S

kn = F.normalize(k, dim=-1)
beta = torch.rand(B, H, T, dtype=torch.float64)
o1, S1 = delta_recurrent(q, kn, v, beta)
o2, S2 = delta_chunk(q, kn, v, beta)
print((o1 - o2).abs().max(), (S1 - S2).abs().max())          # ~1e-15

# 联想记忆性质：β=1 时，最后写入的 (k, v) 能被精确读出
_, S = delta_recurrent(q, kn, v, torch.ones(B, H, T, dtype=torch.float64))
print(((S * kn[..., -1, :, None]).sum(-2) - v[..., -1, :]).abs().max())  # ~1e-15
```

### 2.3 状态 vs KV cache 的显存账

```python
def kv_cache_bytes(T, layers=32, kv_heads=8, d=128, bytes_=2):
    return 2 * layers * kv_heads * d * T * bytes_
def gdn_state_bytes(layers=32, heads=16, dk=128, dv=256, bytes_=4):   # 状态常用 fp32
    return layers * heads * dk * dv * bytes_
for T in [4096, 32768, 131072]:
    print(T, kv_cache_bytes(T) / 2**20, "MiB vs", gdn_state_bytes() / 2**20, "MiB")
# 4096: 512 MiB；131072: 16384 MiB；线性模型恒为 64 MiB
```

---

## 3. 源码导读（按阅读顺序）

**阅读顺序建议**：先读 FLA 的 `naive.py`（纯 PyTorch、几十行、就是公式）→ 再看对应的 chunk / fused_recurrent kernel 入口（看懂"它在算 naive 里的哪一行"即可，不必读懂所有 Triton 细节）→ `fla/layers`（投影、门控、conv、cache 怎么接上）→ `fla/models` → Mamba（SSM 视角）→ RWKV（另一条演化线）。

### 3.1 FLA 参考实现（naive）

#### Stop 1 · 线性注意力：`naive_recurrent_linear_attn` / `naive_chunk_linear_attn`

- 文件：`fla/ops/linear_attn/naive.py`，L12–L44、L47–L76
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/linear_attn/naive.py#L12-L76

```python
 24    B, T, H, K, V = *q.shape, v.shape[-1]
 32    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
 35    for t in range(T):
 36        S = S + torch.einsum('b h k, b h v -> b h k v', k[:, t], v[:, t])
 37        o[:, t] = torch.einsum('b h k v, b h k -> b h v', S, q[:, t] * scale)
 38    if normalize:
 39        k_cum = k.cumsum(1) if z_init is None else k.cumsum(1) + z_init
 40        o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
```

- 输入布局 `[B, T, H, K]`（L24），状态 `S: [B, H, K, V]`，**fp32 累加**（L29、L32）——状态是长期累加量，bf16 会严重丢精度。
- L36：`k_t v_tᵀ` 外积写入；L37：`o_t = Sᵀ q_t`。
- L38–L40：可选的分母 `φ(q)ᵀΣφ(k)`，即 §1.2 的 `z_t`。

```python
 57    q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
 60    kv = k.transpose(-1, -2) @ v
 61    kv = kv.cumsum(2)
 62    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
 63    inter = q @ kv
 64    intra = ((
 65        q @ k.transpose(-1, -2)).masked_fill_(
 66        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
 67        0,
 68    )) @ v
 69    o = inter + intra
```

- L57：切块，`[B, H, N, C, d]`，`N = T/C`。
- L60：每块的 `K_nᵀV_n: [B, H, N, K, V]`；L61 沿块维前缀和；L62 右移一位——第 `n` 块需要的是**之前所有块**的状态（exclusive cumsum）。没有衰减时，"状态传递"就退化成一个 cumsum。
- L63：块间项 `Q_n S_{[n]}`；L64–L68：块内项 `(Q_nK_nᵀ ⊙ 下三角) V_n`（`triu(diagonal=1)` 置零的是严格上三角，对角线保留）。

#### Stop 2 · RetNet：`naive_retention`

- 文件：`fla/ops/retention/naive.py` L11–L20
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/retention/naive.py#L11-L20

```python
 15    s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
 16    n = q.new_tensor(range(seq_len), dtype=torch.float)
 17    n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
 18    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
```

- L15：每个 head 的衰减 `γ_h = 1 − 2^{−5−h}`，取 `log2`。不同 head 衰减速度不同 = 多尺度（multi-scale retention）。
- L17：`D[t, s] = γ^{t−s}·[t ≥ s]`（`exp2((t−s)·log2γ)`），就是 §1.3 并行形式里的衰减掩码，形状 `[H, T, T]`。
- L18：`(QKᵀ) ⊙ D`。注意这里输入布局是 `[B, H, T, d]`（L14）。
- 高效版 `chunk_retention`（`fla/ops/retention/chunk.py` L68–L79）直接把 `g_gamma = log(γ_h)` 交给 `chunk_simple_gla`——**RetNet 就是常数门控的 Simple GLA**。

#### Stop 3 · Simple GLA（标量门控）：三种形式在同一个文件

- 文件：`fla/ops/simple_gla/naive.py`：`naive_chunk_simple_gla` L13–L58、`naive_recurrent_simple_gla` L61–L94、`naive_parallel_simple_gla` L97–L117
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/simple_gla/naive.py#L13-L117

递归形式：

```python
 83    for i in range(T):
 84        gate = g[:, :, i].exp()
 87        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
 88        S = S * gate.unsqueeze(-1).unsqueeze(-1) + kv
 90        o_i = (q_i.unsqueeze(-1) * S).sum(-2)
```

`g: [B, H, T]` 是 log 空间的门（每 head 一个标量），L84 `exp` 回到 `(0,1]`。

并行形式：

```python
108    A = (q @ k.transpose(-1, -2) * scale)
110        g = g.cumsum(-1)
111        D = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
112        A = A * D
```

- L111 的两次 `tril`：第一次把上三角的 `G_t − G_s` 置 0（否则 `exp` 可能溢出成 inf，inf×0=NaN），第二次把 `exp(0)=1` 的上三角再置 0。我们 §2.1 里用 `where(mask, …, -inf)` 达到同样效果。

chunk 形式：

```python
 41    decay = decay.squeeze(-1).cumsum(-1)
 42    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
 47    for i in range(0, T1 // chunk_size):
 49        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
 50        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
 51        o[:, :, i] = o_inter + attn @ v_i
 52        S = S * decay[:, :, i, -1, None, None].exp() + \
 53            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
```

- L41：`decay: [B, H, N, C]`，**块内**前缀和（`cumsum(-1)` 在块维 `C` 上）。
- L49：块内 `(QKᵀ) ⊙ D_local`；L50：块间 `(Q ⊙ exp(Γ)) S`——第 `i` 个 query 距离块起点衰减了 `exp(Γ_i)`。
- L52–L53：状态传递。旧状态整体衰减到块末 `exp(Γ_C)`；块内第 `j` 个 key 只需衰减 `exp(Γ_C − Γ_j)` 就到块末。这与 §2.1 的 `chunk` 逐行对应。
- L28–L35：长度不是 `C` 的整数倍时先 pad，最后 L57 截掉。

#### Stop 4 · GLA（逐通道门控）：`naive_recurrent_gla`

- 文件：`fla/ops/gla/naive.py` L15–L44
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gla/naive.py#L15-L44

```python
 24    q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), (q, k, v, gk))
 33    for i in range(T):
 37        gk_i = gk[:, :, i].exp()
 38        kv_i = k_i[..., None] * v_i[..., None, :]
 39        h = h * gk_i[..., None] + kv_i
 40        o[:, :, i] = (q_i[..., None] * h).sum(-2)
```

- 与 Simple GLA 唯一区别：`gk: [B, H, T, K]` 是**向量**门，L39 `gk_i[..., None]` 广播到 `[B, H, K, 1]`，即 `h = diag(α_t) h + k vᵀ`：状态的每一**行**（每个 key 维度）有自己的遗忘速度。表达力更强，但 chunk 形式里块内不能再用一个 `T×T` 衰减矩阵，要把衰减"分摊"到 q 和 k 上（`q ⊙ exp(Γ)`、`k ⊙ exp(−Γ)`），数值上需要二级分块（`chunk_gla_fwd_A_kernel_intra_sub_*` 那些 kernel 的由来）。

#### Stop 5 · Delta rule 与 Gated Delta rule

- `fla/ops/delta_rule/naive.py`：`delta_rule_recurrence` L12–L37、`delta_rule_chunkwise` L40–L70
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/delta_rule/naive.py#L12-L70

```python
 27    for i in range(l):
 32        _v = _v - (S.clone() * _k[..., None]).sum(-2)
 33        _v = _v * beta_i
 34        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
 35        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
```

- L32：`v̂ = Sᵀk`（`S: [b,h,d_k,d_v]`，沿 `d_k` 求和），`_v ← v − v̂`；L33 乘 `β`；L34 写入。正是 §1.5 的三行公式。`.clone()` 只是为了 autograd 不报原地修改的错。

```python
 44    v = v * beta[..., None]
 45    k_beta = k * beta[..., None]
 50    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
 52    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
 53    for i in range(1, chunk_size):
 54        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
 55    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
 57    u = attn @ v
 58    w = attn @ k_beta
 62    for i in range(0, l // chunk_size):
 64        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
 65        u_i = u[:, :, i] - w[:, :, i] @ S
 66        o_inter = q_i @ S
 67        o[:, :, i] = o_inter + attn @ u_i
 68        S = S + k_i.transpose(-1, -2) @ u_i
```

- L52：`−tril(diag(β)KKᵀ, −1)`（`triu(diagonal=0)` 掩掉对角线及以上）。
- L53–L55：逐行前代法，把 `L = −tril(...)` 变成 `(I − L)⁻¹ = (I + tril(diag(β)KKᵀ,−1))⁻¹`，即 §1.5 的 `T`。
- L57–L58：WY 表示 `u = T(βV)`、`w = T(βK)`，与状态无关，**所有块可以并行算**。
- L65：`U = u − w S_0`，只有这一步依赖上一块的状态；L67 输出；L68 状态更新。
- 注意 L61 重新定义 `mask` 为 `diagonal=1`，块内 attention 包含对角线。

- `fla/ops/gated_delta_rule/naive.py`：`naive_recurrent_gated_delta_rule` L13–L64、`naive_chunk_gated_delta_rule` L67–L161
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gated_delta_rule/naive.py#L13-L161

```python
 50    for i in range(T):
 54        h = h.clone() * g[:, :, i].exp()[..., None, None]
 56        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
 57        b_v = b_v * b_beta[..., None]
 58        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
 59        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
```

- 只比 delta rule 多 L54：**先衰减**再做 delta 更新。docstring（L26–L38）写清了 shape：`q,k: [B,T,H,K]`、`v: [B,T,H,V]`、`beta, g: [B,T,H]`、`initial_state: [B,H,K,V]`。
- chunk 版（L127–L153）把 Simple GLA 的 `L_mask` 和 delta rule 的 WY 合并：L130 `attn = −((k_beta @ kᵀ) * L_mask)`，L136 `k_cumdecay = attn @ (k_beta * decay_exp)`，L148–L149 `v_new = v_i − k_cumdecay @ S`，L152 状态衰减 + 写入。

### 3.2 FLA 高效 kernel：chunk 与 fused_recurrent

#### Stop 6 · `chunk_simple_gla_fwd` → `chunk_fwd_h` → `chunk_fwd_kernel_h`

- 入口：`fla/ops/simple_gla/chunk.py` `chunk_simple_gla_fwd` L17–L58：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/simple_gla/chunk.py#L17-L58
- kernel：`fla/ops/common/chunk_h.py` `chunk_fwd_kernel_h`（L37 起），主循环 L96–L153：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/common/chunk_h.py#L96-L153

`chunk_simple_gla_fwd` 只有两步：`h, ht = chunk_fwd_h(k, v, g, …)` 算每块起始状态，`o = chunk_fwd_o(q, k, v, g, h, …)` 算输出——就是 §1.3 的两行。

```python
 96    for i_t in range(NT):
112        if i_t_int64 % NTS == 0:
113            tl.store(p_h, (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty), mask=m_h)
115        b_k = tl.load(p_k, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0)     # [BK, BT]
117        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)     # [BT, BV]
121        if USE_G:
122            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
124            b_g = tl.load(p_g, mask=(i_t_int64 * BT + tl.arange(0, BT) < T), other=0.)
125            b_h *= exp2(b_g_last)
126            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)
134        if USE_GK:
140            b_h *= exp2(b_gk_last)[:, None]
141            b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)
153        b_h = tl.dot(b_k, b_v, b_h)
```

- 一个 Triton program 负责一个 `(batch, head, K 块, V 块)`，**顺序**遍历所有 chunk（L96）；`b_h: [BK, BV]` 在寄存器/SRAM 里常驻，fp32 累加。
- L112–L113：在处理第 `i_t` 块**之前**把当前状态写到 `h[i_t]`——这就是第 `i_t` 块的 `S_{[n]}`，供 `chunk_fwd_o` 使用。
- L125–L126（标量门 `USE_G`）：`h ← exp(Γ_C)·h`，`v_j ← exp(Γ_C − Γ_j)·v_j`，对应 naive L52–L53（这里把衰减乘在 v 上，等价于乘在 k 上）。
- L140–L141（GLA 的向量门 `USE_GK`）：衰减按 key 维度广播，所以乘在 `k` 上、按行缩放 `h`。**同一个 kernel 通过 constexpr 开关同时服务 Simple GLA / RetNet（`USE_G_GAMMA`）/ GLA**。
- L153：`h += K_nᵀ V_n`，一次 `[BK,BT]×[BT,BV]` 的 `tl.dot`（Tensor Core）。
- **为什么用 `exp2`？** GPU 上 `exp2` 是原生指令。所以上游先把 `g` 的块内前缀和乘上 `RCP_LN2 = 1/ln2`（`fla/ops/utils/constant.py` L10），见 `chunk_gla_fwd` L1222–L1228 的 `chunk_local_cumsum(g, chunk_size, scale=RCP_LN2)`。naive 版用自然对数 + `exp`，kernel 版用 `log2` + `exp2`，两者等价。

#### Stop 7 · `chunk_fwd_kernel_o`：块内 + 块间

- `fla/ops/common/chunk_o.py` `chunk_fwd_kernel_o`（L55 起），核心 L100–L152
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/common/chunk_o.py#L100-L152

```python
100    b_o = tl.zeros([BT, BV], dtype=tl.float32)
101    b_A = tl.zeros([BT, BT], dtype=tl.float32)
106    for i_k in range(tl.cdiv(K, BK)):
127            b_o = tl.dot(b_q, b_h, b_o)          # [BT, BK] @ [BK, BV]  块间：Q S
129        b_A = tl.dot(b_q, b_k, b_A)              # [BT, BK] @ [BK, BT]  块内：Q K^T
131    if USE_G:
134        b_g = tl.load(p_g, mask=m_t, other=0.0)
135        b_o = b_o * exp2(b_g)[:, None]
136        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
142    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
143    b_A = tl.where(m_A, b_A, 0)
151    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
```

- 每个 program 负责一个 chunk（`i_t`），**chunk 之间完全并行**——因为状态 `h` 已经由 Stop 6 预先算好。
- L127 / L129：沿 K 维分块累加 `Q S` 和 `Q Kᵀ`。
- L135–L136：块间乘 `exp(Γ_i)`，块内乘 `exp(Γ_i − Γ_j)`；L142–L143 因果掩码。
- L151：`O = scale·(Q S ⊙ decay) + scale·(A V)`。与 naive L49–L51 一一对应。

#### Stop 8 · `chunk_gated_delta_rule_fwd`：WY + 状态递推 + 输出

- `fla/ops/gated_delta_rule/chunk.py` `chunk_gated_delta_rule_fwd` L33–L123；公开 API `chunk_gated_delta_rule` L397 起
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gated_delta_rule/chunk.py#L33-L123

```python
 52    if use_gate_in_kernel:
 53        g = gdn_gate_chunk_cumsum(g=g, A_log=A_log, chunk_size=chunk_size, scale=RCP_LN2, dt_bias=dt_bias, ...)
 62    else:
 63        g = chunk_local_cumsum(g, chunk_size=chunk_size, scale=RCP_LN2, ...)
 70    # obtain WY representation. u is actually the new v.
 71    # fused kkt + solve_tril + recompute_w_u
 72    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, ...)
 95    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state, ...)
111    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, ...)
```

三步与 naive chunk 版完全对应：

1. L72 `chunk_gated_delta_rule_fwd_intra`（`chunk_fwd.py` L332 起）：块内算 `KKᵀ`、对下三角矩阵求逆（`solve_tril`）、得到 `w, u`——naive L130–L136。所有块并行。
2. L95 `chunk_gated_delta_rule_fwd_h`：顺序扫描所有块，`v_new = u − w S`、`S ← 衰减·S + Kᵀ v_new`——naive L148–L153。返回每块起始状态 `h` 和 `v_new`。
3. L111 **复用** Stop 7 的 `chunk_fwd_o`，只是把 `v` 换成 `v_new`——这说明 delta rule 的输出计算和普通门控线性注意力是同一个 kernel。

L52–L61：当 `use_gate_in_kernel=True` 时，门的参数化 `g = −exp(A_log)·softplus(a + dt_bias)`（Mamba2 风格）和块内 cumsum 融合在一个 kernel 里。

#### Stop 9 · `fused_recurrent_gated_delta_rule_fwd_kernel`：解码用的逐 token kernel

- `fla/ops/gated_delta_rule/fused_recurrent.py` L30 起，时间循环 L114–L162
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gated_delta_rule/fused_recurrent.py#L114-L162

```python
114    for _ in tl.range(0, T):
118        if USE_QK_L2NORM_IN_KERNEL:
119            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
120            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
126        if APPLY_BETA_SIGMOID:
127            b_beta = tl.sigmoid(b_beta)
128        if ALLOW_NEG_EIGVAL:
129            b_beta = b_beta * 2
131        if USE_G:
133            if USE_GATE_IN_KERNEL:
137                b_g = -exp(b_A) * softplus(b_g)
138            b_h *= exp(b_g)
158        else:
159            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
160            b_h += b_k[:, None] * b_v
161            b_o = tl.sum(b_h * b_q[:, None], 0)
162        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
```

- 和 naive 递归版逐行对应（L138 衰减、L159 预测误差、L160 写入、L161 读出），但状态 `b_h` 全程留在片上，只在最后写回（L176–L181）。
- L128–L129 `ALLOW_NEG_EIGVAL`：`β ∈ (0, 2)`，使转移矩阵 `I − βkkᵀ` 的特征值可以为负（`1−β ∈ (−1,1)`），理论上能提升状态追踪能力。
- 这种 kernel 并行度只有 `B×H×(K/BK)×(V/BV)`，长序列训练很慢，但 `T` 很小（解码）时没有 chunk 的额外开销——见 Stop 10 的模式选择。

### 3.3 FLA 层与模型

#### Stop 10 · `GatedDeltaNet` 层

- 文件：`fla/layers/gated_deltanet.py`，`__init__` L91–L198，`forward` L210–L357
- 链接：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/layers/gated_deltanet.py#L91-L357

参数（`__init__`）：

```python
148        self.q_proj = nn.Linear(in_features=hidden_size, out_features=self.key_dim, bias=False)
149        self.k_proj = nn.Linear(in_features=hidden_size, out_features=self.key_dim, bias=False)
150        self.v_proj = nn.Linear(in_features=hidden_size, out_features=self.value_dim, bias=False)
151        self.a_proj = nn.Linear(in_features=hidden_size, out_features=self.num_v_heads, bias=False)
152        self.b_proj = nn.Linear(in_features=hidden_size, out_features=self.num_v_heads, bias=False)
154        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
155        self.A_log = nn.Parameter(torch.log(A))
163        inv_dt = dt + torch.log(-torch.expm1(-dt))
164        self.dt_bias = nn.Parameter(inv_dt)
170            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
194            self.g_proj = nn.Linear(in_features=hidden_size, out_features=self.value_dim, bias=False)
195            self.o_norm = FusedRMSNormGated(hidden_size=self.head_v_dim, eps=norm_eps, activation=output_gate_activation)
```

- `a_proj`、`b_proj` 每个 head 输出**一个标量**：分别生成衰减 `g` 和写入强度 `β`。
- L154–L164：`A_log`、`dt_bias` 的初始化照搬 Mamba2（`dt` 在 `[0.001, 0.1]` 对数均匀采样，`inv_dt` 是 softplus 的反函数，使 `softplus(dt_bias) = dt`）。
- L168–L187：q/k/v 各接一个长度 4 的**因果深度卷积**（ShortConvolution）。L189–L192 的警告说明它对性能"crucial"：它给模型提供了廉价的局部 token 混合（类似 n-gram），线性注意力本身不擅长这个。
- L193–L195：输出门 `o = RMSNorm(o) ⊙ swish(g_proj(x))`，替代了经典线性注意力的分母归一化。
- `head_v_dim = head_dim × expand_v`（L126，默认 2）：状态 `[d_k, d_v]` 中 `d_v` 更大，增加记忆容量。

前向：

```python
227        if torch.is_grad_enabled():
228            mode = 'chunk'
229        elif q_len <= 64 and not self.training:
230            mode = 'fused_recurrent'
236        last_state = get_layer_cache(module=self, past_key_values=past_key_values)
...
298        beta = self.b_proj(hidden_states)
300        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
301        if mode == 'chunk':
302            o, recurrent_state = chunk_gated_delta_rule(
303                q=q, k=k, v=v,
306                g=self.a_proj(hidden_states),
307                beta=beta,
308                initial_state=recurrent_state,
309                output_final_state=use_cache,
310                use_qk_l2norm_in_kernel=True,
311                use_beta_sigmoid_in_kernel=True,
315                use_gate_in_kernel=True,
316                A_log=self.A_log,
317                dt_bias=self.dt_bias,
318            )
340        update_layer_cache(
341            module=self,
342            past_key_values=past_key_values,
343            recurrent_state=recurrent_state,
344            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
345            offset=q_len,
346        )
```

- L227–L232：**训练 / 长 prefill 用 chunk，短输入（≤64，解码）用 fused_recurrent**——两个 kernel 数学等价（§2.1 证明），只是在不同 T 下谁更快。
- L300–L309：上一步的状态作为 `initial_state` 传入，`output_final_state=use_cache` 返回新状态。**这就是"状态替代 KV cache"在代码里的全部样子**：prefill 一次把整个 prompt 压进 `recurrent_state: [B, H, d_v, d_k]`（`state_v_first=True`），之后每个 token 只更新它。
- L310–L311、L315–L317：L2 归一化、`β = sigmoid(·)`、门的 Mamba2 式参数化，全部融合进 kernel，省掉中间张量的显存读写。
- L340–L346：把状态和 conv 缓存写回 cache 对象。

缓存对象：`fla/models/utils.py` `FLALayer.update` L42–L71（https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/models/utils.py#L30-L71）。L62–L71 显示每层的 cache 是一个字典 `{"recurrent_state", "attn_state", "conv_state", "ffn_state", "A_state"}`，`recurrent_state` 直接**覆盖**（L70–L71），而不是像 KV cache 那样 `cat` 增长。`attn_state` 用于混合架构中的 softmax 注意力层（L77–L80 还支持 sliding window 截断）。

#### Stop 11 · GLA 层的门、以及模型如何组装

- `fla/layers/gla.py` L228–L243：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/layers/gla.py#L228-L243

```python
231        gk = self.gk_proj(hidden_states)
238            k, gk = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (k, gk))
241        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
```

`gk` 和 `k` 同形状 `[B, T, H, d_k]`（逐通道门）；`logsigmoid(x) ∈ (−∞, 0)` 直接给出 log 空间衰减，再除以 16（`gate_logit_normalizer`）让初始衰减接近 1（记得久），避免训练初期遗忘过快。

- `fla/models/gated_deltanet/modeling_gated_deltanet.py` `GatedDeltaNetBlock` L43–L87：https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/models/gated_deltanet/modeling_gated_deltanet.py#L43-L87

```python
 52        attn_spec = get_hybrid_attention_spec(attn=config.attn, layer_idx=layer_idx)
 53        if attn_spec is not None:
 54            self.attn = Attention(
...
 64        else:
 65            self.attn = GatedDeltaNet(
```

标准 pre-norm Transformer block，只是把 attention 换成了 `GatedDeltaNet`；`config.attn` 指定的层仍用 softmax `Attention`（带 `window_size` 选项），这就是混合架构。`fla/models/` 下每个模型目录（`gla/`、`retnet/`、`mamba2/`、`rwkv7/`…）都是同样的套路，配合 HF 的 `AutoModelForCausalLM` 注册。

### 3.4 Mamba：SSM 视角

#### Stop 12 · `selective_scan_ref`：Mamba1 的数学定义

- `mamba_ssm/ops/selective_scan_interface.py` L127–L193
- 链接：https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py#L127-L193

```python
145    if delta_bias is not None:
146        delta = delta + delta_bias[..., None].float()
147    if delta_softplus:
148        delta = F.softplus(delta)
149    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
160    x = A.new_zeros((batch, dim, dstate))
162    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
166        if B.dim() == 3:
167            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
174    for i in range(u.shape[2]):
175        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
180                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
188    y = torch.stack(ys, dim=2) # (batch dim L)
189    out = y if D is None else y + u * rearrange(D, "d -> d 1")
190    if z is not None:
191        out = out * F.silu(z)
```

- shape（docstring L130–L140）：`u, delta: [B, D, L]`（注意是**通道在前、长度在后**），`A: [D, N]`，`B, C: [B, N, L]`（输入依赖 = "selective"），状态 `x: [B, D, N]`。
- L147–L148：`Δ = softplus(Δ + bias) > 0`，是"步长"：`Δ` 大 → `exp(ΔA)` 小 → 遗忘旧状态、重视当前输入；`Δ` 小 → 保持状态、忽略当前输入。这就是 selection 机制。
- L162：零阶保持（ZOH）离散化的 `Ā = exp(ΔA)`，`[B, D, L, N]`——每个通道 `d`、每个状态维 `n` 都有自己的衰减。
- L167：`B̄u ≈ Δ·B·u`（简化的离散化）。
- L175：**这一行就是整个 SSM**：`x_t = Ā_t ⊙ x_{t−1} + B̄_t u_t`。与 GLA 递归 `h = h * gk + k vᵀ` 对比：`Ā ↔ α`，`B ↔ k`，`Δu ↔ v`；区别是 Mamba1 中每个通道 `d` 单独一个 `N` 维状态（相当于 `d_v = 1` 的 head 有 `D` 个），而注意力式模型是一个 head 共享 `d_k×d_v` 的矩阵状态。
- L180：`y_t = C_tᵀ x_t`（`C ↔ q`）；L189 跳跃连接 `D`；L191 门控 `silu(z)`。
- 显式构造 `deltaA: [B, D, L, N]` 显存很大，所以真正的 CUDA kernel（`selective_scan_fn`）在 SRAM 里边算边扫描，从不物化它。

#### Stop 13 · `Mamba`（mamba_simple）：训练路径与解码 `step`

- `mamba_ssm/modules/mamba_simple.py`：`__init__` L32–L117、`forward` L119–L206、`step` L208–L253、`allocate_inference_cache` L255–L266
- 链接：https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py#L31-L266

```python
 62        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
 77        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
 80        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
104        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
109        A_log = torch.log(A)  # Keep A_log in fp32
```

- `in_proj` 输出 `x` 和门 `z` 两路；`x_proj` 从 `x` 生成 `[Δ(低秩 dt_rank), B(N), C(N)]`；`dt_proj` 把低秩 `Δ` 升回 `d_inner` 维。
- L104–L109：S4D-real 初始化 `A[d, n] = −(n+1)`（L143 `A = −exp(A_log)` 保证 A<0，即衰减 <1，稳定）。

```python
182            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
183            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
184            dt = self.dt_proj.weight @ dt.t()
189            y = selective_scan_fn(
190                x, dt, A, B, C, self.D.float(), z=z,
197                delta_bias=self.dt_proj.bias.float(),
198                delta_softplus=True,
199                return_last_state=ssm_state is not None,
200            )
201            if ssm_state is not None:
202                y, last_state = y
203                ssm_state.copy_(last_state)
```

- prefill 时 `return_last_state=True`，把扫描结束时的状态写进 `ssm_state`——之后解码就从这个状态继续。L167 同理把最后 `d_conv` 个输入存进 `conv_state`。

```python
211        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
216            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
217            conv_state[:, :, -1] = x
218            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
240            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
241            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
242            dB = torch.einsum("bd,bn->bdn", dt, B)
243            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
244            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
245            y = y + self.D.to(dtype) * x
246            y = y * self.act(z)  # (B D)
```

- `step` 是 `selective_scan_ref` 循环体的单步版：L216–L218 卷积用一个长度 `d_conv` 的环形缓冲；L243 状态更新、L244 读出——**每步只碰 `[B, d_inner, d_state]` 的状态，与已生成长度无关**。
- `allocate_inference_cache`（L255–L266）：`conv_state: [B, d_inner, d_conv]`、`ssm_state: [B, d_inner, d_state]`，这就是 Mamba 的全部"KV cache"。

#### Stop 14 · `ssd_minimal_discrete`：Mamba2 的 chunk 算法（四步）

- `mamba_ssm/modules/ssd_minimal.py`：`segsum` L23–L32、`ssd_minimal_discrete` L34–L78、测试 L82–L103
- 链接：https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/ssd_minimal.py#L23-L103

```python
 23def segsum(x):
 26    x = repeat(x, "... d -> ... d e", e=T)
 27    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
 28    x = x.masked_fill(~mask, 0)
 29    x_segsum = torch.cumsum(x, dim=-2)
 31    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
```

- `segsum(x)[t, s] = Σ_{s<j≤t} x_j`（`s ≤ t`），否则 `−inf`。`exp(segsum(A))` 就是 §1.3 的衰减掩码 `D`。为什么不直接 `cumsum[t] − cumsum[s]`（`segsum_unstable`，L14–L21）？两个大数相减会有精度损失；这里先掩码再累加，只对需要的那段求和。

```python
 48    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
 50    A = rearrange(A, "b c l h -> b h c l")
 51    A_cumsum = torch.cumsum(A, dim=-1)
 54    L = torch.exp(segsum(A))
 55    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
 59    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
 60    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
 66    states = torch.cat([initial_states, states], dim=1)
 67    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
 68    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
 69    states, final_state = new_states[:, :-1], new_states[:, -1]
 73    state_decay_out = torch.exp(A_cumsum)
 74    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
```

shape：`X: [b, c(块数), l(块长), h, p]`，`A: [b, c, l, h]`，`B, C: [b, c, l, h, n]`。对照 §1.3（`q=C, k=B, v=X, g=A`）：

1. **L54–L55 块内（对角块）**：`Y_diag = ((C Bᵀ) ⊙ L) X`——与 Simple GLA naive L49 的 `attn @ v_i` 相同。
2. **L59–L60 每块对状态的贡献**：`states[c] = Σ_l exp(Γ_last − Γ_l) X_l B_lᵀ`，`[b, c, h, p, n]`（注意 Mamba 的状态是 `[p, n] = [d_v, d_k]`，与 FLA 默认的 `[K, V]` 互为转置）。
3. **L66–L69 块间递推**：用块级 `segsum` 一次性算出"第 `z` 块起始状态 = 之前所有块贡献的衰减加权和"。这里用一个 `(c+1)×(c+1)` 的矩阵乘代替了顺序循环（块数少时很划算）；FLA 的 `chunk_fwd_h` 则是顺序扫描。
4. **L73–L74 状态到输出（非对角块）**：`Y_off = (C ⊙ exp(Γ)) S`——与 Simple GLA naive L50 的 `o_inter` 相同。

L102–L103 的测试调用 `ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)`：**输入要先乘 `Δ`**（`v = Δ·x`，`g = Δ·A`），与 §1.6 的对应关系一致。

#### Stop 15 · `Mamba2.step`：标量衰减的单步更新

- `mamba_ssm/modules/mamba2.py`：`A_log` 初始化 L132–L136、`step` L278–L343、`allocate_inference_cache` L345–L355
- 链接：https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2.py#L278-L355

```python
313            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
314            dA = torch.exp(dt * A)  # (batch, nheads)
315            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
316            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
317            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
318            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
...
352        ssm_state = torch.zeros(
353            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
354        )
```

- 与 Mamba1 `step` 对比：`A: [nheads]` 是**每 head 一个标量**（L133 `A` 形状 `[nheads]`），`dA: [B, nheads]`；状态 `[B, nheads, headdim, d_state]` 是矩阵状态。L317 就是 `S_t = α_t S_{t−1} + (Δ x_t) B_tᵀ`，L318 `y = S C`。和 FLA `naive_recurrent_simple_gla` L88/L90 是同一件事（转置布局）。
- 一个 head 内所有通道共享衰减，才使得 §1.3 的 `T×T` 衰减掩码对整个 head 只需算一次，从而可以用矩阵乘（Tensor Core）——这就是 Mamba2 比 Mamba1 训练快得多的根本原因。

### 3.5 RWKV：另一条演化线

#### Stop 16 · RWKV-4 的 WKV：带"bonus"的指数衰减加权平均

- `RWKV-v4/src/model_run.py`：`RWKV_TimeMix` L115–L148、`WKV.forward` L29–L56、权重加载时 `time_decay` 变换 L256–L257、`RWKV_RNN.SA` L318–L354
- CUDA：`RWKV-v4/cuda/wkv_cuda.cu` `kernel_forward` L6–L38
- 链接：https://github.com/BlinkDL/RWKV-LM/blob/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v4/src/model_run.py#L318-L354 ，https://github.com/BlinkDL/RWKV-LM/blob/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v4/cuda/wkv_cuda.cu#L6-L38

RWKV-4 的 WKV（逐通道 `c`，`w = −exp(time_decay) < 0`，`u = time_first`）：

```
wkv_t = ( Σ_{i<t} e^{(t−1−i)w + k_i} v_i  +  e^{u + k_t} v_t ) / ( Σ_{i<t} e^{(t−1−i)w + k_i}  +  e^{u + k_t} )
```

——一个"按 `exp(k)` 加权、随距离指数衰减"的 softmax 式平均，当前 token 额外拿到 bonus `u`。分子分母各是一个线性递归：`a_t = e^w a_{t−1} + e^{k_t} v_t`，`b_t = e^w b_{t−1} + e^{k_t}`。

```cuda
21    F p = 0, q = 0, o = MIN_VALUE;
22    // p and q are running sums divided by exp(o) (to avoid overflows)
23    for (int i = 0; i < T; i++) {
26        F no = max(o, u + k[ii]);
27        F A = exp(o - no);
28        F B = exp(u + k[ii] - no);
29        y[ii] = (A * p + B * v[ii]) / (A * q + B);
31        no = max(w + o, k[ii]);
32        A = exp(w + o - no);
33        B = exp(k[ii] - no);
34        p = A * p + B * v[ii];
35        q = A * q + B;
36        o = no;
```

- 每个线程负责一个 `(batch, channel)`（L10–L13），沿时间顺序扫描——没有 chunk 并行，训练时就是一个 `T` 步的 RNN 扫描。
- `p, q` 是分子、分母的运行和，但都**除以了 `exp(o)`**（L22 注释），`o` 是当前见过的最大指数。这是和 FlashAttention online softmax 完全相同的"减去最大值"技巧：`exp(k)` 很容易溢出 fp32。
- L26–L29：输出时把当前 token 以 `u + k_t` 的权重并入；L31–L36：状态更新时旧状态衰减 `w`、新 token 以 `k_t` 权重写入。
- `RWKV_RNN.SA`（`model_run.py` L335–L350）是同一套公式的 PyTorch 单步版：`aa, bb, pp` 就是 `p, q, o`。每层每通道 3 个标量 + 上一个 token 的 `x`（L328 `self.xx[name] = xx`，用于 token shift）——这就是 RWKV-4 的全部推理状态。
- L137–L139 的 **token shift**：`xk = x·μ + x_{t−1}·(1−μ)`，和当前 token 混合上一个 token，作用类似 FLA/Mamba 里的 short conv（长度 2 的卷积）。
- 注意：RWKV-4 的状态是**向量**（每通道一个标量），不是 `d_k×d_v` 矩阵，容量小。RWKV-5/6 改成矩阵状态（多头线性注意力 + 衰减），RWKV-7 引入广义 delta rule。

#### Stop 17 · RWKV-7 的 time-mix：广义 delta rule

- `RWKV-v7/rwkv_v7_demo_rnn.py` `time_mixing__` L112–L150；状态初始化 L284–L288
- 链接：https://github.com/BlinkDL/RWKV-LM/blob/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v7/rwkv_v7_demo_rnn.py#L112-L150

```python
120    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
123    kk = k * k_k
124    kk = torch.nn.functional.normalize(kk.view(H,N), dim=-1, p=2.0).view(-1)
125    k = k * (1 + (a-1) * k_a)
138    w = w0 + w.float()
140    w = torch.exp(-0.606531 * torch.sigmoid(w)) # 0.606531 = exp(-0.5)
143    vk = v.view(H,N,1) @ k.view(H,1,N)
144    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
145    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
146    out = state.to(dtype=x.dtype) @ r.view(H,N,1)
```

- 状态 `state: [H, N, N]`（L287：`(n_embd // head_size, head_size, head_size)`，fp32），布局是 `[value 维, key 维]`。
- L145：`S_t = S_{t−1}·diag(w_t) + S_{t−1}·(−κ̂)(κ̂ ⊙ a)ᵀ + v_t k_tᵀ = S_{t−1}(diag(w_t) − κ̂ (a_t⊙κ̂)ᵀ) + v_t k_tᵀ`。
  - `diag(w_t)`：逐通道衰减（像 GLA），`w ∈ (exp(−e^{−0.5}), 1)`（L140）。
  - `−κ̂(a⊙κ̂)ᵀ`：delta rule 的擦除项，`κ̂` 是 L2 归一化的"擦除 key"（L123–L124），`a ∈ (0,1)` 是**逐通道**的学习率（L120）——相当于把 DeltaNet 的标量 `β` 推广成向量。
  - `v kᵀ`：写入，L125 的 `k` 还按 `a` 做了调整。
- L146：读出 `o = S r`（`r` receptance ≈ query）。L148–L149：GroupNorm + 一个"当前 token bonus"项（类似 RWKV-4 的 `u`）。
- 对照：取 `w=1`、`a=β`（标量）、`κ̂ = k`，L145 就退化为 DeltaNet 的 `S(I − β k kᵀ) + v kᵀ`（转置布局）。FLA 中有对应的高效实现 `fla/ops/rwkv7/`（`chunk.py`、`fused_recurrent.py`）。

---

## 4. 不同方法对比

| 模型 | 状态形状（每 head） | 状态更新 | 衰减 | 擦除/写入 | 训练并行形式 | 典型 kernel（本章仓库） |
|---|---|---|---|---|---|---|
| Softmax attention | KV cache `T×d`（增长） | append | 无 | — | 并行 `O(T²)` | FlashAttention |
| Linear attention | `d_k×d_v` | `S += k vᵀ` | 无 | 只加 | chunk（cumsum） | `fla/ops/linear_attn` |
| RetNet | `d_k×d_v` | `S = γS + k vᵀ` | 常数标量/head | 只加 | chunk | `chunk_retention`→`chunk_simple_gla` |
| Mamba2 (SSD) | `headdim×d_state` | `S = αₜS + (Δx)Bᵀ` | 数据依赖标量 | 只加 | chunk（SSD 四步） | `ssd_minimal` / `mamba_chunk_scan_combined` |
| GLA | `d_k×d_v` | `S = diag(αₜ)S + k vᵀ` | 数据依赖向量 | 只加 | chunk（二级分块） | `chunk_gla` |
| Mamba1 | `d_inner×d_state`（逐通道） | `x = exp(ΔA)⊙x + ΔBu` | 数据依赖、逐通道逐状态 | 只加 | 并行 scan（不能用矩阵乘） | `selective_scan_fn` |
| RWKV-4 | 每通道 3 个标量 | 指数衰减加权平均 | 常数向量 | 只加 | 顺序扫描（CUDA） | `wkv_cuda.cu` |
| DeltaNet | `d_k×d_v` | `S = (I−βkkᵀ)S + βk vᵀ` | 无 | delta rule | chunk + WY | `fla/ops/delta_rule` |
| Gated DeltaNet | `d_k×d_v` | `S = α(I−βkkᵀ)S + βk vᵀ` | 数据依赖标量 | delta rule | chunk + WY | `chunk_gated_delta_rule` |
| RWKV-7 | `N×N` | `S = S(diag(w) − κ̂(a⊙κ̂)ᵀ) + v kᵀ` | 数据依赖向量 | 广义 delta（向量学习率） | chunk | `fla/ops/rwkv7` |

**复杂度**：训练 chunk 形式都是 `O(T·C·d + T·d_k·d_v)`；解码每步 `O(d_k·d_v)`，与上下文长度无关。

---

## 5. 常见困惑 / 坑

1. **"线性注意力是在近似 softmax 吗？"** 早期（Performer、Katharopoulos et al.）是；现代模型（GLA、DeltaNet、Mamba2）已经不再追求近似 softmax，而是把它当作一种独立的、有状态的序列混合器，`φ` 往往只是恒等 / SiLU + L2 归一化，分母也被输出端 norm 替代。
2. **结合律只在无掩码时直接成立**。因果掩码 `(QKᵀ⊙M)V ≠ Q(KᵀV)`，所以必须用递归或 chunk 形式；chunk 是"块内二次、块间线性"的折中。
3. **布局地狱**：FLA kernel 输入是 `[B, T, H, K]`，naive 里常转成 `[B, H, T, K]`；FLA 状态默认 `[K, V]`，`state_v_first=True` 时是 `[V, K]`；Mamba 状态是 `[headdim, d_state] = [V, K]`；Mamba1 的 `selective_scan` 是 `[B, D, L]`（长度在最后）。对照公式前先把 shape 写下来。
4. **log 空间与 `exp2`**：门都以 `g = log α ≤ 0` 存储，累乘变累加（数值稳定）。FLA kernel 把 cumsum 乘以 `1/ln2` 后用 `exp2`；naive 版用 `exp`。自己写 kernel 对拍时别漏了这个缩放。
5. **衰减掩码的上三角会溢出**：`G_t − G_s` 在 `s > t` 时是正数，`exp` 可能变成 inf，`inf×0 = NaN`。要么先置 `−inf` 再 `exp`（§2.1、Mamba `segsum`），要么像 FLA naive 那样 `tril` 两次。
6. **状态要用 fp32**：FLA naive 一律 `.float()`，kernel 中 `b_h` 是 `tl.float32`，RWKV-7 demo 的 state 也是 fp32。状态是成千上万步的累加量，bf16 误差会累积。
7. **chunk 与 fused_recurrent 数学等价，速度特性相反**：chunk 适合长序列训练/prefill（Tensor Core），fused_recurrent 适合 `T` 很小的解码。FLA 层会自动切换（`GatedDeltaNet.forward` L227–L232）。
8. **"不需要 KV cache"不等于"什么都不缓存"**：还要缓存 short conv 的最后 `conv_size−1` 个输入（Mamba 的 `conv_state`、FLA 的 `conv_state`），RWKV 要缓存上一个 token 的 `x`（token shift）。
9. **Delta rule 必须归一化 key**：`I − βkkᵀ` 只有在 `‖k‖≤1`、`β∈(0,2)` 时特征值才在 `[−1,1]` 内，否则状态会爆炸。FLA 用 `use_qk_l2norm_in_kernel=True` 保证。
10. **Delta rule 的 chunk 形式里有一个 `C×C` 三角矩阵求逆**，它是 delta 系列 kernel 比 GLA 更复杂的主要原因；naive 代码里的 `for i in range(1, chunk_size)` 前代循环就是在做这件事。
11. **Mamba1 vs Mamba2**：Mamba1 的衰减逐通道逐状态维，无法写成 `T×T` 矩阵乘，只能用 scan；Mamba2 每 head 一个标量衰减，于是可以 chunk + 矩阵乘，训练快很多，代价是表达力略降（用更大的 `d_state` 弥补）。
12. **召回能力的上限**：固定大小状态无法无损记住任意长的上下文。需要精确检索的任务（长文档问答、复制）上纯线性模型明显弱于 Transformer，这就是混合架构流行的原因。
13. **数值上 RWKV-4 要做 max-trick**：它的权重是 `exp(k)`（无界），必须像 online softmax 那样维护运行最大值（`wkv_cuda.cu` L22）。而 GLA/Mamba 的门是 `(0,1)` 内的乘法衰减，天然有界。

---

## 6. 自测题 + 动手练习

### 自测题

1. 写出线性注意力的递归形式，并说明为什么解码时它不需要 KV cache。状态的形状是什么？
2. 为什么因果掩码下不能直接用 `Q(KᵀV)` 计算？chunk 形式是如何绕过的？块长 `C` 变大/变小各有什么影响？
3. 在 `naive_chunk_linear_attn` 中，`kv.cumsum(2)` 之后为什么还要右移一位（L62）？
4. 推导：带标量衰减时，并行形式中的 `D[t,s]` 为什么等于 `exp(G_t − G_s)`？
5. RetNet、Mamba2、GLA 的衰减分别是什么形状？为什么 GLA 的 chunk kernel 比 Simple GLA 复杂？
6. 写出 delta rule 的更新公式，并证明 `‖k‖=1, β=1` 时更新后 `Sᵀk = v`。
7. 在 `delta_rule_chunkwise` 中，`u`、`w` 分别是什么？哪一步依赖上一块的状态？
8. 把 Mamba2 的 `(x, Δ, A, B, C)` 映射到线性注意力的 `(q, k, v, g)`。`ssd_minimal_discrete` 的四步分别对应 chunk 形式的哪一项？
9. Mamba1 的 `selective_scan_ref` 中 `Δ` 起什么作用？为什么 Mamba1 不能像 Mamba2 那样用矩阵乘训练？
10. RWKV-4 的 WKV CUDA kernel 中 `o` 变量的作用是什么？它和 FlashAttention 的哪个技巧相同？
11. FLA 的 `GatedDeltaNet` 在什么条件下用 `chunk`、什么条件下用 `fused_recurrent`？它的推理缓存里存了哪些东西？
12. RWKV-7 的状态更新与 Gated DeltaNet 有何异同？

### 动手练习

1. **跑通 §2.1、§2.2**，再加入 GLA 的向量门（`g: [B,H,T,K]`），写出它的并行形式与 chunk 形式并验证等价（提示：块内用 `(q⊙exp(Γ)) (k⊙exp(−Γ))ᵀ`，注意数值稳定）。
2. **复杂度实验**：固定 `d`，令 `T` 从 1K 到 32K，测量 §2.1 中 `parallel` 与 `chunk` 的耗时和峰值显存，画出曲线，找到交叉点。
3. **对拍 FLA**（有 GPU 时）：用 `fla.ops.simple_gla.chunk_simple_gla` 和 `fla.ops.simple_gla.naive.naive_recurrent_simple_gla` 在随机输入上比较输出，并检查 `output_final_state=True` 返回的状态；再把 `initial_state` 设为前半段的终态、只喂后半段，验证与一次喂完整段结果一致（这就是"状态替代 KV cache"）。
4. **联想记忆容量实验**：`d_k = 16`，往状态里依次写入 `n` 个随机 `(k, v)` 对，再用每个 `k` 读出，测量平均误差随 `n` 的变化，对比纯线性注意力和 delta rule（β=1）。你应看到线性注意力在 `n > d_k` 后迅速变差，而 delta rule 总能精确取回最近写入的值。
5. **Mamba 单步 vs 扫描**：用 `selective_scan_ref(…, return_last_state=True)` 处理长度 `L` 的序列，再用 `Mamba.step` 的公式（L240–L246）逐 token 跑，验证输出和最终状态一致。
6. **RWKV-4 数值稳定性**：用 PyTorch 实现不带 max-trick 的 WKV 递归，喂入 `k` 较大的输入（如 `k ~ N(0, 30²)`）观察溢出，再加上 `wkv_cuda.cu` L26–L36 的技巧修复。
7. **显存账**：按 §2.3 的方法，对比一个 7B Transformer 与同规模 Gated DeltaNet 在 4K / 32K / 128K 上下文、batch=16 时的推理缓存大小。
