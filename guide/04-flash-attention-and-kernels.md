# 04 · FlashAttention 与 Attention Kernel：从 online softmax 到 FA2 / FA4 源码

> 阅读前提：你知道 `softmax(QKᵀ/√d)V` 是什么。本篇目标：让你能看懂 Triton 教程的 fused attention、FA2 的 CUDA forward、以及本仓库 FA4（CuTeDSL）的整体结构，并理解 varlen、paged KV、split-KV 这些推理特性在 kernel 里是怎么落地的。
>
> 行号都已对照当前 checkout 核实。本仓库 FA 源码在 `flash-attention/` 下：FA2 = `flash-attention/csrc/flash_attn`，FA3 = `flash-attention/hopper`，FA4 = `flash-attention/flash_attn/cute`。

---

## 0. 要解决的问题 / 一句话直觉

**问题**：标准 attention 先算出 `S = QKᵀ`（`[N, N]`），写回显存，再读出来做 softmax 得 `P`（又一个 `[N, N]`），再读 `P` 乘 `V`。N=8k、32 头、bf16 时单层的 S 就是 8k·8k·32·2B = 4 GiB。瓶颈不是乘法，而是**反复读写这个 N×N 矩阵的显存带宽**。

**一句话直觉**：把 Q 切成行块、K/V 切成列块，一块一块地在**片上 SRAM（shared memory / 寄存器 / Blackwell 的 TMEM）**里算 `S_ij → P_ij → P_ij V_j`，靠 **online softmax** 在没看到整行的情况下也能得到精确结果；N×N 矩阵从不落到显存。这叫 **IO-aware**：以减少 HBM 读写为第一目标设计算法。反向传播时不存 P，只存每行的 `LSE`（logsumexp），需要时重算。

---

## 1. 核心原理

### 1.1 记号与形状

单个 (batch, head)：
```
Q: [N_q, d]    K: [N_k, d]    V: [N_k, d_v]
S = Q Kᵀ · scale : [N_q, N_k]        scale = 1/√d
P = softmax_row(S) : [N_q, N_k]
O = P V : [N_q, d_v]
LSE_i = log Σ_j exp(S_ij) : [N_q]    （反向和 split-KV 合并都要用）
```
分块：`Q` 切成 `T_r = ⌈N_q/B_r⌉` 个 `[B_r, d]` 块（FA 源码里叫 `m_block`，大小 `kBlockM`/`tile_m`），`K,V` 切成 `T_c = ⌈N_k/B_c⌉` 个 `[B_c, d]` 块（`n_block`，`kBlockN`/`tile_n`）。

### 1.2 online softmax 推导

对 Q 的第 i 行，假设已经处理了前 j-1 个 K 块，维护三个量：
```
m   : 目前为止见过的最大 score            标量
ℓ   : Σ exp(S_ik − m)                     标量
acc : Σ exp(S_ik − m) · V_k               [d_v] 向量（未归一化的输出）
```
来了新块 j（列下标集合 J_j，分数 `s = S_i,J_j : [B_c]`）：
```
m_new  = max(m, max(s))
α      = exp(m − m_new)                   # 旧统计量的缩放因子（≤ 1）
p      = exp(s − m_new)                   # [B_c]
ℓ_new  = α·ℓ + Σ p
acc_new= α·acc + p · V_J                  # [d_v]
```
全部块处理完：`O_i = acc / ℓ`，`LSE_i = m + log ℓ`。

**为什么正确**：归纳假设 `acc = Σ_{k<J} exp(S_ik − m) V_k`。乘上 `α = exp(m − m_new)` 后变成 `Σ exp(S_ik − m_new) V_k`，即换成了新的基准 m_new；再加上新块的 `Σ exp(S_ik − m_new) V_k`，就得到全部已见列关于 m_new 的和。ℓ 同理。最后 `acc/ℓ = Σ exp(S_ik−m)V_k / Σ exp(S_ik−m) = Σ softmax(S_i)_k V_k`。减 m 只是为了数值稳定（防止 exp 溢出），不改变结果。

矩阵形式（一个 Q 块 `[B_r, d]` 同时做）：`m, ℓ : [B_r]`，`acc : [B_r, d_v]`，`α[:, None] * acc`。

**两个工程技巧**（所有实现都用）：
1. **exp2 代替 exp**：`exp(x) = 2^(x·log₂e)`。把 `scale·log₂e` 合并成一个常数 `scale_log2`，`p = exp2(S·scale_log2 − m·scale_log2)` 正好是一条 FFMA + 一条 `ex2` 指令。FA2 的 `scale_softmax_log2`、Triton 的 `qk_scale *= 1.44269504`、FA4 的 `softmax_scale_log2` 都是它。
2. **延迟归一化**：循环中不除以 ℓ，只在最后除一次（FA2 `normalize_softmax_lse`、FA4 `finalize`）。

### 1.3 合并两个“部分结果”（split-KV 的数学）

把 K 分成两段 A、B，分别算出（已归一化的）`O_A, LSE_A` 和 `O_B, LSE_B`，则：
```
LSE = log(exp(LSE_A) + exp(LSE_B))          # 用 max 技巧稳定地算
O   = exp(LSE_A − LSE)·O_A + exp(LSE_B − LSE)·O_B
```
推广到 S 段：`w_s = exp(LSE_s − LSE)`，`O = Σ w_s O_s`。这就是 FA 的 `combine` kernel、FlashInfer 的 `merge_state`。它和 online softmax 是同一个“可结合的 (m, ℓ, acc) 幺半群”。

### 1.4 为什么 IO-aware 能快

设 SRAM 大小为 M（字节量级 ~100–200 KB/SM）。
- 标准实现 HBM 访问量：`Θ(N·d + N²)`（S、P 各写读一次 N²）。
- FlashAttention：每个 Q 块要把全部 K/V 扫一遍，K/V 被读 `T_r = N/B_r` 次，而 `B_r ~ M/d`，所以 `Θ(N²·d²/M)`。d=128、M~100KB 时 `d²/M` 远小于 1，HBM 流量小一个数量级，且**额外显存从 O(N²) 降到 O(N)**（只存 O 和 LSE）。

另外 FLOPs 并没有减少（反向还多了一次重算 S），但 attention 原本是带宽受限的，换成计算受限后反而快。

### 1.5 causal 与块跳过

causal（对齐右下角）：query 行 r 只看 `col ≤ r + (N_k − N_q)`。对一个 Q 块 `[m·B_r, (m+1)·B_r)`：
- 完全在对角线右上方的 K 块**整块跳过**（不加载、不计算）→ `n_block_max = ceil(((m+1)·B_r + N_k − N_q) / B_c)`；
- 完全在左下方的块不需要 mask；
- 只有跨对角线的少数块需要逐元素 mask。

FA2 的 `n_masking_steps`、Triton 的 `STAGE 1/2`、FA4 的 `BlockInfo.get_n_block_min_max` 都是这件事。

### 1.6 反向传播（Triton 教程用到）

已存 `O`、`LSE`。令 `dO` 为上游梯度：
```
P   = exp(S·scale − LSE)                # 用 LSE 一步重算出归一化的 P，无需再算 max/sum
dV  = Pᵀ dO
dP  = dO Vᵀ
D_i = rowsum(dO ∘ O) = Σ_j P_ij dP_ij   # 预处理 kernel 先算好，形状 [N_q]
dS  = P ∘ (dP − D[:, None])
dQ  = dS K · scale,   dK = dSᵀ Q · scale
```

---

## 2. 最小实现：纯 PyTorch 的分块 online-softmax attention + split-KV 合并

```python
import math, torch

def naive_attn(q, k, v, causal=False):              # q:[Nq,d] k:[Nk,d] v:[Nk,dv]
    s = q @ k.T / math.sqrt(q.shape[-1])
    if causal:                                       # 右下角对齐，与 FA 一致
        Nq, Nk = s.shape
        s = s.masked_fill(torch.arange(Nk)[None] > torch.arange(Nq)[:, None] + (Nk - Nq), float('-inf'))
    return s.softmax(-1) @ v

def flash_attn_tiled(q, k, v, causal=False, Br=16, Bc=16):
    Nq, d = q.shape; Nk = k.shape[0]; scale = 1 / math.sqrt(d)
    O = torch.empty(Nq, v.shape[1]); LSE = torch.empty(Nq)
    for i in range(0, Nq, Br):                       # 外层：Q 块（GPU 上 = 一个 CTA / program）
        qi = q[i:i + Br]; r = qi.shape[0]
        rows = torch.arange(i, i + r)
        m = torch.full((r,), float('-inf')); l = torch.zeros(r); acc = torch.zeros(r, v.shape[1])
        for j in range(0, Nk, Bc):                   # 内层：K/V 块
            if causal and j > rows[-1] + (Nk - Nq):  # 整块在对角线右上方 → 跳过（n_block_max）
                break
            s = qi @ k[j:j + Bc].T * scale           # [r, Bc]  —— 只存在于“片上”
            if causal:
                cols = torch.arange(j, j + s.shape[1])
                s = s.masked_fill(cols[None] > rows[:, None] + (Nk - Nq), float('-inf'))
            m_new = torch.maximum(m, s.max(-1).values)
            m_safe = torch.where(m_new == float('-inf'), torch.zeros_like(m_new), m_new)  # 防 (-inf)-(-inf)
            p = torch.exp(s - m_safe[:, None])       # [r, Bc]
            alpha = torch.exp(m - m_safe)            # [r]  旧统计量的缩放
            l = l * alpha + p.sum(-1)
            acc = acc * alpha[:, None] + p @ v[j:j + Bc]
            m = m_new
        O[i:i + r] = acc / l[:, None]                # 延迟归一化
        LSE[i:i + r] = m + torch.log(l)
    return O, LSE

def split_kv_attn(q, k, v, num_splits):             # decode 场景：沿 K 切分并行，再合并
    chunks = torch.tensor_split(torch.arange(k.shape[0]), num_splits)
    outs, lses = zip(*[flash_attn_tiled(q, k[c], v[c]) for c in chunks])
    lse = torch.stack(lses)                          # [S, Nq]
    lse_final = torch.logsumexp(lse, dim=0)          # [Nq]
    w = torch.exp(lse - lse_final)                   # [S, Nq] 每段权重
    return (w[..., None] * torch.stack(outs)).sum(0), lse_final

if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = torch.randn(50, 32), torch.randn(70, 32), torch.randn(70, 24)
    for causal in (False, True):
        print("causal", causal, (flash_attn_tiled(q, k, v, causal)[0] - naive_attn(q, k, v, causal)).abs().max().item())
    print("split-kv", (split_kv_attn(q, k, v, 3)[0] - naive_attn(q, k, v)).abs().max().item())
```
（我用逐行等价的 numpy 版本在 float64 下验证：三项误差均在 1e-15 量级。注意 causal 时要求 `Nq ≤ Nk`，否则会有整行全被 mask 的情况，`l=0` 需要像 FA 一样特殊处理成输出 0、LSE=±inf。）

这段代码与真实 kernel 的对应：
| toy | Triton 教程 | FA2 CUDA | FA4 CuTeDSL |
|---|---|---|---|
| 外层 `for i` | `start_m = tl.program_id(0)` | `m_block = blockIdx.x` | tile scheduler 给出 `m_block` |
| 内层 `for j` | `for start_n in tl.range(lo, hi, BLOCK_N)` | `for (; n_block >= n_block_min; --n_block)`（倒序） | load/mma warp 的 `n_block` 循环（倒序） |
| `m, l, acc` | `m_i, l_i, acc` | `softmax.row_max, row_sum, acc_o` | `Softmax.row_max/row_sum`，acc 在 TMEM |
| `alpha` | `alpha = exp2(m_i - m_ij)` | `scores_scale` | `row_scale` / `acc_scale` |
| `break` 跳块 | `STAGE` 1/2 的 `lo, hi` | `n_block_max` | `BlockInfo.get_n_block_min_max` |
| `split_kv_attn` | — | `flash_fwd_splitkv_kernel` + `combine_attn_seqk_parallel` | `is_split_kv` + `FlashAttentionForwardCombine` |

---

## 3. 源码导读（按阅读顺序）

### 3.1 Triton 教程 `06-fused-attention.py`：最容易读懂的真 kernel（fwd + bwd）

文件：`repos/02-attention-kernels/triton/python/tutorials/06-fused-attention.py`（文件头 L1–5 写明 “Triton implementation of the Flash Attention v2 algorithm”）。输入布局 `q, k, v: (Z, H, N_CTX, HEAD_DIM)`（见 `test_op` L639–641）。

**Stop 1：`_attn_fwd` —— 每个 program 负责一个 `[BLOCK_M, HEAD_DIM]` 的 Q 块**

- L176–245：https://github.com/triton-lang/triton/blob/4ee1a827440d453ca276ded4219190d5e7346b8b/python/tutorials/06-fused-attention.py#L176-L245

```python
start_m = tl.program_id(0)                 # 第几个 Q 块
off_hz = tl.program_id(1)                  # (batch, head) 展平
...
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
qk_scale = sm_scale
qk_scale *= 1.44269504  # 1/log(2)         # exp → exp2
q = desc_q.load([qo_offset_y, 0])          # Q 块常驻 SRAM
if STAGE & 1:   # 非对角线块（causal 时只到对角线前）
    acc, l_i, m_i = _attn_fwd_inner(..., 4 - STAGE, ...)
if STAGE & 2:   # 对角线块（需要 mask）
    acc, l_i, m_i = _attn_fwd_inner(..., 2, ...)
m_i += tl.math.log2(l_i)                   # 存 log2 域的 LSE
acc = acc / l_i[:, None]
tl.store(m_ptrs, m_i)
desc_o.store([qo_offset_y, 0], acc.to(dtype))
```
- grid（L548–549）：`(cdiv(N_CTX, BLOCK_M), Z*H, 1)`。
- `stage = 3 if causal else 1`（L514）。causal 时调用两次 inner：STAGE=1 处理 `[0, start_m*BLOCK_M)` 的无 mask 块，STAGE=2 处理对角线块；非 causal 时 inner 拿到 STAGE=3，处理 `[0, N_CTX)`。
- `l_i` 初始化为 1.0 而不是 0：无所谓，因为第一次迭代 `alpha = exp2(-inf - m) = 0` 会把它清零。
- 存的 `M` 是 `m_i + log2(l_i)`，即 **log2 域的 LSE**，反向用它一步重算 P。

**Stop 2：`_attn_fwd_inner` —— online softmax 本体**

- L47–108：https://github.com/triton-lang/triton/blob/4ee1a827440d453ca276ded4219190d5e7346b8b/python/tutorials/06-fused-attention.py#L47-L108

```python
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
    k = desc_k.load([offsetk_y, 0]).T          # [HEAD_DIM, BLOCK_N]
    qk = tl.dot(q, k)                          # [BLOCK_M, BLOCK_N]，fp32 累加
    if STAGE == 2:
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)                       # [BLOCK_M, BLOCK_N]
    alpha = tl.math.exp2(m_i - m_ij)           # 旧 acc 的缩放
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    v = desc_v.load([offsetv_y, 0])            # [BLOCK_N, HEAD_DIM]
    p = p.to(dtype)                            # 降精度后进 tensor core
    acc = tl.dot(p, v, acc)                    # acc += P V
    l_i = l_i * alpha + l_ij
    m_i = m_ij
```
和 §2 toy 一一对应。几个“为什么”：
- mask 用 `-1.0e6` 而不是 `-inf`：对角线块里每行至少有一个有效元素，用大负数避免 `-inf - -inf = NaN`。
- `p.to(dtype)`：第二个 GEMM 的 A 操作数用 fp16/bf16，这是 tensor core 的要求；累加器 `acc` 仍是 fp32。
- `l_i`、`m_i` 更新放在循环末尾是为了降低寄存器压力（注释 L102–103）。
- L85–91 在“非 Hopper、开启 warp_specialize、且 BLOCK_M == HEAD_DIM == 128”时把 `acc` 拆两半分别乘 `alpha`，是给编译器的调度提示，数学不变。

**Stop 3：反向 —— preprocess + 两段循环**

- `_attn_bwd_preprocess` L248–262：`delta = tl.sum(o * do, axis=1)`，即 §1.6 的 `D_i`。
- `_attn_bwd_dkdv` L266–315（固定一个 K/V 块，扫 Q 块）：
```python
qkT = tl.dot(k, qT)                        # [BLOCK_N1, BLOCK_M1] = (Q Kᵀ)ᵀ
pT = tl.math.exp2(qkT - m[None, :])        # 用存下的 LSE 直接得到归一化的 Pᵀ
dv += tl.dot(ppT, do)                      # dV += Pᵀ dO
dpT = tl.dot(v, tl.trans(do)).to(tl.float32)   # dPᵀ = V dOᵀ
dsT = pT * (dpT - Di[None, :])             # dSᵀ = Pᵀ ∘ (dPᵀ − D)
dk += tl.dot(dsT, tl.trans(qT))            # dK += dSᵀ Q
```
- `_attn_bwd_dq` L319–363（固定一个 Q 块，扫 K/V 块）：`dq += tl.dot(ds, tl.trans(kT))`。
- `_attn_bwd` L366–500：同一个 program 先做 dK/dV（L402–451）再做 dQ（L453–500），causal 时对角线块用更小的 `MASK_BLOCK_*` 并开 mask。
- 一个容易看漏的细节：`backward` 里先把 K 预乘了 `sm_scale * RCP_LN2`（L589–590），所以 kernel 里 `qkT` 已经在 log2 域；最后 `dk *= sm_scale`（L449）、`dq *= LN2`（L499）把尺度还原。
- 链接：https://github.com/triton-lang/triton/blob/4ee1a827440d453ca276ded4219190d5e7346b8b/python/tutorials/06-fused-attention.py#L248-L500 ；host 端 `_attention.forward/backward` L503–616。

> 为什么 dK/dV 和 dQ 要分开循环？dK/dV 对 Q 维求和、dQ 对 K 维求和。若在同一个循环里做，就必须对 dQ 做跨 program 的原子加（FA2 正是这么做的：`dq_accum` + atomicAdd）。Triton 教程选择“重算一遍 S”换取无原子操作。

---

### 3.2 FA2 CUDA forward：CuTe C++ 版本的同一个算法

**Stop 1：kernel 入口与 grid**

- `flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h` L32–64
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h#L32-L64

```cpp
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, ...) {
    FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, ...>(params);
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, ...)          // decode/paged 用
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, ...)  // 合并 split 结果
...
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);       // 每个 CTA = (Q块, batch, head)
```
一堆 bool 模板参数（causal、even_MN、even_K…）通过 `BOOL_SWITCH` 在编译期展开成不同 kernel，避免运行时分支。

**Stop 2：`compute_attn_1rowblock` —— 一个 CTA 的全部工作**

- `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h` L54–440（入口 `compute_attn` L1082–1099 把 `blockIdx.{x,y,z}` 拆成 `m_block, bidb, bidh`）
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h#L87-L162

(a) 算本 Q 块要扫哪些 K 块（L87–95）：
```cpp
const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);     // varlen 下取本请求的 seqlen 与偏移
if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
if (Is_causal || Is_local) {
    n_block_max = std::min(n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
}
```
就是 §1.5 的公式（causal 时 window_size_right=0）。

(b) 建全局/共享内存张量视图（L145–173）：`mQ` 形状 `(actual_seqlen_q, h, d)`，`gQ = local_tile(mQ(_, bidh, _), (kBlockM, kHeadDim), (m_block, 0))` 取出本块；`gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), ...)`——**GQA 就是 `bidh / h_h_k_ratio`**。`sQ/sK/sV` 是 shared memory 中的 tile。

(c) 主循环分两段（L297–436）：
```cpp
// For performance reason, we separate out two kinds of iterations:
// those that need masking on S, and those that don't.
constexpr int n_masking_steps = (!Is_causal && !Is_local) ? 1
    : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    clear(acc_s);                                         // acc_s: S 块，fp32，分布在线程寄存器
    ... copy V(n_block) gmem→smem (cp.async，与下面的 GEMM 重叠)
    FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, ...);        // S = Q Kᵀ  (tensor core mma)
    mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, ...);
    ... prefetch K(n_block-1)
    masking_step == 0
        ? softmax.template softmax_rescale_o</*Is_first=*/true,  ...>(acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o</*Is_first=*/false, ...>(acc_s, acc_o, params.scale_softmax_log2);
    Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);   // P 转 fp16/bf16，留在寄存器
    FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, ...);          // O += P V（A 来自寄存器）
}
for (; n_block >= n_block_min; --n_block) { ... 同上但不 mask ... }
Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```
要点：
- **K 块倒序遍历**（从 `n_block_max - 1` 往下）：最先处理的正好是需要 mask 的对角线块，于是 mask 只发生在前 `n_masking_steps` 次迭代，后面的循环体完全不含 mask 分支。
- **软件流水**：计算 S 的同时异步加载 V，计算 softmax 前预取下一个 K（`cp_async_fence/wait`）。
- `gemm_rs`：P 不写回 shared memory，直接作为寄存器操作数进第二个 MMA（`convert_layout_acc_Aregs` 把累加器布局重解释成 A 操作数布局）。

**Stop 3：softmax 的 rescale**

- `flash-attention/csrc/flash_attn/src/softmax.h` `Softmax::softmax_rescale_o` L136–167，`normalize_softmax_lse` L169–186，`scale_apply_exp2` L66–92
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/softmax.h#L128-L187

```cpp
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));  // 视为 (行, 列)
    if (Is_first) {
        reduce_max</*zero_init=*/true>(scores, row_max);
        scale_apply_exp2(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        reduce_max</*zero_init=*/false>(scores, row_max);              // m_new
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_max_cur = !Check_inf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);   // α
            row_sum(mi) *= scores_scale;
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }  // acc *= α
        }
        scale_apply_exp2(scores, row_max, softmax_scale_log2);         // p = exp2(s*scale - m*scale)
        reduce_sum</*zero_init=*/false>(scores, row_sum);              // ℓ += Σp（只在线程内）
    }
};
```
- `kNRows` = 每个线程持有的行数（MMA 布局决定，通常 2 或 4）。一行 S 分布在 4 个相邻线程上，所以 `reduce_max` 里有 `quad_allreduce_`（`Allreduce<4>`，L38–45）用 warp shuffle 做 4 线程归约。
- `reduce_sum` **只做线程内求和**（L59–63 没有 quad allreduce），跨线程的求和推迟到 `normalize_softmax_lse` 开头（L172）——每次迭代省 2 次 shuffle。
- `Check_inf`：causal/local 下整行可能全被 mask，`row_max = -inf`，此时用 0 代替避免 `(-inf) - (-inf) = NaN`。
- `normalize_softmax_lse`（L169–186）：`lse = row_max * softmax_scale + __logf(sum)`（注意这里用自然对数 LSE），`acc_o *= 1/sum`；sum 为 0 或 NaN 时 LSE 置 `INFINITY`（split 时置 `-INFINITY`，让 combine 给它权重 0）。

**Stop 4：varlen 的 `BlockInfo`**

- `flash-attention/csrc/flash_attn/src/block_info.h` L12–45
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/block_info.h#L12-L45

```cpp
: sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
, actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
, actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + ...)
...
index_t q_offset(...) const { return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride; }
```
varlen 时 Q 是扁平的 `[total_q, h, d]`，第 b 个请求从 `cu_seqlens_q[b]` 行开始；grid 的 `num_m_block` 按 `max_seqlen_q` 算，短请求的多余 CTA 在 `if (m_block * kBlockM >= binfo.actual_seqlen_q) return;` 直接退出。`seqused_k` 优先于 `cu_seqlens_k`——paged/cache 场景（见 03 篇）就靠它。

**Stop 5：split-KV + paged KV + combine（decode 路径）**

- `compute_attn_1rowblock_splitkv` L505 起，`compute_attn_splitkv` L1103–1113，`combine_attn_seqk_parallel` L1117 起
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h#L1103-L1221

```cpp
const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
const int n_split_idx = Split ? blockIdx.y : 0;     // grid.y 变成 split 维
```
- 每个 split 只处理 `[n_split_idx * n_blocks_per_split, ...)` 的 K 块（L539），写出 fp32 的部分 O 和部分 LSE（`out_accum: [num_splits, b, h, seqlen_q, d]`、`softmax_lse_accum: [num_splits, b, h, seqlen_q]`，见 `flash_api.cpp` L338–339）。
- paged KV 的指针运算在 L591–601、L950–957（详见 03 篇 §3.6）。
- combine（L1191–1221）就是 §1.3：先 `lse_max`，再 `lse_sum = Σ expf(lse - lse_max)`，`lse_logsum = logf(lse_sum) + lse_max`，权重 `expf(lse_accum(l) - lse_logsum)` 存进 shared memory，再对 `out_accum` 加权求和。

- 何时 split：`flash-attention/csrc/flash_attn/flash_api.cpp` `num_splits_heuristic` L281–315
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/flash_api.cpp#L281-L340
```cpp
// If we have enough to almost fill the SMs, then just use 1 split
if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
...
float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
float eff = n_waves / ceil(n_waves);      // 最后一波 SM 的利用率
```
decode 时 `seqlen_q=1`，CTA 数 = `b × h × 1` 往往远小于 SM 数（H100 有 132 个 SM），不切的话大部分 SM 空闲；切 K 后 CTA 数 ×num_splits。

---

### 3.3 FA4（CuTeDSL）：入口 `interface.py`

FA4 用 Python 写 kernel（CuTeDSL），运行时 JIT 编译成 cubin。先看 host 侧怎么走到 kernel。

**Stop 1：公开 API**

- `flash-attention/flash_attn/cute/interface.py`：`flash_attn_func` L4099–4165，`flash_attn_varlen_func` L4168–4302
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/interface.py#L4168-L4267

docstring 的张量约定（L4204–4213）：
```
q:  (total_q, nheads,   hdim)   or (batch, seqlen_q, nheads,   hdim)
k:  (total_k, nheads_k, hdim)   or (batch, seqlen_k, nheads_k, hdim)
cu_seqlens_q: (batch + 1)       or seqused_q: (batch)
cu_seqlens_k: (batch + 1)       or seqused_k: (batch)
page_table: (batch, max_num_pages_per_seq)
```
L4231–4235 关于 paged KV 的注释很重要：TMA 按整页加载时，最后一页里超出 `seqused_k` 的行也会被读；分数虽然被 mask 成 -inf，但 `0 × NaN = NaN`，所以这些行必须是有限值。

两个函数都只是 `FlashAttnFunc.apply` / `FlashAttnVarlenFunc.apply`（autograd.Function，L3734 / L3872），forward 里调用 `_flash_attn_fwd`。

**Stop 2：`_flash_attn_fwd` —— 形状检查、选配置、编译缓存、按架构分发**

- L678–1966：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/interface.py#L764-L789

(a) paged KV 形状（L764–789）：
```python
if page_table is not None:
    assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
    assert page_table.dtype == torch.int32, "page_table must be int32"
    max_num_pages_per_seq = page_table.shape[1]
    num_pages, page_size = v.shape[:2]          # k/v: (num_pages, page_size, nheads_k, hdim)
```
(b) split-KV 的临时缓冲（L1063–1079）：
```python
is_split_kv = num_splits > 1
if is_split_kv:
    out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
    lse_partial = torch.empty(num_splits, *lse_partial_shape, dtype=torch.float32, device=device)
```
(c) 编译缓存：`compile_key`（L1452 起）包含 dtype、head_dim、causal、score_mod/mask_mod 哈希、是否 varlen、是否 split、page_size 相关等；`if compile_key not in _flash_attn_fwd.compile_cache:`（L1523）才走 `cute.compile`（L1779/L1845），缓存对象由 `get_jit_cache("fwd")` 创建（L1969）。第一次调用慢（编译），之后命中缓存。

(d) 按架构选 kernel 类（L1610–1760）：
```python
if arch // 10 == 8:   fa_fwd = FlashAttentionForwardSm80(...)        # L1613，assert not is_split_kv
elif arch // 10 == 9: fa_fwd = FlashAttentionForwardSm90(...)        # L1632，assert not is_split_kv
elif arch // 10 in [10, 11]:
    ... (MLA 变体) ...
    fa_fwd = FlashAttentionForwardSm100(head_dim, head_dim_v, qhead_per_kvhead=..., is_causal=causal,
        is_local=local, is_split_kv=is_split_kv, pack_gqa=pack_gqa, m_block_size=tile_m, n_block_size=tile_n,
        q_stage=q_stage, ..., paged_kv_non_tma=page_size not in [None, tile_n], ...)    # L1726-1750
elif arch // 10 == 12: fa_fwd = FlashAttentionForwardSm120(...)
```
注意 `paged_kv_non_tma=page_size not in [None, tile_n]`：**page 大小正好等于 K tile 大小（通常 128）时，一页就是一个 TMA tile，直接用 TMA 加载；否则走逐行 cp.async gather（`PagedKVManager`）**。

(e) split 后调用 combine（L1937–1958）：
```python
if is_split_kv:
    _flash_attn_fwd_combine(out_partial, lse_partial.transpose(-1, -2), out, lse_combine, cu_seqlens_q, seqused_q, ...)
```
(f) 默认的 split 启发式 `num_splits_heuristic`（L315–328）比 FA2 简单：`max(1, min(num_SMs // total_mblocks, max_splits, num_n_blocks))`，且 `num_n_blocks <= 4` 时不切。

> 类名对应：CLAUDE.md 说 `flash_fwd.py` 是 Hopper forward，但当前代码里 `flash_fwd.py` 定义的是 `FlashAttentionForwardBase`（L40）和 `FlashAttentionForwardSm80`（L579），Hopper 的 `FlashAttentionForwardSm90` 已移到 `flash_fwd_sm90.py`（L52），`flash_fwd.py` L1245 的 `__getattr__` 只是兼容旧的导入路径。interface.py L42–45 的 import 可以佐证。

---

### 3.4 FA4 Hopper：`flash_fwd_sm90.py` —— 生产者/消费者 warp specialization

- `flash-attention/flash_attn/cute/flash_fwd_sm90.py` L585–615，`mma_one_n_block` L1356–1415
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_sm90.py#L1356-L1415

```python
if warp_idx < 4:  # Producer
    cute.arch.setmaxregister_decrease(self.num_producer_regs)    # 搬运 warp 只需很少寄存器
    self.load(mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, pipeline_k, pipeline_v, pipeline_q, ..., mPageTable, ...)
else:  # Consumer
    cute.arch.setmaxregister_increase(self.num_mma_regs)         # 计算 warpgroup 拿走更多寄存器
    ...
    self.mma(...)
```
- 第一个 warpgroup（4 个 warp）只发 TMA 把 Q/K/V 搬到 shared memory，后面的 warpgroup 做 WGMMA + softmax。两者通过 `pipeline_k / pipeline_v`（多 stage 的 mbarrier 环形缓冲）同步。这就是 **warp specialization**：搬运与计算由不同 warp 并行，而不是同一组线程交替做。
- `setmaxregister_decrease/increase`（Hopper 的 `setmaxnreg`）把寄存器在 warpgroup 间重新分配。

`mma_one_n_block`：
```python
pipeline_k.consumer_wait(smem_pipe_read, ...)          # 等 K_j 到 smem
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)   # S = Q K_jᵀ（异步 WGMMA）
...
pipeline_k.consumer_release(smem_pipe_read)            # K_j 用完，生产者可以覆盖这个 stage
if const_expr(mask_fn is not None):
    mask_fn(acc_S=acc_S, n_block=n_block)
row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
utils.cvt_f16(tOrP_acc, tOrP_cur)                      # P → fp16/bf16
softmax.rescale_O(acc_O, row_scale)                    # acc_O *= α
pipeline_v.consumer_wait(smem_pipe_read, ...)
mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)       # O += P V_j
pipeline_v.consumer_release(smem_pipe_read)
```
完全是 §2 toy 内层循环的形状，只是每一步都夹着 pipeline 同步。`warp_scheduler_barrier_sync/arrive`（L1531–）让两个消费者 warpgroup 交替使用 tensor core（一个做 GEMM 时另一个做 softmax，即 FA3 论文里的 ping-pong）。

**`Softmax.online_softmax` 本体**

- `flash-attention/flash_attn/cute/softmax.py` `Softmax` L153–301
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/softmax.py#L186-L301

```python
for r in cutlass.range(cute.size(row_max), unroll_full=True):
    acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)
    row_max_cur = utils.fmax_reduce(acc_S_row, init_val=row_max[r] if not is_first else None, arch=arch)
    row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)    # 与 FA2 的 quad allreduce 一样
    row_max_prev = row_max[r]
    row_max[r] = row_max_cur
    if check_inf:
        row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
    ...
    acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True)
    row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2, fastmath=True)   # α
    acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch)
    row_sum[r] = acc_S_row_sum
    acc_S_mn[r, None].store(acc_S_row_exp)
return row_scale
```
这是 FA2 `softmax_rescale_o` 的逐行 Python 翻译，区别是它**返回** `row_scale`，由调用方决定何时 `rescale_O`。`finalize`（L252–288）：先 4 线程归约 row_sum，返回 `1/row_sum` 用于最后缩放 O，同时把 `row_sum` 改写成 LSE `(row_max*scale_log2 + log2(row_sum)) * ln2`，并处理 learnable sink（attention sink 的可学习 logit）。

---

### 3.5 FA4 Blackwell：`flash_fwd_sm100.py` —— 16 个 warp 各司其职

Blackwell（SM100）的变化：MMA 结果写进专用的 **Tensor Memory（TMEM）** 而不是寄存器，由单个线程发射（`tcgen05.mma`）；TMA 负责搬运；于是 FA4 把一个 CTA 拆成多种角色。

**Stop 1：warp 角色表**

- `flash-attention/flash_attn/cute/flash_fwd_sm100.py` L312–318
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_sm100.py#L312-L318

```python
self.softmax0_warp_ids = (0, 1, 2, 3)      # softmax warpgroup 0：处理 Q tile 0 的 S
self.softmax1_warp_ids = (4, 5, 6, 7)      # softmax warpgroup 1：处理 Q tile 1 的 S（q_stage=2）
self.correction_warp_ids = (8, 9, 10, 11)  # correction：在 TMEM 里给 O 乘 α，最后归一化写出
self.mma_warp_id = 12                      # 单个 warp 发射所有 tcgen05 MMA
self.epilogue_warp_ids = (13,)             # O: smem → gmem
self.load_warp_ids = (14,)                 # TMA 加载 Q/K/V
self.empty_warp_ids = (15,)
```
（之后 L343–360 会根据配置调整，例如非 TMA 的 paged KV 需要更多 load warp。）

**Stop 2：kernel 主体按 warp_idx 分派**

- `kernel` L903 起，分派段 L1312–1481
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_sm100.py#L1312-L1481

```python
if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
    cute.arch.setmaxregister_decrease(self.num_regs_other)
    self.load(..., mPageTable, tma_atom_Q, tma_atom_K, tma_atom_V, ..., pipeline_q, pipeline_kv, ...)
if warp_idx == self.mma_warp_id:
    tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))   # MMA warp 负责分配 TMEM
    self.mma(tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, tStS, tOtO, tOrP, pipeline_q, pipeline_kv, pipeline_s_p_o, ...)
...
if (q_stage == 2 and warp_idx <= self.softmax1_warp_ids[-1]) or ...:
    cute.arch.setmaxregister_increase(self.num_regs_softmax)     # softmax 最吃寄存器
    softmax_loop(stage=stage, tStS=tStS)
if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
    self.correction_loop(...)
```
数据流（一个 Q tile 内）：
```
load warp ──TMA──► sQ, sK_j, sV_j            (pipeline_q / pipeline_kv)
mma warp  ──tcgen05──► S_j (TMEM) = Q K_jᵀ   (pipeline_s_p_o 通知 softmax)
softmax   ── TMEM→寄存器: m_new, p = exp2(...) ──► P_j 写回 TMEM；α 写到 sScale
correction ── 读 α，若需要则 TMEM 中 O *= α
mma warp  ──tcgen05──► O (TMEM) += P_j V_j
最后 correction: O / ℓ → smem → (epilogue warp) gmem；写 LSE
```
mma warp 内部的软件流水（L1898–1899 注释）：`QK(0); [QK(i+1), PV(i)] for i in 0..n-2; PV(n-1)`——在 softmax 处理 S_i 的同时，tensor core 已经在算 S_{i+1}。两个 Q tile（q_stage=2）+ 两组 softmax warpgroup 交替，进一步把 softmax 的 exp 延迟藏在 MMA 后面。

**Stop 3：softmax warp 的一步 `softmax_step`**

- L2566–2737：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_sm100.py#L2664-L2733

关键几行：
```python
row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)        # L2668：m_new 与 α
...
sScale[thread_idx + sm_stats_index * self.m_block_size] = acc_scale           # L2680：α 交给 correction warp
...
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)                              # L2688：s*scale_log2 - m*scale_log2（packed FFMA）
softmax.apply_exp2_convert(...)                                               # exp2 并转成 bf16/fp16
for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])   # P 写回 TMEM 给 PV MMA
...
softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)                  # L2732：ℓ = α·ℓ + Σp
```
在 SM100 上**一个线程负责一整行**（`SoftmaxSm100.create` 中 `num_rows = 1`，softmax.py L316），因为 TMEM 的 32x32b load 让 128 个线程各拿到 S 的一行，行内归约不需要 shuffle。

**Stop 4：条件 rescale —— FA4 的一个关键优化**

- `softmax.py` `SoftmaxSm100.update_row_max` L374–392
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/softmax.py#L374-L392

```python
row_max_old = self.row_max[0]
row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
if cutlass.const_expr(self.rescale_threshold > 0.0):
    if acc_scale_ >= -self.rescale_threshold:     # 最大值只涨了一点点
        row_max_new = row_max_old                 # 就假装没涨：继续用旧的 m
        row_max_safe = row_max_old
        acc_scale = 1.0                           # α=1，O 不需要 rescale
self.row_max[0] = row_max_new
```
为什么可以：online softmax 里减去的 m 只要是**一个公共的基准**就行，不一定要是真正的最大值；只要 `exp2(s − m)` 不溢出，最后 `acc/ℓ` 结果完全一样。允许 m 滞后最多 `rescale_threshold` 个 log2 单位，就能跳过大多数迭代的 O rescale（O 在 TMEM 里，rescale 要读写 `[128, d_v]`，很贵）。文件头 L88–93 的 “Note [Low Precision Scaling]” 解释了与低精度 P 的饱和问题之间的权衡。

correction warp 的配合（L2882–2895）：
```python
scale = sScale[tidx + sm_stats_index * self.m_block_size]
should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0    # 整个 warp 只要有一行需要就做
if should_rescale:
    self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
```
`correction_rescale`（L3099–3147）：以 16 列为一组 `tcgen05` 把 O 从 TMEM 读到寄存器、`mul_packed_f32x2` 乘 α、再写回 TMEM。

**Stop 5：mask 与块范围**

- `block_info.py` `BlockInfo.get_n_block_min_max` L27–68：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/block_info.py#L27-L68
```python
n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
    m_idx_max = (m_block + 1) * self.tile_m
    if const_expr(self.qhead_per_kvhead_packgqa > 1):
        m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)   # pack_gqa：多个 q 头挤在 M 维
    n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
    ...
if cutlass.const_expr(self.is_split_kv):
    ...
    n_block_min = n_block_min + split_idx * num_n_blocks_per_split          # 本 split 只管自己那段
    n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
```
FA2 的 `n_block_min/max` 公式 + split 切段 + PackGQA 的行号换算，全在这一个函数里。

- `mask.py` `AttentionMask` L159 起，SM100 版 `apply_mask_sm100` L615 起，causal 分支 L713–733：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/mask.py#L713-L733
```python
causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
row_idx = tScS_t2r[0][0] + m_block * self.tile_m
if const_expr(self.qhead_per_kvhead_packgqa != 1):
    row_idx = row_idx // self.qhead_per_kvhead_packgqa
if const_expr(mask_causal):
    col_limit_right = row_idx + causal_row_offset + 1          # 本行在本 tile 内可见的列数
    if const_expr(mask_seqlen):
        col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
    ...
    mask_r2p_lambda(acc_S, lambda s: r2p_bitmask_below(col_limit_right, s), rank1=True)
```
因为一个线程持有一整行，mask 只需要一个“列上限”。`r2p` 路径把比较结果打包成位掩码再用 R2P 指令批量写谓词寄存器（AI/SM90_R2P_MASKING_SASS.md 有 SASS 分析），比逐元素比较生成的代码好得多（L663–666 的注释也吐槽了这一点）。

---

### 3.6 FA4 的 varlen 与 paged KV

**varlen：`SeqlenInfoQK.create`**

- `flash-attention/flash_attn/cute/seqlen_info.py` L83–130：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/seqlen_info.py#L83-L130
```python
offset_q = 0 if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
...
if const_expr(mSeqUsedK is not None):
    seqlen_k = mSeqUsedK[batch_idx]
else:
    seqlen_k = seqlen_k_static if const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx + 1] - offset_k
```
与 FA2 `BlockInfo` 同构：`seqused_*` 优先于 `cu_seqlens_*`。

**paged KV（TMA 路径）**：`load` 中（L1640–1703）
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_sm100.py#L1640-L1703
```python
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx=split_idx, num_splits=num_splits)
...
page_idx = (mPageTable[batch_idx, n_block_first] if const_expr(mPageTable is not None and self.use_tma_KV) else None)
if const_expr(not self.use_tma_KV):
    paged_kv_manager.load_page_table(n_block_first)
load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0
...
for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
    n_block = n_block_max - 2 - i
    page_idx = mPageTable[batch_idx, n_block] ...
    load_K(block=n_block, ...); load_V(block=n_block, ...)
```
当 `page_size == tile_n` 时，**第 n_block 个 K tile 恰好就是第 n_block 页**，所以 `page_idx = mPageTable[batch_idx, n_block]`，TMA 坐标里用 page_idx 代替 batch 维即可（`__call__` L501 的注释：paged 时 K 视为 `(page_size, d, h_k, num_pages)`）。同样倒序遍历 K 块。

**paged KV（非 TMA 路径，任意 page 大小）**：`paged_kv.py` `PagedKVManager`
- L16–156：https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/paged_kv.py#L137-L156
```python
def load_page_table(self, n_block):
    for i in cutlass.range(self.page_entry_per_thread, unroll=1):
        row = (i * self.num_threads + ...)                      # 本线程负责 tile 里的哪一行
        row_idx = n_block * self.n_block_size + row             # 该行的逻辑 token 位置
        page_idx, page_offset = divmod(row_idx + self.leftpad_k, self.page_size_divmod)   # 快速除法
        is_valid = (...) and row_idx < self.seqlen_k
        page = self.mPageTable[page_idx] if is_valid else 0
        self.tPrPage[i] = page
        self.tPrPageOffset[i] = page_offset
```
每个线程为自己要搬的行查页表，然后用 `cp.async` 从 `(page, page_offset)` 逐行拷到 shared memory——这就是“任意 page 大小（包括 1）”的代价：放弃 TMA 的整块搬运，换成按行 gather。对照 FA2：FA2 要求一个 tile 落在一页内（256 的倍数），FA4 两条路都提供了。

---

### 3.7 FA4 的 split-KV combine：`flash_fwd_combine.py`

- `FlashAttentionForwardCombine` L28 起，`kernel` L354 起，LSE 合并 L555–604，加权累加 L653–680
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/cute/flash_fwd_combine.py#L555-L680

```python
lse_max = cute.arch.warp_reduction_max(ts2rrLSE[None, None, m].load().reduce(cute.ReductionOp.MAX, ...), ...)
lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max     # In case all local LSEs are -inf
for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
    scale = cute.math.exp2(ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True)
    lse_sum_cur += scale
    ts2rrLSE[0, s, m] = scale                     # 先存未归一化的权重
lse_sum_cur = cute.arch.warp_reduction_sum(lse_sum_cur, ...)
lse_sum[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max    # 最终 LSE
inv_sum = 0.0 if (lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur) else 1.0 / lse_sum_cur
ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)   # w_s = exp(LSE_s - LSE)
...
for s in cutlass.range(thr_max_valid_split + 1, unroll=4):     # 只累加到最后一个有效 split
    ...
    tOrO[None, m, None].store(tOrO[None, m, None].load() + scale[m] * tOrO_partial[None, m, None].load().to(Float32))
```
就是 §2 的 `split_kv_attn` 后半段，外加：`max_valid_split` 跳过全空的 split（LSE=-inf），cp.async 多 stage 预取各 split 的部分 O。

---

### 3.8 FlashInfer：为 decode / paged 而生的库

- 仓库：`repos/02-attention-kernels/flashinfer`

**Stop 1：`BatchDecodeWithPagedKVCacheWrapper`（plan / run 两段式）**

- `flashinfer/decode.py` L1194–1258（docstring 示例），`plan` L1893 起
- https://github.com/flashinfer-ai/flashinfer/blob/74ca44b318a3be013649baabca698099d0862e3c/flashinfer/decode.py#L1194-L1258

```python
>>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
>>> kv_page_indptr = torch.tensor([0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0")
>>> # 1 <= kv_last_page_len <= page_size
>>> kv_last_page_len = torch.tensor([1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0")
>>> kv_cache_at_layer = [torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim, ...) ...]
>>> decode_wrapper.plan(kv_page_indptr, kv_page_indices, kv_last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, ...)
>>> o = decode_wrapper.run(q, kv_cache)        # 每层复用同一份 plan
```
与 FA 的页表格式对比：FA 用矩形的 `block_table [B, max_pages]` + `seqused_k`；FlashInfer 用 **CSR 格式**：`indptr [B+1]` + 扁平的 `indices [nnz_pages]` + `last_page_len [B]`（第 b 个请求的页是 `indices[indptr[b]:indptr[b+1]]`，长度 = `(页数−1)·page_size + last_page_len[b]`）。`plan` 在 CPU 上根据各请求长度做负载均衡（决定 split 方式、写 workspace），`run` 只发 kernel——一次 plan，32 层复用，并且对 CUDA Graph 友好。

**Stop 2：merge 状态 —— 与 §1.3 完全一样**

- `include/flashinfer/attention/state.cuh` `state_t` L28–84：https://github.com/flashinfer-ai/flashinfer/blob/74ca44b318a3be013649baabca698099d0862e3c/include/flashinfer/attention/state.cuh#L28-L84
```cpp
vec_t<float, vec_size> o;   /* the weighted sum of v: exp(pre-softmax logit - m) * v / d  */
float m;                    /* maximum value of pre-softmax logits */
float d;                    /* sum of exp(pre-softmax logits - m) */
void merge(const vec_t<float, vec_size>& other_o, float other_m, float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    float f1 = math::ptx_exp2(max(m_prev - m, -math::inf));
    float f2 = math::ptx_exp2(max(other_m - m, -math::inf));
    d = d_prev * f1 + other_d * f2;
    for (size_t i = 0; i < vec_size; ++i) { o[i] = o[i] * f1 + other_o[i] * f2; }
}
```
`(o, m, d)` 就是我们的 `(acc, m, ℓ)`；online softmax 的每一步、split-KV 合并、cascade attention（共享前缀算一次 + 各自后缀，再 merge）都是这个 `merge`。Python 侧暴露为 `flashinfer/cascade.py` 的 `merge_state`（L42）/`merge_states`（L170）。decode kernel 本体见 `include/flashinfer/attention/decode.cuh` 的 `BatchDecodeWithPagedKVCacheKernel`（L624）。

### 3.9 xformers memory-efficient attention

- `repos/02-attention-kernels/xformers/xformers/ops/fmha/__init__.py` L6–60：https://github.com/facebookresearch/xformers/blob/4f701fad69a6308e6b9f0c3f7863c40c881b30a7/xformers/ops/fmha/__init__.py#L6-L60

当前 checkout 中，fmha 的实现**已迁移到 `mslk` 包**，xformers 只做 re-export：
```python
# The fmha implementation has moved to the mslk package. This package and its
# submodules re-export mslk symbols to preserve the xformers.ops.fmha API.
if importlib.util.find_spec("mslk"):
    from mslk.attention.fmha import (... memory_efficient_attention, MemoryEfficientAttentionCutlassOp,
        MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionSplitKCkOp, merge_attentions, ...)
    from mslk.attention.fmha.dispatch import (_dispatch_bw, _dispatch_fw, ...)
```
所以在本仓库里读不到 xformers 的 kernel 源码，但接口设计仍值得看：
- `memory_efficient_attention(q, k, v, attn_bias=...)` 是统一入口，`_dispatch_fw` 按输入（dtype、head_dim、bias 类型、硬件）在多个后端（CUTLASS、FlashAttention、CK、Triton split-K…）中挑一个。
- `attn_bias.py`（L21–49）列出了**结构化 bias**：`LowerTriangularMask`（causal）、`BlockDiagonalMask`（varlen，等价 `cu_seqlens`）、`BlockDiagonalCausalWithOffsetPaddedKeysMask`（decode/KV cache）、`PagedBlockDiagonal*`（paged KV）。文件 docstring（L5–19）说明了动机：用类而不是稠密张量表示 mask，kernel 就不必从显存读 bias，还能预知哪些块可以跳过——和 FA 的 `causal`/`cu_seqlens` 参数是同一种思路的不同 API。
- `merge_attentions` 对应 §1.3 的 LSE 合并。

---

## 4. 不同实现对比

| | Triton 教程 | FA2 (CUDA/CuTe C++) | FA3 (Hopper, `hopper/`) | FA4 (CuTeDSL) | FlashInfer | xformers |
|---|---|---|---|---|---|---|
| 语言 | Triton Python | CUDA C++ 模板 | CUDA C++ (CUTLASS 3) | Python DSL → JIT cubin | CUDA C++ + JIT | 现转发到 mslk |
| 目标 GPU | 通用 | SM80+ | SM90 | SM80/90/100/110/120（主力 SM90/SM100） | SM80+ 多架构 | 多后端 |
| S 存放 | 寄存器 | 寄存器 | 寄存器 | **TMEM**（SM100） | 寄存器 | — |
| 搬运方式 | tensor descriptor/TMA（新硬件） | cp.async | TMA + warp specialization | TMA（或 cp.async gather）+ 16 warp 角色 | cp.async / TMA | — |
| softmax 重叠 | 编译器 warp_specialize | 软件流水 | 2 个消费者 warpgroup ping-pong | 2 个 softmax WG + 单独 correction WG + 条件 rescale | — | — |
| varlen | 无（教程） | `cu_seqlens` + `seqused_k` | 同 FA2 + `seqused_q` | 同 FA3 | `indptr` | `BlockDiagonalMask` |
| paged KV | 无 | `block_table`，page%256==0 | `page_table`，任意 page（`hopper/flash_attn_interface.py` L1020：“page_block_size can be arbitrary”） | `page_table`：page==tile_n 走 TMA，否则 gather | CSR `indptr/indices/last_page_len` | `PagedBlockDiagonal*` |
| split-KV | 无 | splitkv kernel + combine | 有（scheduler_metadata） | `num_splits` + `FlashAttentionForwardCombine` | plan 时负载均衡 + merge | `SplitK` 后端 + `merge_attentions` |
| 反向 | 有（dkdv/dq 分离，无原子） | 有（dQ 原子累加） | 有 | 有（sm90/sm100） | 主要推理 | 有 |

---

## 5. 常见困惑 / 坑

1. **“FlashAttention 减少了计算量吗？”** 没有。前向 FLOPs 一样，反向还多重算一次 S。它减少的是 HBM 读写和 O(N²) 的显存。
2. **为什么 FA 的 K 循环是倒序的？** 让需要 mask 的对角线块最先处理，后续迭代就是无分支的快速路径（FA2 L297–301 注释说明了把需要 mask 的迭代单独分出来，倒序正好让它们排在最前）；FA4 的 load warp 同样从 `n_block_max - 1` 开始。
3. **exp 和 exp2 混用**：FA2 kernel 内全部是 `exp2f`，但 `normalize_softmax_lse` 产出的 LSE 是自然对数（`row_max * softmax_scale + __logf(sum)`）；Triton 教程存的 `M` 是 log2 域（`m_i += tl.math.log2(l_i)`）。读不同实现的 LSE 前先确认底数。
4. **全 -inf 行**：causal 且 `seqlen_q > seqlen_k`、或 local window 时，某些行可能一个有效 key 都没有。FA2 `Check_inf`、FA4 `check_inf`/`row_max_safe`、FlashInfer `max(m_prev - m, -inf)` 都在防 `(-inf)-(-inf)=NaN`；输出约定为 0，LSE 为 +inf（FA2 非 split）或 -inf（split，权重为 0）。
5. **causal 对齐右下角**：`seqlen_q ≠ seqlen_k` 时 FA 的 causal 是“最后一个 query 看见全部 key”，适配 decode/续算；PyTorch 的 `torch.tril` 是左上对齐。§2 toy 用的是 FA 的约定。
6. **P 的精度**：P 在进第二个 GEMM 前转成 fp16/bf16。FA4 甚至用 `max_offset` 把 P 放大到低精度格式的上半区间以减少下溢（`flash_fwd_sm100.py` L88–93）。所以 FA 和 fp32 参考实现之间有 1e-3 量级误差是正常的。
7. **paged KV 的 NaN 陷阱**：FA4 docstring L4231–4235：页内超出 `seqused_k` 的行虽被 mask，但 `0 × NaN = NaN`，这些行必须是有限值。用 `torch.empty` 分配 KV 池时要注意。
8. **split-KV 不是越多越好**：每个 split 都要写 fp32 的部分 O（`[num_splits, b, h, s_q, d]`），combine 再读一遍；FA2/FA4 的启发式都是“SM 已经填满就不切”。vLLM 只在 CUDA Graph 下显式设置 `num_splits`（`flash_attn.py` L856–860 的注释说明了显存原因）。
9. **CLAUDE.md 与代码的出入**：FA4 的 Hopper 类现在在 `flash_fwd_sm90.py`，`flash_fwd.py` 是 Base + Sm80（见 §3.3 末尾）。
10. **FA4 第一次调用很慢**：那是 JIT 编译。`compile_key` 里任何一项变化（如换 head_dim、开关 causal、换 mask_mod）都会触发新的编译；可用 `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` 打开磁盘缓存。

---

## 6. 自测题 + 动手练习

**自测题**
1. 写出 online softmax 中 `α` 的表达式，并证明处理完所有块后 `acc/ℓ` 等于精确的 softmax·V。
2. N=4096、d=128、B_r=B_c=128 时，FA 前向中每个 K 块从 HBM 被读几次？标准实现需要多少字节的 S 中间结果（fp16，单头）？
3. FA2 `softmax_rescale_o` 里，为什么 `reduce_max` 要做 `quad_allreduce` 而 `reduce_sum` 不做？
4. Triton 教程的反向为什么把 dK/dV 和 dQ 分成两个循环？FA2 是怎么处理 dQ 的？
5. FA4 的 `rescale_threshold` 为什么不改变数学结果？什么时候会带来精度问题？
6. decode 时 batch=4、Hq=32、seqlen_k=32k、H100 132 个 SM：不切分时有多少个 CTA（FA2 split kernel，kBlockM=64）？FA2 的 `num_splits_heuristic` 大致会选几？
7. FA4 什么情况下 paged KV 走 TMA，什么情况下走 `PagedKVManager`？为什么？
8. 给出 split-KV 合并公式，并说明为什么 LSE=-inf 的 split 权重为 0。
9. FlashInfer 的 `indptr/indices/last_page_len` 如何转换成 FA 的 `block_table/seqused_k`？

**动手练习**
1. 运行 §2 的代码，改变 `Br/Bc`（包括不整除的大小），确认结果不变；把 `torch.exp` 改成 `exp2` + `scale_log2` 版本。
2. 给 `flash_attn_tiled` 加 `window_size=(left, right)`（local attention），并实现 `n_block_min` 跳块；对照 FA2 L89–95 和 FA4 `get_n_block_min_max`。
3. 实现 toy 版反向：只用存下的 `LSE`、`O`、`dO` 重算 P，算出 dQ/dK/dV，与 `torch.autograd` 对拍（参考 §1.6 和 Triton `_attn_bwd_dkdv/_attn_bwd_dq`）。
4. 把 03 篇的 `ToyPagedKV` 与本篇结合：写一个 `paged_flash_decode(q, k_pages, v_pages, block_table, seqlens, num_splits)`，每个 split 按页循环做 online softmax，最后用 LSE 合并。
5. 在有 GPU 的机器上，运行 Triton 教程的 `bench_flash_attention`（L755），再对比 `flash_attn.cute.flash_attn_func`；用 `CUTE_DSL_KEEP_PTX=1` 看 FA4 生成的 PTX 中 `tcgen05.mma` 与 `ex2.approx` 指令。
6. 用 FlashInfer 的 `merge_state` 验证：把一段 KV 拆成两半分别算 attention（`return_lse`），合并结果应与整段一致。
