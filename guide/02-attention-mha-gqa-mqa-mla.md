# 源码导读 02：注意力的四种头 —— MHA / MQA / GQA / MLA

> 目标读者：懂 attention 概念，想弄清"为什么要少放几个 KV 头 / 怎么把 KV 压缩成低秩 / absorb 到底 absorb 了什么"。
> 代码标识符保留英文，解释用中文。所有行号、路径、符号都经过实际阅读验证。

---

## 0. 要解决的问题 / 一句话直觉

**要解决的问题**：自回归解码时，每生成一个 token 都要读一遍**全部历史 KV cache**。
KV cache 的大小 = `2(K和V) × 层数 × KV头数 × head_dim × 序列长度 × dtype字节`。
长上下文 + 大 batch 时，KV cache 会吃光显存，而且**解码是 memory-bound**（瓶颈在搬 KV，不在算）。
所以核心矛盾是：**在尽量不掉精度的前提下，把 KV cache 做小**。

**一句话直觉**：
- MHA：每个 query 头配一套 KV 头（最准，最占）。
- MQA：所有 query 头共享 1 套 KV 头（最省，略掉点）。
- GQA：折中，每组 query 头共享 1 套 KV（主流）。
- MLA：不存"KV 头"，只存一个**低秩压缩的潜向量**，用的时候再投影还原（DeepSeek 的杀手锏）。

---

## 1. 核心原理：KV cache 的尺寸账

设：层数 `L`、query 头数 `Hq`、KV 头数 `Hkv`、head_dim `d`、序列长度 `S`、每元素字节 `b`。

**每 token 每层的 KV cache 元素数** = `2 × Hkv × d`。全模型全序列：

```
KV_cache_bytes = 2 * L * Hkv * d * S * b
```

- **MHA**：`Hkv = Hq`。
- **MQA**：`Hkv = 1`。
- **GQA**：`Hkv = Hq / g`（`g` 是每组头数，即 `num_key_value_groups`）。
- **MLA**：不走这个公式。它每 token 每层只存 `kv_lora_rank + qk_rope_head_dim` 个元素（下面详述）。

形状记号：`q: [B, Hq, S, d]`，`k,v: [B, Hkv, S, d]`。注意力前要把 `k,v` 的 `Hkv` **复制**到 `Hq`
（`repeat_kv`），这样每个 query 头都有对应的 K/V。

---

## 2. 最小实现：一份代码涵盖 MHA/MQA/GQA

```python
import torch, torch.nn.functional as F

def repeat_kv(x, n_rep):                       # x: [B, Hkv, S, d]
    if n_rep == 1:
        return x
    B, Hkv, S, d = x.shape
    x = x[:, :, None, :, :].expand(B, Hkv, n_rep, S, d)   # 插一维再广播
    return x.reshape(B, Hkv * n_rep, S, d)     # [B, Hq, S, d]

def attn(q, k, v, scale):                      # q:[B,Hq,S,d]  k,v:[B,Hkv,S,d]
    Hq, Hkv = q.shape[1], k.shape[1]
    k = repeat_kv(k, Hq // Hkv)                # GQA/MQA: 把 KV 头复制上来
    v = repeat_kv(v, Hq // Hkv)
    scores = (q @ k.transpose(-2, -1)) * scale          # [B, Hq, S, S]
    scores = scores.masked_fill(torch.triu(torch.ones(scores.shape[-2:], device=q.device), 1).bool(), float("-inf"))
    o = F.softmax(scores, -1) @ v              # [B, Hq, S, d]
    return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

# MHA: Hkv = Hq ;  MQA: Hkv = 1 ;  GQA: 1 < Hkv < Hq，且 Hq % Hkv == 0
```

> 注意 `repeat_kv` 用 `expand`（不复制内存，只是广播视图）再 `reshape`。真正的 FlashAttention 内核
> 连这个物化都省了——直接让多个 query 头在 kernel 里读同一份 KV（`pack_gqa`）。

---

## 3. 源码导读（一）：repeat_kv 与 GQA

**Stop 1 — `repeat_kv`（Llama）**
`repos/00-basics/transformers/src/transformers/models/llama/modeling_llama.py`，L168-177
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L168-L177

```python
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape   # [B, Hkv, S, d]
    if n_rep == 1:
        return hidden_states                                           # L175: MHA 时不动
    hidden_states = hidden_states[:, :, None, :, :].expand(            # L176: [B,Hkv,1,S,d]→[B,Hkv,n_rep,S,d]
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)  # L177: [B, Hq, S, d]
```

逐行：L173 解包出 `Hkv`。L175 当 `n_rep==1`（即 MHA）直接返回，省一次 reshape。
L176 在第 2 维后插一维 `None` 再 `expand`——`expand` **不分配新内存**，只是把 stride 设成 0 的广播视图。
L177 `reshape` 把 `Hkv` 和 `n_rep` 两维合并成 `Hq`。
**为什么这样写**：`n_rep = Hq / Hkv`，GQA 本质是"每 `n_rep` 个 query 头共享 1 个 KV 头"。
用 expand+reshape 而不是 `repeat_interleave`，在被 `torch.compile` 捕获时更高效。

**Stop 2 — `num_key_value_groups` 怎么来的**
`.../llama/modeling_llama.py`，L214（定义）、L190-191（使用）
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L214-L214

```python
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads   # L214 = Hq / Hkv
...
key_states   = repeat_kv(key,   module.num_key_value_groups)    # L190
value_states = repeat_kv(value, module.num_key_value_groups)    # L191
```

逐行：L214 就是 `n_rep`。注意 q/k/v 三个投影的输出维度**本来就不同**——
看 `LlamaAttention.__init__`：`q_proj` 输出 `num_attention_heads * head_dim`（L219-221），
而 `k_proj`/`v_proj` 只输出 `num_key_value_heads * head_dim`（L222-227）。**KV 头少，投影矩阵就小，cache 就小**。

**Stop 3 — GQA 下的 attention**
`.../llama/modeling_llama.py`，`eager_attention_forward` L180-202（见导读 01 已逐行）。
关键：L190-191 先把 `Hkv` 展开到 `Hq`，之后就是普通 MHA 的 `QK^T → softmax → PV`。

**配置样例**（`configuration_llama.py`）：`num_key_value_heads` 默认 = `num_attention_heads`（L89-90，即 MHA）。
Llama-3-8B 的真实配置是 `Hq=32, Hkv=8, d=128, L=32`，即 `n_rep=4` 的 GQA。

### KV cache 大小对比（Llama-3-8B 规模，fp16 b=2 字节，S=8192）

| 方案 | `Hkv` | 每 token 每层 KV 元素 `2·Hkv·d` | 整模型 KV cache（`2·L·Hkv·d·S·b`） |
|---|---|---|---|
| MHA | 32 | 8192 | `2·32·32·128·8192·2` ≈ **4.0 GiB** |
| GQA (n_rep=4) | 8 | 2048 | ≈ **1.0 GiB** |
| MQA | 1 | 256 | ≈ **128 MiB** |

（计算：MHA = 2×32层×32头×128×8192×2B = 4,294,967,296 B = 4 GiB。GQA 是其 1/4，MQA 是其 1/32。）

---

## 4. 核心原理：MLA —— 把 KV 压成一个低秩潜向量

MLA（Multi-head Latent Attention）的三个关键想法：

**(a) 低秩 KV 压缩**：不直接存每头的 K、V，而是用一个下投影 `W_DKV` 把 `hidden` 压成一个
**共享的潜向量** `c_KV ∈ R^{kv_lora_rank}`（DeepSeek-V3 里 `kv_lora_rank=512`）。
**KV cache 里只存这个 `c_KV`**。用的时候再用上投影 `W_UK`、`W_UV` 把它"还原"成各头的 K/V。

**(b) 解耦 RoPE（decoupled RoPE）**：问题来了——RoPE 要作用在 K 上，但如果 K 是从 `c_KV` 现场还原的，
RoPE 和"absorb 技巧"会冲突（旋转矩阵依赖位置，没法预先吸收进权重）。
解法：把 head_dim 拆成两段——
- `qk_nope_head_dim`（不加 RoPE 的"内容"部分，`=128`）走低秩压缩；
- `qk_rope_head_dim`（加 RoPE 的"位置"部分，`=64`）单独算，且**所有头共享同一份** `k_pe`。

所以每 token 每层 cache = `c_KV (kv_lora_rank=512)` + `k_pe (qk_rope_head_dim=64)` = **576 个元素**。

**(c) absorb（权重吸收）技巧**：解码时不想真的把 `c_KV` 还原成大的 K/V（那样又占内存又费算力）。
数学上：
```
score = q_nope · k_nope = q_nope · (W_UK · c_KV) = (q_nope · W_UK) · c_KV
```
于是把 `W_UK` **预先吸收进 query 侧**：先算 `q_nope @ W_UK` 得到一个 `kv_lora_rank` 维的 query，
直接和缓存的 `c_KV` 点积。同理输出侧把 `W_UV` 吸收进 `o_proj`。
**结果：解码时完全不物化大 K/V，只和 512 维的潜向量打交道。**

形状（DeepSeek-V3，单层）：
```
hidden x:           [B, S, 7168]
c_KV = W_DKV(x):    [B, S, 512]           ← 存进 cache
k_pe = W_KR(x):     [B, S, 64]  (RoPE)    ← 存进 cache (所有头共享)
q (每头):           nope [B,S,Hq,128] + pe [B,S,Hq,64]
```

---

## 5. 最小实现：MLA 的 naive 与 absorb 两条路

```python
import torch, torch.nn.functional as F

class MiniMLA(torch.nn.Module):
    def __init__(self, dim, n_heads, kv_lora=512, nope=128, rope=64, vdim=128):
        super().__init__()
        self.n, self.nope, self.rope, self.vdim = n_heads, nope, rope, vdim
        self.W_DKV = torch.nn.Linear(dim, kv_lora + rope, bias=False)   # 下投影: 内容潜向量 + 共享位置
        self.W_UKV = torch.nn.Linear(kv_lora, n_heads * (nope + vdim), bias=False)  # 上投影(还原K_nope和V)
        self.W_Q   = torch.nn.Linear(dim, n_heads * (nope + rope), bias=False)
        self.W_O   = torch.nn.Linear(n_heads * vdim, dim, bias=False)
        self.scale = (nope + rope) ** -0.5

    def forward(self, x, cos, sin, absorb=False):
        B, S, _ = x.shape
        q = self.W_Q(x).view(B, S, self.n, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)                    # 只旋转位置部分
        c = self.W_DKV(x)
        c_kv, k_pe = c.split([self.W_UKV.in_features, self.rope], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin)       # 共享的位置 K

        if not absorb:   # naive: 真的把 c_kv 还原成每头 K/V
            kv = self.W_UKV(c_kv).view(B, S, self.n, self.nope + self.vdim)
            k_nope, v = kv.split([self.nope, self.vdim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n, -1)], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.scale
        else:            # absorb: 把 W_UK 吸收进 q，直接和 c_kv 点积
            W = self.W_UKV.weight.view(self.n, self.nope + self.vdim, -1)
            q_absorb = torch.einsum("bshd,hdc->bshc", q_nope, W[:, :self.nope])   # q_nope @ W_UK
            scores = (torch.einsum("bshc,btc->bsht", q_absorb, c_kv)             # 和潜向量点积
                      + torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.scale

        p = scores.softmax(-1)
        if not absorb:
            o = torch.einsum("bsht,bthd->bshd", p, v)
        else:            # 输出侧吸收 W_UV
            W = self.W_UKV.weight.view(self.n, self.nope + self.vdim, -1)
            ctx = torch.einsum("bsht,btc->bshc", p, c_kv)                        # 先和潜向量加权
            o = torch.einsum("bshc,hdc->bshd", ctx, W[:, -self.vdim:])           # 再 @ W_UV
        return self.W_O(o.flatten(2))
```

> 两条路**数学完全等价**，只是 absorb 把矩阵乘的结合律换了个顺序，使解码时无需物化大 K/V。

---

## 6. 源码导读（二）：DeepSeek-V3 官方 MLA（naive vs absorb 同一函数内对照）

这是全篇最值得精读的一段：**同一个 `forward` 里并排写了 naive 和 absorb 两条路**，是对照学习的最佳材料。

**Stop 1 — MLA 的投影定义与两种 cache**
`repos/01-attention/DeepSeek-V3/inference/model.py`，`MLA.__init__` L412-444
🔗 https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L412-L444

```python
self.wkv_a  = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)   # L430: 下投影→[c_KV | k_pe]
self.kv_norm = RMSNorm(self.kv_lora_rank)                                   # L431: 潜向量归一化
self.wkv_b  = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # L432: 上投影
self.wo     = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)   # L433: 输出投影
...
if attn_impl == "naive":
    self.register_buffer("k_cache", torch.zeros(..., self.n_local_heads, self.qk_head_dim))  # L440: 存还原后的大K
    self.register_buffer("v_cache", torch.zeros(..., self.n_local_heads, self.v_head_dim))   # L441: 存大V
else:  # absorb
    self.register_buffer("kv_cache", torch.zeros(..., self.kv_lora_rank))     # L443: 只存潜向量 c_KV
    self.register_buffer("pe_cache", torch.zeros(..., self.qk_rope_head_dim)) # L444: 只存共享位置 k_pe
```

逐行关键对照 L440-444：**naive 存的是还原后的、每头独立的大 K/V**（`n_local_heads × qk_head_dim`），
**absorb 只存 `kv_lora_rank(512) + qk_rope_head_dim(64) = 576`**，和头数无关。
这就是 MLA 省 cache 的本质——absorb 路径的 cache **与头数无关**。
L430 一个 `wkv_a` 同时产出内容潜向量和共享位置向量（`split` 开）。

**Stop 2 — Q 的处理与 RoPE 拆分**
`.../inference/model.py`，`MLA.forward` L459-470
🔗 https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L459-L470

```python
if self.q_lora_rank == 0:
    q = self.wq(x)                                         # L462: 小模型直接投影
else:
    q = self.wq_b(self.q_norm(self.wq_a(x)))               # L464: 大模型 Q 也走低秩(q_lora_rank=1536)
q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # L466: 拆内容/位置
q_pe = apply_rotary_emb(q_pe, freqs_cis)                   # L467: 只旋转位置部分
kv = self.wkv_a(x)                                         # L468: 下投影
kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # L469: 拆 c_KV / k_pe
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)      # L470: 共享位置 K 旋转(unsqueeze 出 head 维=1)
```

逐行：L464 连 Q 都用了低秩（`q_lora_rank=1536`），进一步省激活显存。
L466-467 是**解耦 RoPE 的核心**——只有 `q_pe`（64 维）被旋转，`q_nope`（128 维）不转。
L470 的 `k_pe` 形状是 `[B,S,1,64]`，**1 代表所有头共享**（这就是为什么 cache 里 k_pe 不乘头数）。

**Stop 3 — naive 路径（物化大 K/V，便于理解）**
`.../inference/model.py`，L471-479、L491-492
🔗 https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L471-L479

```python
if attn_impl == "naive":
    q = torch.cat([q_nope, q_pe], dim=-1)                  # L472: 拼回完整 q [.,.,H,192]
    kv = self.wkv_b(self.kv_norm(kv))                      # L473: 上投影还原 → [.,.,H*(128+128)]
    kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)  # L475
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)   # L476: 共享 k_pe 广播到每头
    self.k_cache[:bsz, start_pos:end_pos] = k              # L477: 缓存还原后的大 K
    self.v_cache[:bsz, start_pos:end_pos] = v              # L478
    scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale  # L479
...
    x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])   # L492: 普通 PV
```

逐行：naive 就是"把 `c_KV` 用 `wkv_b` 还原成每头 K/V（L473），再走标准 MHA”。
L476 把共享的 `k_pe` 用 `expand` 广播到每个头（视图，不复制），拼在 `k_nope` 后面凑成 192 维的完整 K。
**naive 直观但 cache 大、解码慢**，适合 prefill 或理解用。

**Stop 4 — absorb 路径（权重吸收，解码的真正用法）**
`.../inference/model.py`，L480-487、L493-496
🔗 https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L480-L496

```python
else:
    wkv_b = self.wkv_b.weight ...                          # L481: 拿到上投影权重
    wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)   # L482: [H, 256, 512]
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])  # L483: q_nope @ W_UK → [.,.,H,512]
    self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)       # L484: 只缓存潜向量 c_KV
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)        # L485: 只缓存共享 k_pe
    scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +   # L486: q_absorb · c_KV
              torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale  # L487: q_pe · k_pe
...
    x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])   # L494: P 和潜向量加权 → [.,.,H,512]
    x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])          # L495: 再 @ W_UV → [.,.,H,128]
```

逐行（**本篇最关键**）：
- L483：把 `W_UK`（= `wkv_b` 的前 128 维）**吸收进 query**——`q_nope`（128维）左乘 `W_UK` 变成 512 维的
  `q_absorb`。这样它能**直接**和缓存的 512 维 `c_KV` 点积。
- L486：score 分两部分——内容分 `q_absorb · c_KV`（einsum `bshc,btc`）+ 位置分 `q_pe · k_pe`（`bshr,btr`）。
  **注意内容分里根本没出现大 K**，全程只碰 512 维潜向量。
- L494-495：输出侧对称——先用 attention 权重 `P` 对 `c_KV` 加权（L494，得 512 维），
  再左乘 `W_UV`（= `wkv_b` 的后 128 维，L495）还原成 `v_head_dim`。`W_UV` 被吸收进了输出路径。

**为什么 absorb 能省**：矩阵乘结合律 `(q W_UK) c = q (W_UK c)`。把 `W_UK` 提前乘到 q 侧，
就不必对每个历史 token 都做 `W_UK c`（那是物化大 K）。解码时历史很长，这个省法是数量级的。

**Stop 5 — 配置参数**
`.../inference/model.py`，`ModelArgs` L74-79（默认）；`configs/config_671B.json`（V3 真实值）
🔗 https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L74-L79

```python
q_lora_rank: int = 0          # L75  (671B 配置里是 1536)
kv_lora_rank: int = 512       # L76  ← KV cache 的主体
qk_nope_head_dim: int = 128   # L77  内容部分
qk_rope_head_dim: int = 64    # L78  位置部分(共享, 加RoPE)
v_head_dim: int = 128         # L79
```

---

## 7. 源码导读（三）：HuggingFace transformers 的 MLA（训练友好版）

HF 的实现偏"训练/通用"，所以它走的是**类似 naive 的 expand_kv**，但 cache 存压缩态。

**Stop 1 — `DeepseekV3Attention.__init__`**
`repos/00-basics/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`，L388-398
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L388-L398

```python
self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=...)  # L388: 下投影(名字点明 MQA 式共享)
self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)    # L393
self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)  # L394: 上投影
```

逐行：`kv_a_proj_with_mqa` 的命名直接告诉你——压缩后的 KV 是**MQA 式共享**（单份潜向量供所有头）。
`config.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`（configuration L123），
`config.head_dim = qk_rope_head_dim`（L124，决定 RoPE 作用维度）。

**Stop 2 — forward：在压缩态读写 cache**
`.../deepseek_v3/modeling_deepseek_v3.py`，L443-464
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L443-L464

```python
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)   # L443: 拆Q
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)                     # L445: 下投影
kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # L446
kv_nope = self.kv_a_layernorm(kv_nope)                                     # L447
kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)       # L449: 视作单头 4D
k_rot   = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)     # L450: 共享位置
q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)                # L456: 只旋转位置段
if past_key_values is not None:
    kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)  # L460: cache 存压缩态!
query_states = torch.cat((q_pass, q_rot), dim=-1)                          # L462
key_states, value_states = self.expand_kv(kv_nope, k_rot)                   # L464: 还原成大K/V
```

逐行：**L460 是重点**——`past_key_values.update` 存的是 `kv_nope`（512）和 `k_rot`（64）的**压缩态**，
所以 cache 很小；但 L464 `expand_kv` 又把它还原成大 K/V 走标准 attention（没做 absorb）。
L449-450 把潜向量 view 成"单头"的 4D，是为了复用通用 Cache 层（它假定 `[B,H,S,d]`）。

**Stop 3 — `expand_kv` 的还原**
`.../deepseek_v3/modeling_deepseek_v3.py`，L408-425
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L408-L425

```python
kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)          # L417: 上投影还原
k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)  # L418
k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)                          # L419: 共享位置广播到每头
key_states = kv_nope.new_empty(*kv_nope.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)  # L422
key_states[..., : self.qk_nope_head_dim].copy_(k_nope)                      # L423: 前段=内容K
key_states[..., self.qk_nope_head_dim :].copy_(k_rot)                       # L424: 后段=共享位置K
```

逐行：和 DeepSeek 官方 naive 路径（model.py L473-476）一一对应。L419 `expand` 把单份 `k_rot` 广播到每头。
**HF 版本偏重正确性与可训练性**，解码时的 absorb 优化交给 vLLM/SGLang/FlashMLA 等推理框架做。

**Stop 4 — 还原后走通用 `eager_attention_forward`**
`.../deepseek_v3/modeling_deepseek_v3.py`，`repeat_kv` L225-234、`eager_attention_forward` L237-259
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L237-L259

和 Llama 的同名函数几乎一致（L248 `QK^T`，L251 softmax fp32，L253 `PV`）。MLA 还原后就是普通 attention。

---

## 8. 源码导读（四）：FlashMLA 内核接口 —— 工程上怎么用

FlashMLA 是把上面 absorb 的 decode 路径写成高性能 CUDA/NPU kernel 的库。
注意：该仓库最新 commit 已移除 Hopper 支持、改了 FP8/FP4 KV cache 格式（见 README 顶部 Breaking change）。

**Stop 1 — decode 接口 `flash_mla_with_kvcache`**
`repos/01-attention/FlashMLA/flash_mla/flash_mla_interface.py`，L59-124
🔗 https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/flash_mla/flash_mla_interface.py#L59-L124

关键签名与文档（引自源码 docstring）：
```python
def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                           tile_scheduler_metadata, num_splits=None, softmax_scale=None,
                           causal=False, is_fp8_kvcache=True, indices=None, ...):
    # q:        (batch, seq_len_q, num_heads_q, head_dim)  head_dim 必须 512   (L80-)
    # head_dim_v: 必须 512   (L89)
    # causal:   必须 False，只支持 sparse attention   (L99, L132 assert)
    # 返回 out: (batch, seq_len_q, num_heads_q, head_dim_v)  (L123)
    #     lse: (batch, num_heads_q, seq_len_q), fp32         (L124)
```

逐行要点：
- `head_dim == 512`（L80 区注释）正是 MLA **absorb 后的潜向量维度**（`kv_lora_rank=512`）——
  kernel 直接吃压缩态的 q（已吸收 `W_UK`）和 `c_KV`，印证了第 6 节的 absorb 数学。
- `causal` 必须 `False`（L132 的 `assert not causal`），因为这是 **DSA（DeepSeek Sparse Attention）**
  的 top-k 稀疏路径，由 `indices`（L70）指定每个 query 实际 attend 哪些 token。
- `block_table` + `cache_seqlens`：**paged KV cache**（把 KV 切成固定大小的 page，像虚拟内存），
  是长上下文推理的标配。
- `is_fp8_kvcache=True`（L68）：KV cache 用 FP8 存，进一步省一半显存（README 详述 528/288 字节格式）。

**Stop 2 — prefill 接口 `flash_mla_sparse_fwd`**
`.../flash_mla/flash_mla_interface.py`，L187-222
🔗 https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/flash_mla/flash_mla_interface.py#L187-L222

README 给出的等价 PyTorch（节选）直接说明 kernel 算什么：
```python
focused_kv = kv[indices]                       # 按 top-k indices 只 gather 选中的 KV
P = (Q @ focused_kv.transpose(-1,-2)) * sm_scale
P = P.masked_fill(invalid, -inf)
out = softmax(P) @ focused_kv
```
**为什么稀疏**：长上下文里大部分历史 token 对当前 query 贡献极小，DSA 用一个轻量模块选出 top-k 个相关 token，
kernel 只对这 k 个算 attention，把 `O(S)` 降到 `O(topk)`。

---

## 9. 源码导读（五）：flash-attention 本仓库的 MLA / qv 支持

本仓库（FA4，CuteDSL）通过 `qv` 这个可选参数支持 MLA-absorbed 形状。

**Stop 1 — absorb 公式的官方注释**
`flash-attention/flash_attn/cute/interface.py`，L4222-4225
```python
# qv: we write the MLA weight absorbed formula as
#     O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
#     where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache.
```
逐行：这正是第 6 节 DeepSeek absorb 路径（model.py L486）的 kernel 版——
`Q@K.T` 是位置分（`q_pe · k_pe`），`Qv@V.T` 是内容分（`q_absorb · c_KV`），两者相加再 softmax。
`V = kv_cache` 即 512 维潜向量，印证"内容 K 和 V 共用同一个潜向量"。

**Stop 2 — 支持的 MLA 形状校验**
`flash-attention/flash_attn/cute/interface.py`，`_validate_head_dims` L121-137
```python
is_deepseek_shape = head_dim == 192 and head_dim_v == 128                       # L123: naive(192=128+64)
is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512  # L124: absorbed
```
逐行：L123 的 `(192, 128)` 是 **naive 形状**（`qk_head_dim=128+64=192`, `v_head_dim=128`）；
L124 的 `head_dim_v == 512` 是 **absorbed 形状**（和 FlashMLA 的 512 一致）。

**Stop 3 — qv 参数与 scale**
`flash-attention/flash_attn/cute/interface.py`，`_flash_attn_fwd` 签名 L682、scale L852-853
```python
qv: Optional[torch.Tensor] = None,                            # L682
...
softmax_scale = (1.0 / math.sqrt(head_dim) if qv is None or q is None
                 else 1.0 / math.sqrt(head_dim + head_dim_v)) # L852-853: 有qv时 scale 用 (head_dim+head_dim_v)
```
逐行：L852-853 的 scale 规则呼应 absorb 公式——position 分走 `head_dim`、content 分走 `head_dim_v`，
合并打分所以归一化用两者之和。`flash_attn_func(q, k, v, qv=...)` 的 `qv` 即 `q_nope`，`q` 即 `q_pe`。

对应测试：`flash-attention/tests/cute/test_flash_attn_mla.py`（L139 `qv_ref`、L161 `shared_kv` 路由、
L348 `qv = randn(..., 512)`），演示了 dense/sparse/paged、1CTA 与 2CTA 的 qv 调用方式。

---

## 不同实现对比（表）

| 维度 | MHA | MQA | GQA | MLA (absorb) |
|---|---|---|---|---|
| KV 头数 `Hkv` | `Hq` | 1 | `Hq/g` | —（不存头）|
| cache / token / 层 | `2·Hq·d` | `2·d` | `2·(Hq/g)·d` | `kv_lora + rope`（V3=512+64=576）|
| 是否随头数增长 | 是 | 否 | 部分 | **否** |
| 代表 | GPT-2, 原始 Llama | PaLM, Falcon | Llama-3, Qwen2 | DeepSeek-V2/V3 |
| 精度 | 基准 | 略降 | 接近 MHA | 接近 MHA（低秩够用）|
| 解码 memory 带宽 | 高 | 低 | 中 | 低（只搬 576/token/层）|
| 工程复杂度 | 低 | 低 | 低 | 高（RoPE 解耦 + absorb）|

**naive vs absorb（MLA 内部两条等价路径）**

| | naive (DeepSeek model.py L471-479) | absorb (L480-487) |
|---|---|---|
| cache 内容 | 还原后的大 K/V（随头数）| 潜向量 `c_KV` + 共享 `k_pe`（576）|
| `W_UK`/`W_UV` | 在前向里物化 K/V | 吸收进 q 侧 / o 侧，不物化 |
| 适用 | prefill / 理解 | decode（历史长，省得多）|
| 数学 | `q·(W_UK c)` | `(q·W_UK)·c`（结合律换序）|

---

## 常见困惑 / 坑

1. **"MLA 的 KV cache 和头数无关？"** 对，absorb 路径 cache = `kv_lora_rank + qk_rope_head_dim`，
   DeepSeek-V3 有 128 个头却只存 576/token/层。对比同规模 MHA（`2·128·128=32768`），省约 **57×**。
2. **为什么必须解耦 RoPE？** RoPE 是位置相关的旋转矩阵，不能和 `W_UK` 交换顺序（旋转夹在中间）。
   把位置部分单独拎出来（`q_pe/k_pe`，共享、不走低秩），剩下的内容部分才能做 absorb。
3. **`k_pe` 只有一份（单头）**：model.py L470 `unsqueeze(2)` 出 head 维 =1，naive 里再 `expand` 到每头（L476）。
   别误以为每头有独立的位置 K。
4. **naive 和 absorb 结果必须一致**：它们只是矩阵结合律换序。调试 MLA kernel 时，
   拿 naive 当 reference 对拍 absorb，是标准做法（见本仓库 test_flash_attn_mla.py）。
5. **`repeat_kv` 用 `expand` 不是 `repeat`**：expand 是零拷贝视图；真正 kernel（pack_gqa）连展开都省了。
6. **FlashMLA 的 `causal` 必须 False**：它是稀疏 top-k 路径（DSA），不是稠密 causal。别混淆。
7. **HF 的 DeepSeek-V3 不做 absorb**：它 cache 存压缩态（L460）但前向 `expand_kv` 还原（L464）。
   absorb 的显存/带宽收益要靠 vLLM/SGLang/FlashMLA 等推理栈实现。

---

## 自测题 + 动手练习

**自测题**
1. Llama-3-8B：`Hq=32, Hkv=8, d=128, L=32`。fp16、S=4096、batch=16 时 GQA 的 KV cache 多少 GiB？
   （公式 `2·L·Hkv·d·S·b·batch`）换成 MHA 是多少？
2. DeepSeek-V3：`kv_lora_rank=512, qk_rope_head_dim=64, L=61, Hq=128, d=128`。
   absorb 的 KV cache（`(512+64)·L·S·b·batch`）与"假想 MHA"（`2·L·Hq·d·S·b·batch`）之比是多少？
3. absorb 路径里，`q_nope @ W_UK` 得到的向量是几维？为什么它能直接和 cache 里的 `c_KV` 点积？
   （答：`kv_lora_rank=512`；见 model.py L483）
4. 为什么 MLA 的 `softmax_scale` 用 `head_dim + head_dim_v`（interface.py L853）而不是单个 head_dim？

**动手练习**
1. 跑本仓库 `pytest tests/cute/test_flash_attn_mla.py -k "not sparse" -x`（选一张空闲 GPU：
   `CUDA_VISIBLE_DEVICES=<id>`），观察 dense MLA 前向是否通过。
2. 在 DeepSeek `MLA.forward`（model.py L479 和 L486）各加一行 `print("scores.shape", scores.shape)`，
   分别用 `attn_impl="naive"` 和 `"absorb"` 跑同一输入，确认 scores 形状一致、数值接近。
3. 在 HF `DeepseekV3Attention.forward`（modeling_deepseek_v3.py L460）后加
   `print("cache sizes", kv_nope.shape, k_rot.shape)`，确认 cache 存的是 `[B,1,S,512]` 和 `[B,1,S,64]`。
4. 写一个脚本，对 `MiniMLA`（本文第 5 节）用随机权重分别跑 `absorb=False/True`，
   断言 `torch.allclose(o_naive, o_absorb, atol=1e-4)`，亲手验证两条路等价。
5. 用第 1 题的公式做一张表：固定 S=32768，横轴 batch∈{1,8,32,128}，比较 MHA/GQA/MLA 三者的 KV cache 显存，
   找出各自撑爆 80GB 显卡的 batch 上限。
