# 源码导读 01：从 token 到文字 —— 一个 Decoder-only Transformer 的全链路

> 目标读者：懂 LLM / attention 的概念，但看不懂底层实现代码的学习者。
> 本文按"一句话输入 → 一句话输出"的真实数据流，逐段读真实源码。
> 代码标识符保留英文，解释用中文。所有行号、路径、符号都经过实际阅读验证。

---

## 0. 要解决的问题 / 一句话直觉

**要解决的问题**：给定一段文字，模型要能"接着往下写"。这需要把文字变成数字（tokenizer + embedding），
让每个位置都能"看到"前面所有位置（attention），反复加工语义（decoder layer 堆叠），
最后把每个位置的向量映射回"下一个 token 的概率分布"（lm_head + sampling）。

**一句话直觉**：Transformer 就是一个"查表 → 反复混合上下文 → 再查表"的函数，
中间所有算子（RMSNorm / RoPE / attention / SwiGLU）都是为了让"混合上下文"这步又稳又准。

整条链路：

```
文字  --tokenizer(BPE)-->  [id, id, ...]
      --embedding-->        [B, S, H]      (每个 id 查一行向量)
      --N × DecoderLayer--> [B, S, H]      (每层: norm→attn→残差, norm→mlp→残差)
      --final norm + lm_head--> [B, S, V]  (每个位置的 logits)
      --取最后一个位置 + sampling--> 下一个 id
      --把 id 拼回输入, 循环--> 生成文字
```

记号约定：`B`=batch, `S`/`T`=序列长度, `H`=hidden size (n_embd), `nh`=head 数,
`hd`=head_dim (`H = nh * hd`), `V`=vocab size。

---

## 1. Tokenizer（BPE）：文字 ↔ 整数

### 核心原理

计算机只能算数字。BPE（Byte Pair Encoding）的思想：

1. 先把文字编码成 UTF-8 字节，每个字节是 0..255 的整数（天然就有 256 个基础 token）。
2. 统计相邻字节对出现频率，把**最高频的一对**合并成一个新 token（id 从 256 往上发）。
3. 重复 `vocab_size - 256` 次。常见词（如 " the"）会被合并成单个 token，稀有词仍是多个字节。

形状上：一个长度 `L` 的字符串 → 一个长度 `≤ L` 的整数列表 `ids`。

### 最小实现

```python
def get_stats(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):        # 相邻对
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(idx); i += 2       # 命中就替换成新 id
        else:
            out.append(ids[i]); i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))      # 0..255
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for i in range(vocab_size - 256):
        pair = max(get_stats(ids), key=get_stats(ids).get)   # 最高频对
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # 挑"最早被学会"（merge 序号最小）的那一对先合并
        pair = min(get_stats(ids), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: break
        ids = merge(ids, pair, merges[pair])
    return ids
```

### 源码导读（按阅读顺序）

**Stop 1 — 两个核心 helper：`get_stats` / `merge`**
`repos/00-basics/minbpe/minbpe/base.py`，L13-41
🔗 https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py#L13-L41

```python
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):            # L20: 相邻两元素组成 pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx); i += 2        # L35-37: 命中 pair，替换为新 id，跳过两个
        else:
            newids.append(ids[i]); i += 1
    return newids
```

逐行：`get_stats` 用 `zip(ids, ids[1:])` 这个经典技巧拿到所有相邻对，返回一个 `{(int,int): count}` 字典。
`merge` 做字符串替换的整数版——注意 L35 的 `i < len(ids) - 1` 边界检查，防止在最后一个元素越界。
**为什么这样写**：这两个函数不关心"文字"，只处理整数列表，所以训练和推理都能复用，职责干净。

**Stop 2 — 训练循环 `BasicTokenizer.train`**
`repos/00-basics/minbpe/minbpe/basic.py`，L20-49
🔗 https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L20-L49

```python
text_bytes = text.encode("utf-8")     # L25: 文字→字节
ids = list(text_bytes)                # L26: 字节→0..255 的整数 list
...
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)  # L35: 选当前最高频对
    idx = 256 + i                     # L37: 新 token 的 id
    ids = merge(ids, pair, idx)       # L39: 全量替换
    merges[pair] = idx                # L41: 记下"谁合并成谁"
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # L42: 新 token 的字节串 = 两段拼接
```

逐行：L25-26 是"文字→整数"的入口。`max(stats, key=stats.get)`（L35）即"取 value 最大的 key"。
`merges` 记录合并规则（encode 要用），`vocab` 记录 id→字节（decode 要用）。
**为什么这样写**：贪心地每轮合并最高频对，是 BPE 的定义；id 从 256 递增保证不与字节冲突。

**Stop 3 — 编码 `BasicTokenizer.encode`**
`repos/00-basics/minbpe/minbpe/basic.py`，L57-74
🔗 https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L57-L74

```python
while len(ids) >= 2:
    stats = get_stats(ids)
    pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # L64
    if pair not in self.merges:
        break                         # L70: 没有可合并的对了
    idx = self.merges[pair]
    ids = merge(ids, pair, idx)
```

逐行关键是 L64：encode 时要**按训练时的合并顺序**来合并，所以用 `merges` 里的 id（= 学会的先后）作排序键，
取**序号最小**的对先合并。L65-70 的注释解释了"没有任何对能合并"的终止条件。

**Stop 4 — 真实世界加一层：正则预切分 `RegexTokenizer`**
`repos/00-basics/minbpe/minbpe/regex.py`，L19（split pattern）、L36-41（train 里的切分）
🔗 https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/regex.py#L36-L41

```python
text_chunks = re.findall(self.compiled_pattern, text)   # L41: 先按正则切成 chunk
ids = [list(ch.encode("utf-8")) for ch in text_chunks]  # L44: 每个 chunk 独立成 list
```

**为什么这样写**：GPT 系列不允许跨越空格/标点去合并（否则 "dog." 和 "dog!" 会学到奇怪的 token）。
正则先把文本切成"词块"，BPE 只在块内合并。这就是生产级 tokenizer（tiktoken/nanochat）的做法。

**Stop 5 — 生产级：nanochat 用 rustbpe 训练 + tiktoken 推理**
`repos/00-basics/nanochat/nanochat/tokenizer.py`，L9-26（special tokens + split pattern）、L42-61（train）
🔗 https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L42-L61

要点：L9-21 定义了对话专用的特殊 token（`<|bos|>`、`<|user_start|>` 等），它们不参与 BPE 训练、
在 L54 直接以固定 id 注册。L26 的 split pattern 用 `\p{N}{1,2}`（最多两位数字成一 token）而不是 GPT-4 的 `{1,3}`。
**为什么这样写**：训练用 Rust 实现图快，推理用 tiktoken 图稳，特殊 token 用来标记对话角色，是 chat 模型的刚需。

---

## 2. Embedding：整数 → 向量

### 核心原理

`embedding` 就是一张 `[V, H]` 的大查找表。输入 `ids` 形状 `[B, S]`，输出 `[B, S, H]`——
每个整数被替换成表里对应的一行 `H` 维向量。**没有矩阵乘，就是按下标取行**。

很多模型还有 **weight tying**：输入 embedding 和输出 lm_head 共享同一张表（省一大坨参数）。

### 源码导读

**Stop 1 — nanoGPT 的 embedding 与 weight tying**
`repos/00-basics/nanoGPT/model.py`，L126-138（定义）、L177-179（前向）
🔗 https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L126-L138

```python
wte = nn.Embedding(config.vocab_size, config.n_embd),  # L127: token embedding [V, H]
wpe = nn.Embedding(config.block_size, config.n_embd),  # L128: 位置 embedding [S_max, H]
...
self.transformer.wte.weight = self.lm_head.weight      # L138: weight tying
```

前向（L177-179）：

```python
tok_emb = self.transformer.wte(idx)   # [B, T, H]  按 id 查表
pos_emb = self.transformer.wpe(pos)   # [T, H]     按位置 0..T-1 查表
x = self.transformer.drop(tok_emb + pos_emb)   # 相加后 [B, T, H]
```

**为什么这样写**：GPT-2 用**可学习的绝对位置编码**（`wpe`），直接加到 token embedding 上。
而 Llama/nanochat 改用 RoPE（见第 4 节），没有 `wpe`。L138 的 tying 让 `wte` 和 `lm_head` 是同一块内存。

**Stop 2 — Llama 的 embedding（无绝对位置编码）**
`repos/00-basics/transformers/.../llama/modeling_llama.py`，L342（定义）、L369-370（前向）
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L369-L370

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)  # L342
...
inputs_embeds = self.embed_tokens(input_ids)   # L370: [B, S, H]，没有 +pos_emb
```

**为什么这样写**：Llama 的位置信息全部交给 attention 里的 RoPE，所以 embedding 这步**只查表不加位置**。

---

## 3. RMSNorm：让每层输入稳定

### 核心原理

归一化让激活值的尺度稳定，训练不发散。LayerNorm 会减均值再除标准差；
**RMSNorm** 更简单：不减均值，只用"均方根"缩放，再乘一个可学习的 per-channel 权重 `weight`。

公式（对最后一维 `H` 做）：

```
rms = sqrt(mean(x^2) + eps)          # 标量 (按行)
y   = x / rms * weight               # [..., H]
```

输入输出形状都是 `[B, S, H]`。比 LayerNorm 少一次减均值，更快，效果相当。

### 最小实现

```python
import torch
def rmsnorm(x, weight, eps=1e-6):
    # x: [B, S, H]
    dtype = x.dtype
    x = x.float()                                  # 升到 fp32 算，防溢出
    var = x.pow(2).mean(-1, keepdim=True)          # [B, S, 1]
    x = x * torch.rsqrt(var + eps)                 # 归一化
    return (weight * x.to(dtype))                  # 乘可学习权重
```

### 源码导读

**Stop 1 — `LlamaRMSNorm`**
`.../llama/modeling_llama.py`，L52-66
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L52-L66

```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)              # L63: 升 fp32
    variance = hidden_states.pow(2).mean(-1, keepdim=True)       # L64: 均方 [B,S,1]
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # L65
    return self.weight * hidden_states.to(input_dtype)           # L66: 乘权重后降回原 dtype
```

逐行：L63 升到 fp32 是关键——`pow(2)` 在 fp16/bf16 下容易溢出或丢精度。L64 `mean(-1)` 对最后一维求均方，
`keepdim=True` 让它保持 `[B,S,1]` 便于广播。L65 用 `rsqrt`（1/sqrt）一步到位。L66 乘权重后**降回输入 dtype**。
**为什么这样写**：fp32 中间计算 + 原 dtype 输出，是数值稳定和速度的折中。

**Stop 2 — nanochat 直接用 PyTorch 内置**
`repos/00-basics/nanochat/nanochat/gpt.py`，L42-43
🔗 https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L42-L43

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))   # 无可学习权重的 RMSNorm
```

**为什么这样写**：nanochat 走极简路线，连 norm 的权重都省了（现代研究发现可省），直接调 `F.rms_norm`。

---

## 4. RoPE：把"相对位置"注入 Q/K

### 核心原理

RoPE（Rotary Position Embedding）的思想：把 head_dim 的向量**两两分组当成复平面上的点**，
在位置 `m` 处把第 `i` 组旋转角度 `m * θ_i`。两个向量点积时，旋转量相减，于是点积只依赖**相对位置** `m-n`。
这就实现了"相对位置编码"，且不增加参数、能外推。

频率：`θ_i = base^(-2i/dim)`，低维转得快、高维转得慢。只作用在 Q 和 K 上（V 不转）。

形状：`q, k` 形如 `[B, nh, S, hd]`（或 `[B, S, nh, hd]`），`cos/sin` 形如 `[S, hd]`，广播相乘。

### 最小实现

```python
import torch
def build_cos_sin(S, hd, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))  # [hd/2]
    t = torch.arange(S).float()
    freqs = torch.outer(t, inv_freq)          # [S, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)   # [S, hd]  (前后半相同)
    return emb.cos(), emb.sin()

def rotate_half(x):                            # [..., hd]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):                # q,k: [B, nh, S, hd]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)   # 加 head 维便于广播
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

### 源码导读

**Stop 1 — 频率计算 `compute_default_rope_parameters`**
`.../llama/modeling_llama.py`，L100-106
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L100-L106

```python
base = config.rope_parameters["rope_theta"]
dim  = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # L105: [hd/2]
```

逐行：L105 就是 `θ_i`。`torch.arange(0, dim, 2)` 取偶数下标 `0,2,4,...`，除以 `dim` 归一化，
`base` 的负幂得到从大到小的一串频率。**为什么这样写**：这是 RoPE 论文的定义；`base`（常见 10000 或 500000）
越大，高维转得越慢、越能支持长上下文。

**Stop 2 — 生成 cos/sin**
`.../llama/modeling_llama.py`，L110-116
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L110-L116

```python
freqs = position_ids[..., None].float() * self.inv_freq.to(...)   # L111: [B,S,1]*[hd/2]→[B,S,hd/2]
emb = torch.cat((freqs, freqs), dim=-1)      # L112: [B, S, hd]  前后半复制
cos = emb.cos() * self.attention_scaling     # L113
sin = emb.sin() * self.attention_scaling     # L114
```

**为什么 `cat((freqs,freqs))`**：配合下面 `rotate_half` 的"前半 / 后半"切分方式，
保证第 `i` 维和第 `i+hd/2` 维用的是同一个角度 `freqs[i]`，构成一组旋转对。

**Stop 3 — `rotate_half` 与 `apply_rotary_pos_emb`**
`.../llama/modeling_llama.py`，L119-149
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L119-L149

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]      # L121: 前半
    x2 = x[..., x.shape[-1] // 2 :]      # L122: 后半
    return torch.cat((-x2, x1), dim=-1)  # L123: (-后半, 前半)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)   # L145: 加 head 维 [B,1,S,hd]
    sin = sin.unsqueeze(unsqueeze_dim)   # L146
    q_embed = (q * cos) + (rotate_half(q) * sin)   # L147
    k_embed = (k * cos) + (rotate_half(k) * sin)   # L148
    return q_embed, k_embed
```

逐行：`(x*cos) + (rotate_half(x)*sin)` 就是复数乘 `(a+bi)(cosθ+i sinθ)` 展开后的实现——
`rotate_half` 提供了那个"乘 i"的虚部交换与变号。`unsqueeze_dim=1`（L145）在 head 维插 1，让 `[B,S,hd]` 的 cos/sin
能广播到 `[B,nh,S,hd]` 的 q/k。**为什么这样写**：避免显式复数运算，用纯实数张量就能表达旋转，对硬件友好。

**Stop 4 — nanochat 的 RoPE（半分割写法）**
`repos/00-basics/nanochat/nanochat/gpt.py`，L57-65（apply）、L270-285（precompute）
🔗 https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L57-L65

```python
def apply_rotary_emb(x, cos, sin):           # x: [B, T, nh, hd]
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]          # L62: 前后半
    y1 = x1 * cos + x2 * sin                 # L63
    y2 = x1 * (-sin) + x2 * cos              # L64
    return torch.cat([y1, y2], 3)
```

对比：nanochat 的 x 布局是 `[B, T, nh, hd]`（head 在倒数第二维，正是 FlashAttention 喜欢的布局），
cos/sin 预计算成 `[1, T, 1, hd/2]`（L284）。数学等价，只是旋转方向相反（见 L58 注释，不影响相对关系）。

---

## 5. Attention：让每个位置看见前文

### 核心原理

对每个 query 位置，和所有 key 做点积得到"相关度"分数，softmax 归一化成权重，再按权重加权求和 value。
Causal mask 保证第 `t` 个位置只能看到 `≤ t` 的位置（自回归不能偷看未来）。

形状流（单头）：`Q,K,V: [B, nh, S, hd]`
```
scores = Q @ K^T / sqrt(hd)      # [B, nh, S, S]
scores = scores + causal_mask    # 上三角填 -inf
P = softmax(scores, dim=-1)      # [B, nh, S, S]
O = P @ V                        # [B, nh, S, hd]
```

### 最小实现

```python
import torch, torch.nn.functional as F
def mha(x, Wqkv, Wo, nh):
    B, S, H = x.shape
    hd = H // nh
    q, k, v = Wqkv(x).split(H, dim=2)                 # 各 [B, S, H]
    q = q.view(B, S, nh, hd).transpose(1, 2)          # [B, nh, S, hd]
    k = k.view(B, S, nh, hd).transpose(1, 2)
    v = v.view(B, S, nh, hd).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / hd**0.5      # [B, nh, S, S]
    mask = torch.triu(torch.ones(S, S), 1).bool()     # 上三角(不含对角)=True
    scores = scores.masked_fill(mask, float("-inf"))
    o = F.softmax(scores, -1) @ v                     # [B, nh, S, hd]
    o = o.transpose(1, 2).reshape(B, S, H)            # 拼回 [B, S, H]
    return Wo(o)
```

> FlashAttention 的全部意义，就是**不显式生成 `[B,nh,S,S]` 的大矩阵**，而是分块在线 softmax，
> 把中间结果留在 SRAM 里。数学完全等价，省显存且更快。本系列后续 kernel 导读会深入。

### 源码导读

**Stop 1 — nanoGPT 的 `CausalSelfAttention`（最易读）**
`repos/00-basics/nanoGPT/model.py`，L52-76
🔗 https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L52-L76

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # L56: 一次投影出 QKV
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # L57: [B,nh,T,hd]
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # L58
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # L59
if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # L64: 走 flash 内核
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # L67
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # L68: causal
    att = F.softmax(att, dim=-1)                                     # L69
    y = att @ v                                                      # L71
y = y.transpose(1, 2).contiguous().view(B, T, C)   # L72: 拼回 [B, T, H]
```

逐行：L56 把 Q/K/V 三个投影合成一个 `nn.Linear(H, 3H)`（一次矩阵乘更快），再 `split`。
L57-59 的 `view + transpose` 把 `[B,T,H]` 拆成多头 `[B,nh,T,hd]`。L62-71 给出"有无 flash"两条等价路径——
**L64 和 L67-71 算的是同一个东西**，这是理解 FlashAttention 的锚点。L72 把多头结果拼回。
**为什么这样写**：合并 QKV 投影、`is_causal=True` 让内核内部生成 mask（比显式 `masked_fill` 省内存）。

**Stop 2 — Llama 的 `eager_attention_forward`（标准数学实现）**
`.../llama/modeling_llama.py`，L180-202
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L180-L202

```python
key_states   = repeat_kv(key,   module.num_key_value_groups)   # L190: GQA 展开(见导读02)
value_states = repeat_kv(value, module.num_key_value_groups)   # L191
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # L193: QK^T
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask               # L195: 加 mask(非 masked_fill)
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # L197
attn_output = torch.matmul(attn_weights, value_states)         # L199: P @ V
attn_output = attn_output.transpose(1, 2).contiguous()         # L200
```

逐行：L193 的 `scaling = head_dim**-0.5`（在 `LlamaAttention.__init__` L215 定义）。
L195 **加法式 mask**（mask 本身是含 `-inf` 的加性张量），比 nanoGPT 的 `masked_fill` 更通用（能表达任意偏置）。
L197 **softmax 用 fp32** 再降回，和 RMSNorm 同理。L190-191 的 `repeat_kv` 是 GQA 的关键，下一篇详讲。

**Stop 3 — `LlamaAttention.forward` 的完整编排**
`.../llama/modeling_llama.py`，L232-270
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L232-L270

```python
hidden_shape = (*input_shape, -1, self.head_dim)
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # L243
key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # L244
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # L245
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # L248
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)  # L251: 写KV cache
attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)  # L257
attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # L268
attn_output = self.o_proj(attn_output)                            # L269: 输出投影
```

逐行：这是"真实世界"的 attention 骨架——分开的 q/k/v proj（因为 GQA 下 k/v 维度更小）、
L248 先 RoPE 再写 cache（**顺序很重要**，cache 里存的是已旋转的 K）、L257 通过 `attention_interface`
分发到 eager/sdpa/flash 后端。**为什么先 RoPE 再 cache**：这样解码时历史 K 不用重新旋转。

---

## 6. SwiGLU MLP：每层的"深加工"

### 核心原理

attention 负责"混合位置间信息"，MLP 负责"对每个位置独立做非线性变换"。
现代 LLM 用 **SwiGLU**：两个并行上投影 `gate` 和 `up`，用 `gate` 过 SiLU 当"门"去调制 `up`，再下投影。

```
h = down( SiLU(gate(x)) * up(x) )     # x:[B,S,H] → 中间[B,S,I] → 回[B,S,H]
```

`I`（intermediate_size）通常是 `H` 的 ~2.7×（因为有两个上投影，总参数量与传统 4×H 的 MLP 相当）。

### 最小实现

```python
import torch.nn as nn, torch.nn.functional as F
class SwiGLU(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.gate = nn.Linear(H, I, bias=False)
        self.up   = nn.Linear(H, I, bias=False)
        self.down = nn.Linear(I, H, bias=False)
    def forward(self, x):                      # [B, S, H]
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

### 源码导读

**Stop 1 — `LlamaMLP`（SwiGLU 标准实现）**
`.../llama/modeling_llama.py`，L152-165
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L152-L165

```python
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # L158
self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # L159
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # L160
self.act_fn    = ACT2FN[config.hidden_act]        # L161: "silu"（见 configuration L70）
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # L164
```

逐行：L164 一行就是 SwiGLU——`act_fn(gate_proj(x))` 是门，逐元素乘 `up_proj(x)`，再 `down_proj`。
`config.hidden_act == "silu"`（configuration_llama.py L70）。**为什么两个上投影**：门控比单一激活表达力强，
是 PaLM/Llama 验证过的有效设计。

**Stop 2 — 对比：nanoGPT 的传统 MLP（GELU，无门控）**
`repos/00-basics/nanoGPT/model.py`，L78-92
🔗 https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L78-L92

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  # L82: 上投影 4×
self.gelu   = nn.GELU()                                    # L83
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # L84: 下投影
def forward(self, x):
    x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x)  # L88-90
```

**对比**：GPT-2 是"上投影 4× → GELU → 下投影"，**没有门控、只有一个上投影**。SwiGLU 是其进化版。

**Stop 3 — 对比：nanochat 的 ReLU²**
`repos/00-basics/nanochat/nanochat/gpt.py`，L131-141
🔗 https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L131-L141

```python
x = self.c_fc(x)
x = F.relu(x).square()     # L139: ReLU 后平方（ReLU²激活），也无门控
x = self.c_proj(x)
```

---

## 7. Decoder Layer：残差把一切串起来

### 核心原理

一层 = `x = x + attn(norm(x))` 然后 `x = x + mlp(norm(x))`。这是 **pre-norm 残差**结构：
norm 放在子层**之前**，残差直连。形状全程 `[B, S, H]` 不变，所以能无脑堆 N 层。

### 源码导读

**Stop 1 — `LlamaDecoderLayer`**
`.../llama/modeling_llama.py`，L284-313
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L284-L313

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)           # L295: 先 norm
hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)  # L297: attention
hidden_states = residual + hidden_states                      # L306: 残差

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)  # L310: 再 norm
hidden_states = self.mlp(hidden_states)                       # L311: MLP
hidden_states = residual + hidden_states                      # L312: 残差
```

逐行：**两个子块，结构完全对称**。L295/L310 是 pre-norm（在进入子层前归一化），
L306/L312 残差把子层输出加回。**为什么 pre-norm**：残差路径上没有 norm，梯度能直达底层，深层模型更好训。

**Stop 2 — nanoGPT 的 `Block`（一样的模式）**
`repos/00-basics/nanoGPT/model.py`，L103-106
🔗 https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L103-L106

```python
x = x + self.attn(self.ln_1(x))   # L104
x = x + self.mlp(self.ln_2(x))    # L105
```

两行讲清 Transformer block 的全部骨架。nanochat 的 `Block.forward`（gpt.py L150-152）同理。

---

## 8. lm_head + Sampling：向量 → 下一个词

### 核心原理

堆完 N 层、做完最后一次 norm，得到 `[B, S, H]`。`lm_head`（一个 `[H, V]` 线性层）把每个位置映射成
`V` 维 logits。**生成时只需最后一个位置**的 logits `[B, V]`，经过 temperature / top-k / top-p 调整后，
softmax 成概率，采样出下一个 token，拼回输入，循环。

- temperature：`logits / T`，T<1 更确定、T>1 更随机。
- top-k：只保留最大的 k 个 logits，其余置 -inf。
- top-p (nucleus)：按概率累积到 p 为止的最小集合。

### 最小实现

```python
import torch, torch.nn.functional as F
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]            # 只取最后位置 [B, V]
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")   # top-k 截断
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)    # 采样 [B, 1]
        idx = torch.cat([idx, next_id], dim=1)   # 拼回
    return idx
```

### 源码导读

**Stop 1 — 最终 norm + lm_head（Llama）**
`.../llama/modeling_llama.py`，L402（norm）、L466-469（lm_head）
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L466-L469

```python
hidden_states = outputs.last_hidden_state
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # L468
logits = self.lm_head(hidden_states[:, slice_indices, :])   # L469: [B, kept, V]
```

逐行：L468-469 的 `logits_to_keep` 技巧——生成时只需最后一个位置，传 `logits_to_keep=1` 就**只对最后一行**
算 `lm_head`，省掉对整段序列算 `[B,S,V]`（V 很大，这个优化很值）。nanoGPT 同样做法见 model.py L190
(`self.lm_head(x[:, [-1], :])`)。

**Stop 2 — nanoGPT 的 `generate`（最直白的采样循环）**
`repos/00-basics/nanoGPT/model.py`，L305-330
🔗 https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L305-L330

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  # L314: 裁剪上下文
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :] / temperature       # L318: 取最后位置 + 温度
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')   # L322: top-k 截断
    probs = F.softmax(logits, dim=-1)              # L324
    idx_next = torch.multinomial(probs, num_samples=1)  # L326: 采样
    idx = torch.cat((idx, idx_next), dim=1)        # L328: 拼回继续
```

逐行：L314 处理上下文超长（GPT-2 无 KV cache，每步重算整段，所以要裁到 block_size）。
L318/L322/L324/L326 就是 temperature→top-k→softmax→multinomial 的教科书顺序。**没有 KV cache，所以很慢**。

**Stop 3 — 生产级采样循环 `GenerationMixin._sample`（HF transformers）**
`repos/00-basics/transformers/src/transformers/generation/utils.py`，L3208-3267
🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/generation/utils.py#L3208-L3267

```python
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=...):  # L3208
    model_inputs = self.prepare_inputs_for_generation(input_ids, ...)   # L3211: 只喂新 token + KV cache
    outputs = model_forward(**model_inputs, return_dict=True)           # L3214
    next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, ...)  # L3226: 最后位置
    next_token_scores = logits_processor(input_ids, next_token_logits)  # L3229: temperature/top-k/top-p
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)        # L3253
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # L3255
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)           # L3257: 贪心
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)    # L3264
```

逐行：和 nanoGPT 同构，但多了三件生产级的事：
1. L3211 `prepare_inputs_for_generation` 配合 KV cache，**只把新 token 喂进去**（不重算历史）。
2. L3229 `logits_processor` 是一串可插拔的处理器（temperature/top-k/top-p 各是一个类）。
3. L3208 `_has_unfinished_sequences` 处理 batch 里不同序列早停（遇到 EOS 填 pad）。

**Stop 4 — 采样处理器的真身**
`repos/00-basics/transformers/src/transformers/generation/logits_process.py`
- `TemperatureLogitsWarper.__call__` L301-303（`scores / self.temperature`）
  🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/generation/logits_process.py#L301-L303
- `TopKLogitsWarper.__call__` L606-611（`scores < topk(...)[...,-1]` 置 -inf）
  🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/generation/logits_process.py#L606-L611
- `TopPLogitsWarper.__call__` L543-556（排序→累积概率→截断）
  🔗 https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/generation/logits_process.py#L543-L556

它们在 `_get_logits_processor`（utils.py L1595/L1600/L1604）里被按配置组装进 `logits_processor`。

**Stop 5 — 带 KV cache + 工具调用的完整引擎（nanochat）**
`repos/00-basics/nanochat/nanochat/engine.py`，`Engine.generate` L176-275
🔗 https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py#L194-L234

```python
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)  # L205: 一次性 prefill 整段 prompt
logits = logits[:, -1, :].expand(num_samples, -1)            # L206: 最后位置, 复制成多样本
kv_cache_decode.prefill(kv_cache_prefill)                    # L217: 把 prefill 的 KV 复制给解码 cache
while True:
    next_ids = sample_next_token(logits, rng, temperature, top_k)  # L234: 采样一列
    ...
    ids = torch.tensor(token_column, ...).unsqueeze(1)             # L274: 只喂新 token
    logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # L275: decode 一步
```

逐行：这是真实推理引擎的样子——**prefill**（L205，一次算完 prompt 填满 KV cache）和
**decode**（L274-275，每步只喂 1 个新 token）两阶段分离。`sample_next_token`（engine.py L141-156）
封装了 temperature=0→argmax、否则 top-k+multinomial。KV cache 结构见下节导读 02。

---

## 不同实现对比

| 维度 | nanoGPT (GPT-2) | nanochat | Llama (HF transformers) |
|---|---|---|---|
| 位置编码 | 可学习绝对 `wpe` (model.py L128) | RoPE (gpt.py L57) | RoPE (modeling_llama.py L127) |
| Norm | LayerNorm+bias (L18) | `F.rms_norm` 无权重 (L42) | RMSNorm 有权重 (L52) |
| 激活/MLP | GELU, 4× 单上投影 (L78) | ReLU², 单上投影 (L131) | **SwiGLU** 双上投影 (L152) |
| KV heads | 全头 MHA (L56) | GQA `n_kv_head` (L34) | GQA `num_key_value_heads` |
| KV cache | 无，每步重算 | 有 (engine.py `KVCache`) | 有 (`DynamicCache`) |
| 采样 | 内联循环 (L312) | `Engine` + 工具调用 (L176) | `LogitsProcessor` 流水线 |
| attention 内核 | SDPA (L64) | FlashAttention3 (gpt.py L110) | 可插拔 interface (L253) |

---

## 常见困惑 / 坑

1. **"RoPE 为什么能表达相对位置？"** 关键在点积：旋转矩阵满足 `R(m)^T R(n) = R(n-m)`，
   所以 `q_m · k_n` 只依赖 `n-m`。V 不转，因为 V 不参与"打分"。
2. **cos/sin 的 `cat((freqs,freqs))` vs nanochat 的半分割**：两种实现把哪两维配成旋转对的约定不同
   （HF 配 `i` 与 `i+hd/2`，有些实现配 `2i` 与 `2i+1`）。同一套权重必须用同一套约定，混用会乱。
3. **softmax 前 scale 容易漏**：`scaling = head_dim**-0.5`（llama L215）。少了它，深层/大 head_dim 时 logits 过大，
   softmax 饱和、梯度消失。
4. **先 RoPE 还是先写 cache？** 必须**先 RoPE 再写 cache**（llama L248→L251），否则解码时要对历史 K 重算旋转。
5. **生成时别算整段 logits**：用 `logits_to_keep=1` / `x[:, [-1], :]` 只算最后一行，V 很大时这是显著优化。
6. **weight tying 的内存别算两遍**：`wte.weight is lm_head.weight`（nanoGPT L138）是同一块张量。

---

## 自测题 + 动手练习

**自测题**
1. 一个 32K vocab 的 BPE，训练做了多少次 merge？（答：`32768 - 256 - 特殊token数`；nanochat 见 tokenizer.py L46）
2. `rotate_half` 里为什么是 `(-x2, x1)` 而不是 `(x2, -x1)`？（提示：对应复数乘 `i` 的实部/虚部）
3. Llama 的 `eager_attention_forward` 用加法式 mask（L195），nanoGPT 用 `masked_fill`（L68）。
   哪种能表达"任意加性偏置"（如 ALiBi）？为什么？
4. `_sample` 里 KV cache 让第 2 步起只喂 1 个 token，但 nanoGPT 的 `generate` 每步喂整段。
   如果序列长 `S`、生成 `N` 个 token，两者的总前向计算量各是 `O(?)`？

**动手练习**
1. 在 `repos/00-basics/nanoGPT/model.py` 的 `CausalSelfAttention.forward`（L52）里，
   `q = q.view(...)` 之后加一行 `print("q shape:", q.shape)`，跑一个 forward，确认是 `[B, nh, T, hd]`。
2. 在 `LlamaMLP.forward`（modeling_llama.py L164）拆成三行，分别打印 `gate_proj(x)`、`up_proj(x)`、
   `act_fn(...)*up` 的形状，确认中间维是 `intermediate_size`。
3. 用上面"最小实现"的 `build_cos_sin` 画出 `cos[:, 0]` 和 `cos[:, hd//2-1]` 随位置的曲线，
   直观感受"低维转得快、高维转得慢"。
4. 实现一个 `top_p` 截断函数，和 `TopPLogitsWarper`（logits_process.py L543）对拍，输入同一组 logits 看结果是否一致。
