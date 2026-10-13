# 03 · KV Cache 与 Paged Attention：从 HF DynamicCache 到 vLLM / SGLang

> 阅读前提：你已经知道 Transformer 自回归生成、Q/K/V、多头/GQA 这些概念。本篇带你从“为什么要缓存 K/V”一路走到“block_table / slot_mapping 是怎么一层层传进 attention kernel 的”。
>
> 所有源码位置都在本仓库的子模块里，行号均已对照当前 checkout 核实（commit 见每个链接）。

---

## 0. 要解决的问题 / 一句话直觉

**问题**：自回归生成第 t 步时，新 token 的 query 需要和前面所有 token 的 key/value 做 attention。如果每步都把整段前缀重新过一遍模型，计算量是 O(t²·层数)，而前缀的 K/V 其实每一步都一样。

**一句话直觉**：
- **KV Cache**：把每层算过的 K/V 存起来，下一步只算新 token 的 Q/K/V，然后“新 Q × 旧 K/V（加上新 K/V）”。
- **Paged Attention**：KV Cache 不再给每个请求预留一整块连续显存，而是像操作系统的虚拟内存一样切成固定大小的 **block（page）**，每个请求用一张 **block_table（页表）** 记录“逻辑第 i 块 → 物理第几块”。attention kernel 读 K/V 时查页表。
- **Prefix Caching**：两个请求前缀相同（同一个 system prompt），它们的前缀 block 可以共享——用 **哈希**（vLLM / nano-vllm）或 **radix tree**（SGLang）找到已经算过的 block。
- **Continuous Batching**：每一步（iteration）都重新决定 batch 里有谁：结束的请求立即退出、释放 block，新请求立即进来，而不是等一整个 batch 都生成完。

---

## 1. 核心原理

### 1.1 Prefill vs Decode（带张量形状）

设单层、batch=B、query 头数 `Hq`、KV 头数 `Hkv`（GQA 下 `Hq = G·Hkv`）、head_dim `D`。

**Prefill（处理 prompt，长度 L）**
```
x:  [B, L, d_model]
Q:  [B, L, Hq,  D]      K, V: [B, L, Hkv, D]
S = Q Kᵀ / √D:  [B, Hq, L, L]   (加 causal mask)
O = softmax(S) V: [B, L, Hq, D]
写入 cache: K_cache[:, 0:L] = K,  V_cache[:, 0:L] = V
```
特点：一次处理 L 个 token，矩阵乘是“胖”的 GEMM，**算力瓶颈（compute-bound）**。

**Decode（生成第 t 个 token，已有 t-1 个缓存）**
```
x:  [B, 1, d_model]
q:  [B, 1, Hq, D]       k_new, v_new: [B, 1, Hkv, D]
K_cache[:, t-1] = k_new ; V_cache[:, t-1] = v_new
S = q K_cacheᵀ: [B, Hq, 1, t]
o = softmax(S) V_cache: [B, 1, Hq, D]
```
特点：每个请求只有 1 个 query，但要把整段 K/V（t 个）从显存读一遍，算术强度极低，**带宽瓶颈（memory-bound）**。这就是为什么 decode 要尽量把很多请求拼成一个 batch（continuous batching），以及为什么要有 split-KV（见 04 篇）。

### 1.2 显存账

每个 token 的 KV 字节数：

```
bytes_per_token = 2 (K和V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

例子（bf16，2 字节）：
| 模型 | layers | kv_heads | head_dim | 每 token | 4096 token / 请求 |
|---|---|---|---|---|---|
| Llama-2-7B（MHA） | 32 | 32 | 128 | 2·32·32·128·2 = 512 KiB | 2 GiB |
| Llama-3-8B（GQA） | 32 | 8 | 128 | 2·32·8·128·2 = 128 KiB | 512 MiB |

所以 KV cache 往往比权重还大，**谁能把 KV 显存用得更满，谁的并发就更高**。传统做法给每个请求按 `max_seq_len` 预分配连续空间，平均只用到一小部分（内部碎片）+ 不同请求长度不一（外部碎片）。Paged 方案把浪费压到“每个请求最后一个 block 没填满的部分”。

nano-vllm 里正好有这个公式（见下文 `allocate_kv_cache`）：
```python
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
```

### 1.3 Paged KV 的三个核心数据结构

```
物理 KV 池（每层一份）:  k_cache: [num_blocks, block_size, Hkv, D]
block_table（每请求一行）: [B, max_blocks_per_seq]  int32，第 b 个请求的逻辑块 i 存在物理块 block_table[b, i]
slot_mapping（每个“本步新 token”一个）: [num_new_tokens] int，= block_table[b, pos // block_size] * block_size + pos % block_size
```

- **写**（store K/V）用 `slot_mapping`：把 `[num_new_tokens, Hkv, D]` 的新 K/V scatter 到 `k_cache.view(-1, Hkv, D)[slot]`。
- **读**（attention）用 `block_table` + `seqlens`：kernel 计算第 j 个 key 的物理位置 `block_table[b, j // block_size] * block_size + j % block_size`。

### 1.4 Prefix caching 的哈希链

把 token 序列按 block 切开，第 i 块的哈希 = `hash(第 i-1 块的哈希, 第 i 块的 token_ids)`。这样一个块的哈希**隐含了它之前的整段前缀**，相同哈希 ⇒ 相同前缀 ⇒ 可以复用物理块（引用计数 +1）。只有**填满**的块才算哈希（未满的块内容还会变）。

Radix tree（SGLang）是另一种组织方式：树的边是 token 序列片段，从根走到某节点的路径就是一个前缀，节点的 value 是这段 token 在 KV 池里的索引。匹配 = 沿树往下走。

---

## 2. 最小实现：玩具 Paged KV Cache + 前缀哈希（PyTorch）

下面这段代码捕捉了 paged attention 的本质：物理池、block_table、slot_mapping、按页 gather 的 decode attention，以及前缀哈希。它和连续存储的结果逐位一致（我用等价的 numpy 版本跑过对比，误差为 0）。

```python
import math, torch

class ToyPagedKV:
    """单层的 paged KV 池。k/v: [num_blocks, block_size, Hkv, D]"""
    def __init__(self, num_blocks, block_size, n_kv_heads, head_dim):
        self.bs = block_size
        self.k = torch.zeros(num_blocks, block_size, n_kv_heads, head_dim)
        self.v = torch.zeros_like(self.k)
        self.free = list(range(num_blocks))[::-1]   # 空闲物理块
        self.block_tables = {}                      # seq_id -> [物理块号...]
        self.lens = {}                              # seq_id -> 已缓存 token 数

    def append(self, sid, k, v):                    # k, v: [T, Hkv, D]  (prefill: T=L, decode: T=1)
        bt = self.block_tables.setdefault(sid, [])
        L, T = self.lens.get(sid, 0), k.shape[0]
        need = (L + T + self.bs - 1) // self.bs - len(bt)
        for _ in range(need):                       # 只在跨块时才分配新块
            bt.append(self.free.pop())
        pos = torch.arange(L, L + T)                # 这些新 token 的逻辑位置
        slot = torch.tensor(bt)[pos // self.bs] * self.bs + pos % self.bs   # = slot_mapping
        self.k.view(-1, *k.shape[1:])[slot] = k     # scatter 写入（store_kvcache 做的事）
        self.v.view(-1, *v.shape[1:])[slot] = v
        self.lens[sid] = L + T
        return slot

def paged_decode_attention(q, cache, sids):         # q: [B, Hq, D]，每个请求 1 个 query
    B, Hq, D = q.shape
    Hkv = cache.k.shape[2]; G = Hq // Hkv
    max_blocks = max(len(cache.block_tables[s]) for s in sids)
    block_table = torch.zeros(B, max_blocks, dtype=torch.long)       # 填充位随便填，靠 seqlens 屏蔽
    for i, s in enumerate(sids):
        block_table[i, :len(cache.block_tables[s])] = torch.tensor(cache.block_tables[s])
    seqlens = torch.tensor([cache.lens[s] for s in sids])           # = cache_seqlens / seqused_k
    K = cache.k[block_table].flatten(1, 2)        # [B, max_blocks*bs, Hkv, D]  按页表 gather
    V = cache.v[block_table].flatten(1, 2)
    K = K.repeat_interleave(G, dim=2); V = V.repeat_interleave(G, dim=2)   # GQA: 每 G 个 q 头共享 1 个 kv 头
    s = torch.einsum('bhd,bnhd->bhn', q, K) / math.sqrt(D)          # [B, Hq, N]
    valid = torch.arange(K.shape[1])[None, :] < seqlens[:, None]    # [B, N]
    s = s.masked_fill(~valid[:, None, :], float('-inf'))
    return torch.einsum('bhn,bnhd->bhd', s.softmax(-1), V)          # [B, Hq, D]

def block_hashes(token_ids, bs):
    """前缀哈希链：只对满块计算；第 i 块哈希依赖第 i-1 块哈希。"""
    h, out = None, []
    for i in range(len(token_ids) // bs):
        h = hash((h, tuple(token_ids[i * bs:(i + 1) * bs])))
        out.append(h)
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    cache, ref = ToyPagedKV(16, 4, 2, 8), {}
    for sid, T in [(0, 7), (1, 3)]:                                  # 两个请求 prefill
        k, v = torch.randn(T, 2, 8), torch.randn(T, 2, 8)
        cache.append(sid, k, v); ref[sid] = (k, v)
    for _ in range(3):                                               # 交替 decode，物理块会交错
        for sid in (0, 1):
            k, v = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
            print(sid, "slot_mapping =", cache.append(sid, k, v).tolist())
            ref[sid] = (torch.cat([ref[sid][0], k]), torch.cat([ref[sid][1], v]))
    q = torch.randn(2, 4, 8)
    o = paged_decode_attention(q, cache, [0, 1])
    for i, s in enumerate((0, 1)):                                   # 对照：连续存储的朴素 attention
        K, V = (t.repeat_interleave(2, dim=1) for t in ref[s])
        r = (torch.einsum('hd,nhd->hn', q[i], K) / math.sqrt(8)).softmax(-1)
        print("max err", (torch.einsum('hn,nhd->hd', r, V) - o[i]).abs().max().item())
    print("block_tables:", cache.block_tables)                       # 例如 {0: [0, 1, 3], 1: [2, 4]}
    print(block_hashes([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)[:1] == block_hashes([1, 2, 3, 4, 0, 0, 0, 0], 4)[:1])  # True：首块可共享
```

对照真实系统：
- `append` 中的 `slot` 计算 = nano-vllm `prepare_prefill/prepare_decode` 里的 slot_mapping、vLLM 的 `ComputeSlotMappingKernel`。
- `self.k.view(-1, ...)[slot] = k` = nano-vllm 的 Triton `store_kvcache_kernel`、vLLM 的 `reshape_and_cache_flash`。
- `paged_decode_attention` 里显式 gather 出 `[B, N, Hkv, D]` 只是为了好懂；真实 kernel **不会**物化这个张量，而是在 kernel 内部按 tile 查页表直接从物理块加载（见 §3.6 FA2 kernel 内的 `block_table[...]` 指针运算）。

---

## 3. 源码导读（按阅读顺序）

### 3.1 起点：HF transformers 的 `DynamicCache`——“能用但不适合服务”

**Stop 1：`DynamicLayer.update` —— 每步 `torch.cat`**

- 文件：`repos/00-basics/transformers/src/transformers/cache_utils.py`，`DynamicLayer` L113–159
- 链接：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/cache_utils.py#L113-L159

```python
class DynamicLayer(CacheLayerMixin):
    """... It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`."""
    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)     # L145
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
```
逐行解释：
- 布局是 `[B, H_kv, seq_len, D]`，`dim=-2` 就是 seq 维。prefill 时 `key_states` 是 `[B, H_kv, L, D]`，decode 时是 `[B, H_kv, 1, D]`。
- `torch.cat` **每步都新分配一块 `[B, H, t, D]` 并拷贝**，总拷贝量 O(t²)；而且 batch 内所有请求必须同长（靠 padding + attention_mask），一个请求结束了也得陪跑。这正是 vLLM 要解决的痛点。
- `get_seq_length`（L155–159）直接返回 `self.keys.shape[-2]`，即“已缓存长度”，模型用它算 position_ids。

**Stop 2：`Cache.update` 按层分发**

- `Cache.update` L1414–1446：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/cache_utils.py#L1414-L1446
- 关键一行 L1441：`keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)`。`Cache` 只是“每层一个 `CacheLayer`”的容器；`DynamicCache`（L1808 起，`__init__` L1851–1896）在没有 config 时用 `layer_class_to_replicate=DynamicLayer` 懒创建每层。

**Stop 3：模型里在哪调用**

- `repos/00-basics/transformers/src/transformers/models/llama/modeling_llama.py`，`LlamaAttention.forward` L232–270
- 链接：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/models/llama/modeling_llama.py#L232-L270

```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [B, Hq, T, D]
key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [B, Hkv, T, D]
...
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)   # 先 RoPE 再进 cache
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)  # L251
```
注意：**缓存的是加过 RoPE 的 K**（所以位置一旦写入就固定）。模型入口 L372–373 `if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)`，L376–377 用 `get_seq_length()` 推 position_ids。

> 对比：`StaticLayer.update`（L467–499）预分配 `max_cache_len` 并用 `index_copy_` 原地写（L492），地址固定，便于 `torch.compile`/CUDA Graph——这就是“连续预分配”方案，碎片问题依旧。
> 链接：https://github.com/huggingface/transformers/blob/7912d6045eafc60bbd0b2dc33f871105b463dea2/src/transformers/cache_utils.py#L467-L499

---

### 3.2 nano-vllm：最值得逐行读的小型 vLLM（~1200 行）

整体数据流：
```
LLMEngine.step()
  └─ Scheduler.schedule()      → 选出本步的 seqs，BlockManager 分配/复用 block（改 seq.block_table）
  └─ ModelRunner.run(seqs)     → prepare_prefill / prepare_decode 生成 slot_mapping、block_tables、cu_seqlens
       └─ set_context(...)     → 放进全局 Context
       └─ model forward → 每层 Attention.forward: store_kvcache(slot_mapping) + flash_attn_*(block_table)
  └─ Scheduler.postprocess()   → 追加 token、对满块算哈希、结束的请求释放 block
```

**Stop 1：引擎主循环 `LLMEngine.step`**

- `repos/03-inference/nano-vllm/nanovllm/engine/llm_engine.py` L49–55
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py#L49-L55

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```
每调用一次 = 一次 iteration。`generate`（L60–90）就是 `while not self.is_finished(): self.step()`——**这就是 continuous batching**：每一步重新调度。

**Stop 2：`Sequence` —— 一个请求的状态**

- `nanovllm/engine/sequence.py`，`Sequence` L14–83
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py#L14-L70

要点：
- L15 `block_size = 256`（类属性）；L28 `self.block_table = []`：逻辑块 → 物理块号列表。
- `num_cached_tokens`：已经写进 KV cache 的 token 数；`num_scheduled_tokens`：本步要算的 token 数（支持 chunked prefill）。
- `num_blocks`（L55–57）= `ceil(num_tokens / block_size)`；`last_block_num_tokens`（L59–61）最后一块里的 token 数；`block(i)`（L63–65）返回第 i 块的 token_ids，用于哈希。

**Stop 3：`BlockManager` —— 物理块分配 + 引用计数 + 前缀哈希**

- `nanovllm/engine/block_manager.py` L1–120
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L26-L120

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]       # 每个物理块的元数据：ref_count, hash, token_ids
        self.hash_to_block_id: dict[int, int] = dict()            # 前缀哈希 → 物理块
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    @classmethod
    def compute_hash(cls, token_ids, prefix=-1):                  # L35-41
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))                # 链式：把上一块的哈希喂进去
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
```
- `compute_hash` 就是 §1.4 的哈希链。
- `_allocate_block`（L43–51）：从 free 队列头取一块；如果这块以前有哈希且映射还指向它，就**删除旧映射**（这块要被覆盖了，前缀缓存失效）。注意：块被释放（ref_count=0）时**并不清掉哈希**，所以“已释放但还没被重用”的块仍可被前缀命中——这就是 LRU 式的“缓存在空闲块里”。

`can_allocate`（L58–73）：
```python
for i in range(seq.num_blocks - 1):              # 注意 -1：最后一块永远重算，保证至少有 token 产生 logits
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:   # 再比对 token，防哈希碰撞
        break
    num_cached_blocks += 1
    if block_id in self.used_block_ids:          # 命中正在被别人用的块：不消耗空闲块
        num_new_blocks -= 1                      # 命中“空闲中的缓存块”：仍要从 free 里拿走，所以不减
if len(self.free_block_ids) < num_new_blocks:
    return -1
return num_cached_blocks
```
`allocate`（L75–92）：命中的块 `ref_count += 1`（或从 free 队列里捞回来），其余块 `_allocate_block()`；最后 `seq.num_cached_tokens = num_cached_blocks * block_size`——**前缀命中的 token 本步不用算了**。

`deallocate`（L94–101）：倒序 `ref_count -= 1`，降到 0 才放回 free（共享前缀的块要等所有使用者都释放）。

`can_append / may_append`（L103–108）：
```python
def may_append(self, seq):
    if len(seq) % self.block_size == 1:          # 刚追加的 token 是新块的第 1 个 → 需要新块
        seq.block_table.append(self._allocate_block())
```
`hash_blocks`（L110–120）：在 postprocess 时，对本步**新填满**的块（`start..end` 都按整块计）计算哈希并登记到 `hash_to_block_id`。

**Stop 4：`Scheduler.schedule` —— prefill 优先 + 抢占**

- `nanovllm/engine/scheduler.py` L25–79
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L25-L92

```python
# prefill
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    ...
    if not seq.block_table:
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1: break                     # 显存不够，等
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size   # 扣掉前缀命中
    ...
    if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
        break
    seq.num_scheduled_tokens = min(num_tokens, remaining)    # 可能只算一部分 = chunked prefill
    ...
if scheduled_seqs:
    return scheduled_seqs, True                               # 有 prefill 就本步只做 prefill
# decode
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running: self.preempt(self.running.pop())     # 抢占最晚进来的请求
        else: self.preempt(seq); break
    else:
        seq.num_scheduled_tokens = 1
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```
- **token budget**：`max_num_batched_tokens` 限制一步处理多少 token（prefill 的 GEMM 不能无限大）。
- nano-vllm 的策略是“有 prefill 就不做 decode”，prefill 和 decode **不混在一个 batch**（vLLM v1 会混，见 §3.4）。
- `preempt`（L75–79）：直接 `deallocate` 全部 block，放回 waiting 队首——**recompute 式抢占**（下次重新 prefill，前缀缓存可能还能命中一部分）。
- `postprocess`（L81–92）：`hash_blocks` → `num_cached_tokens += num_scheduled_tokens` → chunked prefill 未完成就 `continue`（不采样） → 否则 `append_token`，遇 EOS/长度上限就 `deallocate`。

**Stop 5：`ModelRunner.allocate_kv_cache` —— 一次性占满显存**

- `nanovllm/engine/model_runner.py` L103–121
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L103-L121

```python
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]     # [num_blocks, block_size, Hkv, D]
        module.v_cache = self.kv_cache[1, layer_id]
```
- 先 `warmup_model`（L91–101）跑一次最大 batch 测 peak 激活显存，剩下的全部给 KV。
- 每层拿到的 `k_cache` 形状 `[num_blocks, block_size, Hkv, D]`——正是 FA2 `flash_attn_with_kvcache` 对 paged KV 要求的布局（§3.6）。

**Stop 6：`prepare_prefill / prepare_decode` —— 生成 slot_mapping 和 block_tables**

- `nanovllm/engine/model_runner.py` L123–188
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L123-L188

prefill（多个请求拼成一条 varlen 序列）：
```python
for seq in seqs:
    start = seq.num_cached_tokens                 # 前缀命中/已 chunk 完成的部分
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end                                # K 要包含前缀
    input_ids.extend(seq[start:end]); positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    ...
    for i in range(start_block, end_block):       # 逐块生成本步新 token 的物理槽位
        slot_start = seq.block_table[i] * self.block_size
        ...
        slot_mapping.extend(range(slot_start, slot_end))
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```
- `cu_seqlens_q`：形状 `[num_seqs+1]` 的前缀和，告诉 varlen kernel “扁平 Q `[total_q, Hq, D]` 中第 b 个请求是 `[cu[b], cu[b+1])`”。
- 只有当 K 比 Q 长（有前缀缓存或 chunked prefill 续算）时才需要 block_tables——此时 K/V 必须从 cache 读。

decode：
```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
...
block_tables = self.prepare_block_tables(seqs)    # [B, max_blocks]，不足用 -1 填充
set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
```
`prepare_block_tables`（L123–127）把不等长的 `seq.block_table` 右填 -1 成矩形 int32 张量。

**Stop 7：`Attention.forward` —— 写 cache + 调 FlashAttention**

- `nanovllm/layers/attention.py` L10–75；全局 Context 在 `nanovllm/utils/context.py` L5–27
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L10-L75

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                        # 每个 program 处理一个新 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                         # CUDA graph 填充位
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))     # D = Hkv * head_dim，一整行
    ...
    cache_offsets = slot * D + tl.arange(0, D)    # k_cache 视为 [num_blocks*block_size, Hkv*head_dim]
    tl.store(k_cache_ptr + cache_offsets, key)
```
```python
def forward(self, q, k, v):                       # q: [N, Hq, D]  k,v: [N, Hkv, D]，N = 本步 token 总数
    context = get_context()
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache               # 从 paged cache 读完整前缀
        o = flash_attn_varlen_func(q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,     # q: [B, 1, Hq, D]
                cache_seqlens=context.context_lens, block_table=context.block_tables,
                softmax_scale=self.scale, causal=True)
```
**这就是 block_table / slot_mapping 抵达 kernel 的终点**：
- 写：`slot_mapping` → Triton scatter kernel。
- 读：`block_table` + `cache_seqlens`（decode）或 `cu_seqlens_k`（prefill）→ FA2 kernel。
- 为什么 `Config.__post_init__` 里有 `assert self.kvcache_block_size % 256 == 0`（`nanovllm/config.py` L22）？因为 FA2 的 paged KV 要求 page 大小是 256 的倍数（§3.6）。

**Stop 8：CUDA Graph 只用于 decode**

- `run_model` L195–212，`capture_cudagraph` L222–257
- https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L195-L257

decode 的形状只跟 batch size 有关，于是对 `[1,2,4,8,16,32,...]` 各录一张 graph；replay 前把真实的 `slot_mapping / context_lens / block_tables` 拷进固定地址的 buffer（`graph_vars[...]`），多余位置 `slot_mapping.fill_(-1)`——对应 store kernel 里的 `if slot == -1: return`。

---

### 3.3 mini-sglang：Radix Tree 前缀缓存 + token 级页表

mini-sglang 与 nano-vllm 最大区别：**页表存的是 token 级“物理槽位”而不是块号**，并且前缀用 radix tree 管理。

**Stop 1：全局 page_table**

- `repos/03-inference/mini-sglang/python/minisgl/engine/engine.py` L65–73
- https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L65-L73

```python
# NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
    (config.max_running_req + 1, aligned_max_seq_len), dtype=torch.int32, device=self.device)
```
形状 `[max_running_req+1, max_seq_len]`：`page_table[req.table_idx, pos]` = 第 pos 个 token 在 KV 池里的槽位（等价于 SGLang 的 `req_to_token`）。默认 `page_size = 1`（`engine/config.py` L25）。

**Stop 2：KV 池**

- `kvcache/mha_pool.py` `MHAKVCache` L10–56：https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/mha_pool.py#L10-L56
- `_kv_buffer` 形状 `(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)`（L28–32），`store_kv` 把它 view 成 `(num_pages*page_size, Hkv, D)` 再按 `out_loc`（就是 slot_mapping）写入（L45–56）。

**Stop 3：Radix tree 本体**

- `kvcache/radix_cache.py` L17–237：https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/radix_cache.py#L17-L237

```python
class RadixTreeNode:
    def __init__(self, key_fn, tic=None):
        self.children: Dict[Any, RadixTreeNode] = {}   # 子节点按“首个 page 的 token”索引
        self.ref_count: int = 0                         # >0 表示有正在运行的请求在用，不能驱逐
        self.timestamp = tic or time.monotonic_ns()     # LRU 用
        self._key: torch.Tensor     # 这条边上的 token_ids
        self._value: torch.Tensor   # 这些 token 在 KV 池中的槽位
```
- `_tree_walk`（L205–231）：从根往下，`children.get(key_fn(input_ids[prefix_len:]))` 找子节点，`get_match_len` 比对公共前缀长度（按 page 向下对齐），没走完一条边就 `split_at` 把节点劈成两段（L69–81）。
- `match_prefix`（L132–134）返回一个 handle；`RadixCacheHandle.get_matched_indices`（L91–98）沿父指针把各段 `value` 拼起来 = 命中前缀的 KV 槽位。
- `insert_prefix`（L136–146）：请求算完后把 `(input_ids, indices)` 插回树里，新增部分挂成新叶子。
- `lock_handle`（L113–130）：沿路径 ref_count ±1，维护 `evictable_size / protected_size`。
- `evict`（L148–175）：收集 `ref_count==0` 的叶子放小根堆（按 timestamp，即 LRU），弹出并删掉；父节点变成叶子后也入堆。**只驱逐叶子**保证前缀树的结构不被破坏。
- `_get_key_fn`（L234–237）：`page_size==1` 用第一个 token 当 key，否则用前 `page_size` 个 token 的 tuple。

**Stop 4：CacheManager —— 分配、写页表、回收**

- `scheduler/cache.py` L15–146：https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L15-L146

```python
def match_req(self, req):
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])   # 同样留最后一个 token 重算

def allocate_paged(self, reqs):
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        ...
    allocated = self._page_to_token(self._allocate(needed_pages))     # 页号 → token 槽位
    _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```
- `_allocate`（L106–113）：free 不够就 `prefix_cache.evict(...)` 驱逐树叶。
- `cache_req`（L55–79）的注释是理解前缀缓存“谁负责释放哪段”的最好材料：已经在树里的部分要 `_free` 掉自己那份重复分配，finished 时尾巴（不满一页、不能入树）也要释放。

**Stop 5：metadata 到 kernel**

- `scheduler/scheduler.py` `_prepare_batch` L204–211：`batch.out_loc = self.engine.page_table[input_mapping]`（slot_mapping 从页表里取出来）。
- `attention/fa.py` `prepare_metadata` L67–105、`forward` L48–65、`_fa_sgl_impl` L139–182：https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/attention/fa.py#L48-L182

```python
page_table = get_global_ctx().page_table
new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
    [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs])
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")   # token 槽位 → 页号
```
然后调用 `sgl_kernel.flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, page_table=..., cache_seqlens=..., cu_seqlens_q=..., cu_seqlens_k_new=..., max_seqlen_q=...)`（FA3 风格接口，见 `flash-attention/hopper/flash_attn_interface.py` 的 `flash_attn_with_kvcache` L942 起，参数名就是 `page_table`、`cu_seqlens_q`）。**同一个函数同时处理 prefill（`cu_seqlens_q` 不全是 1）和 decode（`max_seqlen_q == 1`，L84–85 直接用 `arange`）**。

---

### 3.4 vLLM v1：工业级的同一套思想

**Stop 1：块元数据与空闲链表**

- `repos/03-inference/vllm/vllm/v1/core/kv_cache_utils.py`：`KVCacheBlock` L154–178，`FreeKVCacheBlockQueue` L224 起，`hash_block_tokens` L673–703
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/core/kv_cache_utils.py#L154-L246

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None      # 满块且已缓存才有
    prev_free_block: "KVCacheBlock | None" = None        # 双向链表指针，O(1) 从中间摘除
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False
```
`FreeKVCacheBlockQueue` 的 docstring 说明：空闲队列本身就是 **LRU 驱逐顺序**（队头最久未用）；之所以自己写双向链表而不用 deque，是为了“前缀命中时把一个空闲块从队列中间摘出来”是 O(1)。

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```
与 nano-vllm 的 `compute_hash` 完全同构，只是多了 `extra_keys`（LoRA id、多模态 hash 等，避免不同 adapter 误共享）。

**Stop 2：`BlockPool`**

- `vllm/v1/core/block_pool.py` L136–805
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/core/block_pool.py#L136-L196

| 方法 | 行号 | 作用 | nano-vllm 对应 |
|---|---|---|---|
| `__init__` | L157–196 | 建所有块、空闲队列、`cached_block_hash_to_block`；块 0 当 `null_block` 占位 | `BlockManager.__init__` |
| `get_cached_block` | L198–224 | 按哈希查缓存块 | `hash_to_block_id.get` |
| `get_new_blocks` | L666–700 | 从队头弹 n 块，`_maybe_evict_cached_block` 清掉旧哈希，`ref_cnt += 1` | `_allocate_block` |
| `touch` | L752–768 | 命中缓存：ref_cnt==0 的先从空闲队列摘掉，再 +1 | `allocate` 中的命中分支 |
| `free_blocks` | L774–805 | ref_cnt−1；无哈希的块放队头（尽快复用），有哈希的放队尾（尽量保留缓存） | `deallocate` |

`free_blocks` 那段很值得看：
```python
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)    # LIFO reuse of non-cached blocks for better GPU locality.
else:
    blocks_to_evict_last.append(block)     # FIFO reuse of cached blocks for LRU eviction behavior.
self.free_block_queue.prepend_n(blocks_to_evict_first)
self.free_block_queue.append_n(blocks_to_evict_last)
```

**Stop 3：`KVCacheManager.get_computed_blocks / allocate_slots`**

- `vllm/v1/core/kv_cache_manager.py` L264–321，L371–470
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/core/kv_cache_manager.py#L264-L321

```python
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
max_cache_hit_length = request.num_tokens - 1
computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length))
```
“最后一个 token 必须重算”这条规则，三个系统（nano-vllm 的 `num_blocks - 1`、mini-sglang 的 `input_len - 1`、vLLM 的 `num_tokens - 1`）都有。

`allocate_slots` 的 docstring（L420–452）画了一张非常清楚的块布局图：`| comp | new_comp | ext_comp | new | lookahead |`——已计算的、本次前缀命中的、外部 connector（如 KV offload / P-D 分离）提供的、本步要算的、投机解码预留的。

**Stop 4：Scheduler —— 没有 prefill/decode 阶段之分**

- `vllm/v1/core/sched/scheduler.py` `schedule` L591 起
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/core/sched/scheduler.py#L591-L611

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec. ...
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
```
- 先排 RUNNING（L657 起）：每个请求要几个新 token（decode=1，chunked prefill 续算=一段），`allocate_slots`（L778）不够就 `self.running.pop()` 抢占队尾（L834–841）。
- 再排 WAITING（L899 起）：`get_computed_blocks`（L1037）查前缀，`allocate_slots`（L1256）。
- 共用一个 `token_budget`（L611）。因此**一个 batch 里同时有 decode 请求和 prefill chunk**——这是比 nano-vllm 更“连续”的 continuous batching，prefill 不会把 decode 卡住。

**Stop 5：slot_mapping 在 GPU 上算**

- `vllm/v1/worker/block_table.py` `compute_slot_mapping` L153–179，`ComputeSlotMappingKernel.kernel` L305–376
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/worker/block_table.py#L305-L376

```python
req_idx = tl.program_id(0)                                   # 每个请求一个 program
start_idx = tl.load(query_start_loc_ptr + req_idx)           # 本请求在扁平 token 维的区间
end_idx = tl.load(query_start_loc_ptr + req_idx + 1)
...
pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
virtual_block_indices = pos // virtual_block_size            # 逻辑块号
virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
block_numbers = tl.load(block_table_ptr + row_offset + virtual_block_indices, ...)  # 查页表
slot_ids = block_numbers * KV_CACHE_BLOCK_SIZE + local_block_offsets
```
忽略 context-parallel（`TOTAL_CP_WORLD_SIZE=1` 时 `is_local` 恒真），就是我们 toy 里的 `bt[pos // bs] * bs + pos % bs`。最后一个 program（L337–346）专门把 CUDA graph 的填充位写成 `PAD_ID`。

**Stop 6：FlashAttention backend**

- `vllm/v1/attention/backends/flash_attn.py`：`FlashAttentionMetadata` L513–545，`FlashAttentionImpl.forward` L1177–1425，`do_kv_cache_update` L1427 起
- https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/v1/attention/backends/flash_attn.py#L1177-L1460

```python
# |---------- N-1 iteration --------|
# |---------------- N iteration ---------------------|
# |- tokenA -|......................|-- newTokens ---|
# |---------- context_len ----------|
# |-------------------- seq_len ---------------------|
#                                   |-- query_len ---|
query_start_loc: torch.Tensor     # = cu_seqlens_q
seq_lens: torch.Tensor            # = seqused_k（每个请求 K 的总长）
block_table: torch.Tensor
slot_mapping: torch.Tensor
```
forward 中：
```python
cu_seqlens_q = attn_metadata.query_start_loc        # L1276
seqused_k = attn_metadata.seq_lens
block_table = attn_metadata.block_table
...
_FA4_DENSE_ATTENTION_KERNEL(q=query[:num_actual_tokens], k=key_cache, v=value_cache, out=...,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
    causal=causal, block_table=block_table, scheduler_metadata=scheduler_metadata, num_splits=num_splits, ...)
```
`FA4DenseAttentionKernel.kernel`（L114–116）最终就是 `flash_attn_varlen_func(*args, **kwargs)`。**注意 vLLM 用 `seqused_k` 而不是 `cu_seqlens_k`**：因为 K 不是一条扁平的连续张量，而是在 paged cache 里，kernel 只需要知道“每个请求用了多少个 key”，位置靠 `block_table` 找。docstring（L1196–1197）说明此处 `kv_cache` 形状为 `[num_blocks, num_kv_heads, block_size, 2 * head_size]`（K 和 V 拼在最后一维），`do_kv_cache_update` 里 `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` 拆成 `(B, N, H, D)` 的 K/V 视图，再 `reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)` 写入（L1443–1460）。

---

### 3.5 SGLang：RadixAttention 的完整版

**Stop 1：两级映射 `ReqToTokenPool` + `TokenToKVPool`**

- `repos/03-inference/sglang/python/sglang/srt/mem_cache/memory_pool.py`：`ReqToTokenPool` L374–412，`MHATokenToKVPool` L2198 起，`set_kv_buffer` L2825 起
- https://github.com/sgl-project/sglang/blob/3f1e185f6e83ca02ec19498aa2917ccaa0f0fd43/python/sglang/srt/mem_cache/memory_pool.py#L374-L412

```python
# +1 padding row at index 0: cuda-graph padded batches default
# req_pool_indices to 0, so dummy reads/writes land here harmlessly.
self._alloc_size = size + 1
self.req_to_token = torch.zeros((self._alloc_size, max_context_len), dtype=torch.int32, device=device)
```
`req_to_token[req_idx, pos]` = 第 pos 个 token 的 KV 槽位（和 mini-sglang 的 page_table 一样是 token 级）。`MHATokenToKVPool` 持有每层的 `k_buffer/v_buffer`，`set_kv_buffer(layer, loc, cache_k, cache_v)` 按 `loc`（slot_mapping）写入。

**Stop 2：`RadixCache`**

- `mem_cache/radix_cache.py`：`TreeNode` L250–278，`match_prefix` L364–417，`_match_prefix_helper` L651–675，`_split_node` L677–700，`_insert_helper` L705–760，`evict` L552–580，`inc_lock_ref` L582–595
- https://github.com/sgl-project/sglang/blob/3f1e185f6e83ca02ec19498aa2917ccaa0f0fd43/python/sglang/srt/mem_cache/radix_cache.py#L651-L760

```python
def _match_prefix_helper(self, node, key):
    child_key = key.child_key(self.page_size)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):                    # 只匹配了这条边的一部分
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)                      # value = 这段 token 的 KV 槽位
            node = child
            key = key[prefix_len:]
            ...
    return value, node
```
与 mini-sglang `_tree_walk` 逐行对应；SGLang 多了：`lock_ref`（= ref_count）、`hit_count`、`priority`（优先级驱逐）、`extra_key`（LoRA 隔离，见 `match_prefix` docstring L367–377）、KV 事件上报、HiCache（host 内存二级缓存，`TreeNode.evicted` 指 value 被换出）。`evict`（L552–580）同样是“可驱逐叶子进堆 → 弹出 → `free_segment(x.value)` → 父节点变叶子再入堆”。

**Stop 3：`RadixAttention` 层**

- `python/sglang/srt/layers/radix_attention.py`：`class RadixAttention` L98，`forward` L159 起，最终 L294 `return get_attn_backend().forward(...)`
- https://github.com/sgl-project/sglang/blob/3f1e185f6e83ca02ec19498aa2917ccaa0f0fd43/python/sglang/srt/layers/radix_attention.py#L159-L294

名字叫 RadixAttention，但这个层本身只是把 q/k/v 交给 attention backend（FA3、FlashInfer、Triton…）；“radix”体现在调度器/内存层用 radix tree 决定哪些 KV 可以复用、`req_to_token` 怎么填。backend 读 `req_to_token` 得到页表交给 kernel。

---

### 3.6 终点：block_table 在 FlashAttention kernel 里怎么用

**Python 接口（FA2）**

- `flash-attention/flash_attn/flash_attn_interface.py`：`flash_attn_varlen_func` L1391 起（`block_table=None` 参数在 L1407），`flash_attn_with_kvcache` L1485–1627
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/flash_attn/flash_attn_interface.py#L1485-L1627

docstring 里的形状约定（L1549–1568）：
```
q: (batch_size, seqlen, nheads, headdim)
k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
    or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
    page_block_size must be a multiple of 256.
cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the KV cache.
block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
```
C++ 侧 `flash-attention/csrc/flash_attn/flash_api.cpp` L613（varlen）与 L1305（kvcache）都有
`TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");`
https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/flash_api.cpp#L1286-L1305

**kernel 内部（FA2 split-KV kernel）**

- `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h` L591–601 与 L950–957
- https://github.com/zhiliangzhang/flash-attention/blob/e8a6ad52291b7aaf5448fd3ae1dde7a362dcfe92/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h#L591-L601

```cpp
const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
const index_t row_offset_k = block_table == nullptr
    ? ... (连续布局)
    : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
```
逐项解释：
- `bidb` 是当前 CTA 负责的请求，`block_table` 指到它那一行。
- kernel 从最后一个 K tile（`n_block_max - 1`）往前处理，tile 大小 `kBlockN`（64/128/256）。第一个 key 的逻辑位置 `(n_block_max-1)*kBlockN` 除以 `page_block_size` 得逻辑页号、余数得页内偏移。
- **物理地址 = 物理页号 × `k_batch_stride`（一页的跨度）+ 页内行 × `k_row_stride` + KV 头偏移**。`bidh / h_h_k_ratio` 就是 GQA：第 bidh 个 query 头读第 `bidh / G` 个 KV 头。
- L950–957 每处理完一个 tile，就用“下一个 tile 的物理页 − 当前物理页”的差值移动指针。
- 因为一个 K tile（`kBlockN` ≤ 256 行）必须完整落在一页里，指针运算才这么简单——**这就是 page 必须是 256 倍数的原因**。FA3/FA4 用 TMA 或逐行 gather 放宽了这一限制（FA4 见 04 篇 §3.5 的 `PagedKVManager`）。

---

## 4. 不同实现对比

| 维度 | HF DynamicCache | nano-vllm | mini-sglang | vLLM v1 | SGLang |
|---|---|---|---|---|---|
| KV 布局 | 每层 `[B,H,T,D]` 连续，`torch.cat` 增长 | `[2,L,num_blocks,block_size,Hkv,D]` 一次性预分配 | `[2,L,num_pages,page_size,Hkv,D]` | 每层 `[num_blocks,Hkv,block_size,2D]`（FA backend） | `k_buffer/v_buffer` 每层 `[size+page, Hkv, D]` |
| 页表粒度 | 无 | 块号 `seq.block_table` | token 槽位 `page_table[req, pos]` | 块号 `block_table`（GPU CpuGpuBuffer） | token 槽位 `req_to_token` |
| 默认 page 大小 | — | 256（受 FA2 限制，`config.py` L17/L22） | 1（`engine/config.py` L25） | 16（`vllm/config/cache.py` L72 `DEFAULT_BLOCK_SIZE`） | CUDA 上默认 1（`srt/arg_groups/overrides.py` `_page_size_default` L1304–1325） |
| 前缀复用 | 无 | 链式 xxhash，满块 | radix tree | 链式哈希 + extra_keys，满块 | radix tree + lock_ref/优先级/HiCache |
| 驱逐 | — | 释放块保留哈希，FIFO 复用 | LRU 叶子堆 | 空闲双向链表即 LRU | 可插拔策略堆 |
| 调度 | 静态 batch | prefill 优先、不混批；只第一个请求可 chunk | prefill/decode 分批 | 统一 token budget，prefill chunk 与 decode 混批 | 混合/分离多种模式 |
| 抢占 | — | recompute（释放全部块） | — | recompute（`_preempt_request`） | retract（`ScheduleBatch.retract_decode`，`managers/schedule_batch.py` L3294） |
| 写 KV | `torch.cat` | Triton `store_kvcache_kernel` | `store_cache` | `reshape_and_cache_flash` | `set_kv_buffer` |
| 读 KV kernel | SDPA/eager | FA2 `flash_attn_varlen_func` / `flash_attn_with_kvcache` | sgl_kernel FA3 `flash_attn_with_kvcache(page_table=...)` | vllm_flash_attn `flash_attn_varlen_func(seqused_k, block_table)`（FA2/3/4） | FA3/FlashInfer/Triton backend |

---

## 5. 常见困惑 / 坑

1. **block_table 和 slot_mapping 有什么区别？** block_table 是“每请求、每逻辑块”一项，给**读**用（kernel 读全部历史 K/V）；slot_mapping 是“每个本步新 token”一项，给**写**用（只写新 K/V）。decode 时 slot_mapping 长度 = batch size，block_table 是 `[B, max_blocks]`。
2. **为什么前缀命中后还要重算最后一个 token？** 采样需要最后位置的 logits，而 logits 不缓存。所以三个系统都把可命中长度上限设为 `len - 1`（nano-vllm 甚至以块为单位少算一整块）。
3. **哈希为什么要链式？** 同样的 16 个 token 出现在不同前缀后面，K/V 完全不同（K 带了 RoPE 位置、而且经过前面层的 attention 混合了上下文）。链式哈希保证“相同哈希 ⇔ 相同的完整前缀”。nano-vllm 还会再比 `token_ids` 防碰撞（`can_allocate` L66）。
4. **为什么只有满块才能共享？** 未满块后续还要被同一请求追加写，若共享会被别人的 token 覆盖。radix tree 方案（page_size=1）则可以 token 级共享，这是 SGLang 的优势之一。
5. **FA2 paged 要求 256 的倍数** → nano-vllm `kvcache_block_size % 256 == 0`。用 16 的 block 去调 FA2 会直接报错；vLLM 自带的 vllm_flash_attn / FA3 / FA4 / FlashInfer 都支持小 page。
6. **`cu_seqlens_k` vs `seqused_k` vs `cache_seqlens`**：`cu_seqlens_k` 是扁平连续 K 的前缀和（K 是一条 `[total_k, H, D]`）；有 paged cache 时 K 不扁平，于是用 `seqused_k`/`cache_seqlens`（每个请求一个长度）。FA4 `_flash_attn_fwd` 里直接断言 `page_table` 与 `cu_seqlens_k` 互斥（`interface.py` L765）。
7. **causal mask 在 decode/前缀续算时的对齐**：`flash_attn_with_kvcache` docstring（L1530–1540）说明 causal 对齐**右下角**——seqlen_q=2、seqlen_k=5 时第一行能看到前 4 个 key。这正是“新 query 位于序列末尾”的语义，所以 nano-vllm decode 也传 `causal=True` 而不会错。
8. **释放了的块还算缓存吗？** nano-vllm/vLLM 中，ref_count=0 的块进 free 队列但保留哈希，直到被重新分配时才失效（`_allocate_block` L47–48、`_maybe_evict_cached_block`）。所以“空闲显存”和“前缀缓存”是同一批块。
9. **CUDA Graph 下 padding 的请求会不会乱写 KV？** nano-vllm 用 `slot_mapping=-1` 跳过写入；vLLM 写 `PAD_ID`；SGLang/mini-sglang 让 padding 请求指向 index 0 的 dummy 行（`ReqToTokenPool` 注释 L398–399，mini-sglang `engine.py` L69 的 “+1 for dummy request”）。
10. **DynamicCache 里存的 K 已经做过 RoPE**；paged 系统同理（nano-vllm `Qwen3Attention` 里先 rotary 再进 `self.attn`）。因此你不能简单地“挪动”缓存位置（StreamingLLM 一类方法要额外处理）。

---

## 6. 自测题 + 动手练习

**自测题**
1. Qwen2.5-7B（28 层、4 个 KV 头、head_dim 128、bf16）每个 token 的 KV 多少字节？80 GB 卡留 40 GB 给 KV，block_size=16 时能有多少个 block？
2. 一个请求长度 1000，block_size=256，它的 `block_table` 有几项？最后一块有多少 token？下一次 decode 时 `may_append` 会不会分配新块？
3. nano-vllm `can_allocate` 中，命中一个**空闲中的缓存块**为什么不减 `num_new_blocks`？
4. 为什么 vLLM 的 attention 调用传 `seqused_k` 而 nano-vllm prefill 传 `cu_seqlens_k`？nano-vllm 在什么情况下 prefill 也要传 `block_table`？
5. radix tree 的 `evict` 为什么只驱逐叶子？如果驱逐中间节点会发生什么？
6. vLLM 调度器里“没有 prefill/decode 阶段”带来什么好处？对 decode 的 TPOT 抖动有什么影响？
7. FA2 kernel 中 `bidh / params.h_h_k_ratio` 实现了什么？

**动手练习**
1. 运行 §2 的 toy：把 `block_size` 改成 1、4、16，观察 `block_tables` 的碎片情况，统计平均浪费的槽位。
2. 给 `ToyPagedKV` 加上 `ref_count` 与 `hash_to_block`，实现 `fork(sid_src, sid_dst)`：共享所有满块（ref+1），最后一个未满块做 copy-on-write——这就是 beam search / parallel sampling 的共享机制。
3. 实现一个极简 radix tree（token 级，page_size=1）：`match(tokens) -> (prefix_len, slots)`、`insert(tokens, slots)`、`evict(n)`，对照 mini-sglang `radix_cache.py` L132–231 检查。
4. 在 §2 的 `paged_decode_attention` 里去掉 gather：改成按块循环、每块算一次局部 softmax 并用 04 篇的 online softmax 合并——这就是 paged attention kernel 的真正写法。
5. 读 vLLM `ComputeSlotMappingKernel.kernel`，用 numpy 写一个等价的 `compute_slot_mapping(query_start_loc, positions, block_table, block_size)`，并和 nano-vllm `prepare_prefill` 的结果对拍。
