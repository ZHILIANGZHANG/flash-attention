# 12 · 量化（Quantization）源码导读：RTN / GPTQ / AWQ 与推理引擎

> 对应源码：
> - `repos/11-quantization/gptq`（IST-DASLab/gptq @ `2d65066e`）：`gptq.py`、`quant.py`、`llama.py`
> - `repos/11-quantization/llm-awq`（mit-han-lab/llm-awq @ `d6e797a4`）：`awq/quantize/{pre_quant,auto_scale,auto_clip,quantizer,qmodule}.py`
> - `repos/03-inference/vllm`（vllm-project/vllm @ `f5fefbff`）：`vllm/model_executor/layers/quantization/`
>
> 强烈建议先读 [guide 10 的 SparseGPT 部分](./10-model-pruning.md#37-sparsegptlibsparsegptpy)：GPTQ 与 SparseGPT 出自同一组作者，**代码骨架几乎逐行相同**，只是把"置 0"换成了"四舍五入到网格"。

---

## 0. 要解决的问题 / 一句话直觉

**问题**：7B 模型 bf16 要 14GB。decode 阶段每生成一个 token 都要把所有权重从显存读一遍——**带宽瓶颈**。把权重压到 4 bit，显存降到 ~3.5GB，读权重的时间也降到 1/4，decode 速度显著提升。

**一句话直觉**：

- **量化** = 用一个 "比例尺 scale + 零点 zero" 把浮点数映射到少量整数格点上：`q = round(w/s) + z`，`ŵ = s·(q − z)`。
- **RTN（Round-To-Nearest）**：每个权重直接四舍五入。简单，但 4bit 以下误差大。
- **GPTQ**：量化一列权重会产生误差，那就**调整还没量化的列来补偿**——和 SparseGPT 用同一套二阶（Hessian `H = XXᵀ`）OBS 数学。
- **AWQ**：1% 的输入通道激活特别大（"显著通道"），它们对应的权重量化误差会被放大。给这些通道的权重**乘一个 s>1 再量化**（相对误差变小），再把 `1/s` 吸收进上一个算子（LayerNorm 或上一层 Linear），数学上输出不变。

---

## 1. 核心原理（带张量形状）

### 1.1 量化映射

设 b bit，`maxq = 2^b − 1`。

| | 映射 | scale / zero |
|---|---|---|
| 非对称 (asymmetric) | `q = clamp(round(w/s) + z, 0, maxq)`，`ŵ = s(q − z)` | `s = (max − min)/maxq`，`z = round(−min/s)` |
| 对称 (symmetric) | 同上但 z 固定为中点，或用有符号整数 `q ∈ [−2^{b−1}, 2^{b−1}−1]`，`ŵ = s·q` | `s = max|w| / (2^{b−1}−1)` 或 `2max|w|/maxq` |

非对称能利用 `[min, max]` 不对称的分布（如 ReLU/SiLU 之后的激活），对称则计算更省（无需减 zero）。

### 1.2 粒度（scale 共享的范围）

对 `W ∈ ℝ^{C_out × C_in}`：

| 粒度 | scale 形状 | 说明 |
|---|---|---|
| per-tensor | `()` | 整个矩阵一个 s；fp8 W8A8 常用 |
| per-channel（per-output-row） | `(C_out, 1)` | 每行一个；GPTQ 默认 `perchannel=True` |
| per-group（group size g） | `(C_out, C_in/g)` | 每行每 g 个连续输入元素一个；4bit 主流 g=128 |
| per-token（激活） | `(N, 1)` | 激活量化常用，动态计算 |
| per-block（如 128×128） | `(C_out/128, C_in/128)` | DeepSeek-V3 式 fp8 |

group 越小精度越高，但额外存储 `s,z`：g=128、fp16 scale + 4bit zero ≈ 每权重多 `(16+4)/128 ≈ 0.16` bit。

### 1.3 量化什么

| 方案 | 权重 | 激活 | 收益 | 典型 |
|---|---|---|---|---|
| Weight-only (W4A16 / W8A16) | int4/int8 | bf16/fp16 | 省显存、decode 带宽；计算仍是 fp16（kernel 里反量化） | GPTQ、AWQ + Marlin |
| W8A8 (int8 / fp8) | int8/fp8 | int8/fp8 | 能用 int8/fp8 Tensor Core，prefill 计算也加速 | SmoothQuant、fp8 |
| KV cache 量化 | — | K/V fp8/int8 | 长上下文显存，注意力带宽 | vLLM `kv_cache_dtype=fp8` |

fp8 有两种：`E4M3`（精度高、范围 ±448，常用于前向）与 `E5M2`（范围大、精度低）。fp8 是浮点格式，量化时只需一个 scale（无 zero），`ŵ = s · fp8(w/s)`。int4 只有 16 个格点，必须配合 group 量化与 GPTQ/AWQ 这类误差补偿方法。

### 1.4 RTN

逐元素 `round`。误差项 `δ = ŵ − w` 近似均匀分布在 `[−s/2, s/2]`，方差 `s²/12`。层输出误差 `‖δ X‖²` 没有被利用来优化。

### 1.5 GPTQ：OBS 用于量化

逐层目标（与 guide 10 相同）：

```
min_{Ŵ ∈ 网格} ‖W X − Ŵ X‖²_F,     X ∈ ℝ^{C_in × N}
```

按行可分，每行误差 `δwᵀ H δw`，`H = 2XXᵀ ∈ ℝ^{C_in×C_in}`（所有行共享）。

OBQ（Optimal Brain Quantization）：量化第 q 个权重（不是置 0 而是取最近格点 `quant(w_q)`），其余权重的最优补偿

```
δw = − (w_q − quant(w_q)) / [H⁻¹]_qq · H⁻¹[:, q]
```

然后把 q 从 H 中消去，再选下一个。GPTQ 的三个工程洞察：

1. **任意固定顺序几乎和贪心一样好** → 所有行按**相同的列顺序**（从左到右）量化 → 所有行共享同一串 `H⁻¹` 子矩阵 → 一次 Cholesky 搞定（与 SparseGPT 的推导完全相同：`H⁻¹ = UᵀU`，第 i 步用 `U[i, i:]`）。
2. **Lazy batch updates**：块内（128 列）逐列更新，块间一次 GEMM 推送误差（提高算术强度，GPU 友好）。
3. **Cholesky 重构**：避免反复做 rank-1 降阶时的数值误差累积（大模型上会让 H⁻¹ 失去正定性）。

**act-order（desc_act）**：先量化 `diag(H)`（≈ 该输入通道激活能量 `Σx_j²`）大的列。直觉：重要列先量化时，后面还有很多列可以"吸收"它的误差；若最后才量化，就没有补偿余地了。代价：如果同时用 group 量化，按置换后顺序分组会让同一组的列在原矩阵里**不连续**，推理时需要 `g_idx` 做 gather（kernel 变慢），于是有了 `static_groups`（先按原顺序定好每组的 scale，再按 act-order 顺序量化）。

### 1.6 AWQ：激活感知的等价缩放

对 Linear `y = W x`，任取逐输入通道的正向量 `s ∈ ℝ^{C_in}`：

```
y = W x = (W · diag(s)) · (diag(s)⁻¹ x)
```

对 `W' = W diag(s)` 量化。对某个通道 j：量化误差 `Δ_j ≈ s'·RoundErr`（s' 是该组的量化步长），若只放大少数显著通道，组内最大值基本不变 → 步长 s' 不变 → 该通道权重的**相对**误差缩小为原来的 `1/s_j`，输出误差 `|Δ_j · x_j / s_j|` 随之缩小。

搜索空间（一维）：

```
s = mean_tokens(|x|)^α,     α ∈ {0, 0.05, …, 0.95}
s ← s / sqrt(max(s) · min(s))           （归一化，避免整体缩放）
α* = argmin_α ‖ Block(x; Q(W·diag(s))·diag(s)⁻¹) − Block(x; W) ‖²
```

**折叠（fold）**：`diag(s)⁻¹ x` 这一步不能真在运行时做（那就多了一次逐元素除法），而是吸收进产生 x 的那个算子：

- x 来自 RMSNorm/LayerNorm：`γ ← γ / s`（Llama 的 q/k/v 与 gate/up 输入）
- x 来自上一个 Linear 的输出：上一层权重的对应**输出行** `W_prev[j,:] ← W_prev[j,:] / s_j`（`v_proj → o_proj`，`up_proj → down_proj`）
- x 来自激活函数且无法吸收：插入 `ScaledActivation`（显式除）

此外 AWQ 还做 **clip 搜索**：把每组的 `max|w|` 缩小到 `(1 − i/20)` 倍再量化，选输出误差最小的截断阈值（牺牲离群权重，换更细的步长）。

---

## 2. 最小实现（自己写的 PyTorch 片段）

### 2.1 Group-wise int4 RTN 量化 / 反量化 + 打包

```python
import torch

def quant_group(w: torch.Tensor, bits=4, group=128, sym=False):
    """w: (C_out, C_in) -> q:(C_out,C_in) uint8, scale/zero:(C_out, C_in//group)"""
    C_out, C_in = w.shape
    assert C_in % group == 0
    maxq = 2 ** bits - 1
    wg = w.float().view(C_out, C_in // group, group)
    if sym:
        amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        scale = 2 * amax / maxq
        zero = torch.full_like(scale, (maxq + 1) / 2)            # 4bit 时 z=8，与 GPTQ sym 相同
    else:
        mx = wg.amax(-1, keepdim=True).clamp_min(0)              # 保证 0 可被精确表示
        mn = wg.amin(-1, keepdim=True).clamp_max(0)
        scale = ((mx - mn) / maxq).clamp_min(1e-8)
        zero = torch.round(-mn / scale)
    q = torch.clamp(torch.round(wg / scale) + zero, 0, maxq)
    return q.to(torch.uint8).view(C_out, C_in), scale.squeeze(-1), zero.squeeze(-1)

def dequant_group(q, scale, zero, group=128):
    C_out, C_in = q.shape
    qg = q.float().view(C_out, C_in // group, group)
    return ((qg - zero[..., None]) * scale[..., None]).view(C_out, C_in)

def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """沿输入维把 8 个 4bit 打进一个 int32（GPTQ 式: (C_in//8, C_out)）。"""
    qt = q.t().contiguous().to(torch.int32)                        # (C_in, C_out)
    qt = qt.view(-1, 8, qt.shape[1])                               # (C_in/8, 8, C_out)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).view(1, 8, 1)
    return (qt << shifts).sum(1, dtype=torch.int32)                # 位不重叠，求和 == 按位或

W = torch.randn(256, 512)
q, s, z = quant_group(W)
print((W - dequant_group(q, s, z)).abs().mean(), pack_int4(q).shape)   # -> (64, 256)
```

（int32 溢出说明：最高的 nibble 移到 bit 28–31，`sum` 在 int32 上按补码回绕，结果的比特模式与按位或一致。）

### 2.2 迷你 GPTQ（含 group、act-order）

```python
@torch.no_grad()
def gptq_quant(W, X, bits=4, group=128, blocksize=128, percdamp=0.01, actorder=False):
    """W:(C_out,C_in) X:(C_in,N)。要求 blocksize % group == 0。返回反量化后的 Q。"""
    W = W.clone().float()
    C_out, C_in = W.shape
    maxq = 2 ** bits - 1
    H = 2 * X @ X.T / X.shape[1]
    if actorder:
        perm = torch.argsort(H.diag(), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H += percdamp * H.diag().mean() * torch.eye(C_in)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, C_in, blocksize):
        i2 = min(i1 + blocksize, C_in)
        W1, U1 = W[:, i1:i2].clone(), U[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if i % group == 0:                                   # 用"已补偿"的权重确定本组 scale/zero
                g = W1[:, i:i + group]
                mx, mn = g.amax(1).clamp_min(0), g.amin(1).clamp_max(0)
                s = ((mx - mn) / maxq).clamp_min(1e-8); z = torch.round(-mn / s)
            w = W1[:, i]
            q = (torch.clamp(torch.round(w / s) + z, 0, maxq) - z) * s
            Q[:, i1 + i] = q
            err = (w - q) / U1[i, i]
            W1[:, i:] -= err[:, None] * U1[i, i:][None, :]        # 补偿本块后续列
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]                         # lazy batch update
    if actorder:
        Q = Q[:, torch.argsort(perm)]
    return Q

torch.manual_seed(0)
W = torch.randn(64, 256)
X = torch.randn(256, 2048); X[:8] *= 20                          # 制造几个"显著通道"
rtn = dequant_group(*quant_group(W, group=128), group=128)
gq = gptq_quant(W, X, group=128)
print(((W - rtn) @ X).norm(), ((W - gq) @ X).norm())             # GPTQ 的输出误差应更小
```

和 2.1 的 RTN 相比，GPTQ 唯一的区别是那两行"补偿"。注意 act-order 时这里的 group 是按**置换后**顺序划分的——真实部署需要额外记录 `g_idx`（见 3.3 与 3.6）。

### 2.3 迷你 AWQ（单个 Linear 的 scale 搜索）

```python
@torch.no_grad()
def awq_search_scale(linear: torch.nn.Linear, x: torch.Tensor, bits=4, group=128, n_grid=20):
    """x: (N, C_in) 该层的校准输入。返回最优 s:(C_in,)。"""
    W = linear.weight.float()
    ref = x.float() @ W.T                                          # (N, C_out)
    x_mean = x.abs().float().mean(0)                               # (C_in,) 每通道平均幅值
    best, best_s = float("inf"), None
    for k in range(n_grid):
        a = k / n_grid
        s = x_mean.pow(a).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        Wq = dequant_group(*quant_group(W * s, bits, group), group) / s   # Q(W·diag(s))·diag(s)^-1
        err = (x.float() @ Wq.T - ref).pow(2).mean().item()
        if err < best:
            best, best_s = err, s
    return best_s

def fold_into_rmsnorm(norm_weight: torch.Tensor, linears, s):
    norm_weight.div_(s)                                            # γ ← γ / s
    for l in linears:
        l.weight.mul_(s.view(1, -1))                               # W ← W·diag(s)，之后再量化
```

`α=0` 时 `s≡1` 就是 RTN，所以 AWQ 的搜索结果**不会比 RTN 差**（在校准集上）。

---

## 3. 源码导读（按阅读顺序）

### 3.1 GPTQ：量化原语 `quant.py`

**(a) `quantize()`：网格映射** —— [quant.py#L6-L10](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py#L6-L10)

```python
def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)
```

返回的是**反量化后的浮点值**（"伪量化 / fake quant"），所以 GPTQ 算法全程在浮点空间里工作，最后才 pack 成整数。`maxq < 0` 分支是三值（ternary, `trits=True`）的特殊情形。

**(b) `Quantizer.find_params()`：求 scale/zero** —— [quant.py#L36-L117](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py#L36-L117)

```python
if self.perchannel:
    if weight:
        x = x.flatten(1)                     # (C_out, C_in) 每行一组
...
tmp = torch.zeros(x.shape[0], device=dev)
xmin = torch.minimum(x.min(1)[0], tmp)       # 强制 min<=0<=max，保证 0 可精确表示
xmax = torch.maximum(x.max(1)[0], tmp)
if self.sym:
    xmax = torch.maximum(torch.abs(xmin), xmax)
    tmp = xmin < 0
    if torch.any(tmp):
        xmin[tmp] = -xmax[tmp]
tmp = (xmin == 0) & (xmax == 0)              # 全零行：避免 scale=0
xmin[tmp] = -1
xmax[tmp] = +1
...
  self.scale = (xmax - xmin) / self.maxq
  if self.sym:
      self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
  else:
      self.zero = torch.round(-xmin / self.scale)
```

- **对称量化在这里也用无符号存储 + 固定零点** `(maxq+1)/2`（4bit → 8）。这正是 vLLM 里 `scalar_types.uint4b8`（"uint4，bias 8"）的来历（见 3.6）。
- `mse=True` 分支（L78-L95）在 `[p·xmin, p·xmax]`（`p` 从 1 递减到 `1−maxshrink`）上网格搜索使 `Σ|q−x|^2.4` 最小的范围——和 AWQ 的 clip 搜索思想一样，只是不看激活。
- 末尾 `weight=True` 时把 scale reshape 成 `(C_out, 1)`，便于对一整列 `w.unsqueeze(1)` 广播。

**(c) 打包：`Quant3Linear.pack`** —— [quant.py#L149-L184](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py#L149-L184)

```python
self.zeros = zeros * scales
...
intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
intweight = intweight.t().contiguous()        # (C_in, C_out)：沿输入维打包
...
qweight = np.zeros((intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32)
```

- `w + z·s` 再除以 s 得到整数 q（因为 `ŵ = s(q − z)`）。
- 3bit：每 32 个输入元素占 3 个 uint32（96 bit），所以 qweight 形状 `(C_in/32*3, C_out)`。有的值跨两个 uint32（L167-L169、L174-L176 的 `<< 30`、`>> 2` 就是在处理跨字边界）。
- 前向 [L186-L200](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py#L186-L200) 调 CUDA 扩展 `vecquant3matmul`，**只支持单 token（batch=1 decode）**——原始论文 kernel 只为证明 decode 加速。

### 3.2 GPTQ：校准与 Hessian 累积 `gptq.py`

[gptq.py#L32-L58](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L32-L58)

```python
if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))
    inp = inp.t()                                        # (C_in, N)
...
self.H *= self.nsamples / (self.nsamples + tmp)
self.nsamples += tmp
inp = math.sqrt(2 / self.nsamples) * inp.float()
self.H += inp.matmul(inp.t())                            # H = (2/n) Σ X Xᵀ
```

与 [wanda/lib/sparsegpt.py#L27-L38](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L27-L38) **完全相同**（注释掉的 L57 就是原始写法）。`sqrt(2/n)` 乘在 `inp` 上是为了让 `inp @ inp.T` 直接得到 `2/n` 倍，而前一行的 `*=` 是把旧的平均重新加权。

### 3.3 GPTQ 核心：`GPTQ.fasterquant`

**(a) 初始 scale、死通道、静态分组** —— [gptq.py#L60-L87](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L60-L87)

```python
if not self.quantizer.ready():
    self.quantizer.find_params(W, weight=True)          # 无分组时：整行一个 scale（per-channel）
H = self.H
del self.H
dead = torch.diag(H) == 0
H[dead, dead] = 1
W[:, dead] = 0
if static_groups:
    import copy
    groups = []
    for i in range(0, self.columns, groupsize):
        quantizer = copy.deepcopy(self.quantizer)
        quantizer.find_params(W[:, i:(i + groupsize)], weight=True)   # 按原始列顺序预先定组参数
        groups.append(quantizer)
```

`static_groups`：在任何补偿/置换之前，**用原始权重按原始连续列**确定每组的 scale/zero。这样即使开了 act-order，每组仍是连续的列——推理端不需要 `g_idx`。

**(b) act-order** —— [gptq.py#L89-L93](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L89-L93)

```python
if actorder:
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    invperm = torch.argsort(perm)
```

按 `H_jj = (2/n)Σx_j²` 降序重排列（W 的列和 H 的行列同步置换），最后在 [L155-L156](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L155-L156) 用 `invperm` 还原。

**(c) 阻尼 + Cholesky 逆** —— [gptq.py#L98-L104](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L98-L104)：与 SparseGPT 逐字相同（`percdamp=.01`，`cholesky → cholesky_inverse → cholesky(upper=True)`），原理见 guide 10 §1.4。

**(d) 逐列量化 + 补偿** —— [gptq.py#L106-L143](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L106-L143)

```python
for i1 in range(0, self.columns, blocksize):
    i2 = min(i1 + blocksize, self.columns)
    count = i2 - i1
    W1 = W[:, i1:i2].clone()                       # (C_out, 128)
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    Losses1 = torch.zeros_like(W1)
    Hinv1 = Hinv[i1:i2, i1:i2]
    for i in range(count):
        w = W1[:, i]                               # 一整列 (C_out,)
        d = Hinv1[i, i]
        if groupsize != -1:
            if not static_groups:
                if (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
            else:
                idx = i1 + i
                if actorder:
                    idx = perm[idx]
                self.quantizer = groups[idx // groupsize]
        q = quantize(w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()
        Q1[:, i] = q
        Losses1[:, i] = (w - q) ** 2 / d ** 2
        err1 = (w - q) / d
        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        Err1[:, i] = err1
    Q[:, i1:i2] = Q1
    Losses[:, i1:i2] = Losses1 / 2
    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
```

逐行（对比 [sparsegpt.py#L90-L111](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L90-L111)）：
- SparseGPT 是 `q = w; q[mask]=0`；GPTQ 是 `q = quantize(w, …)`。**其余所有行都一样**：`d = U_ii`、`err = (w−q)/d`、块内外积补偿、块间 GEMM 懒更新。
- **动态分组**（非 static）：每遇到一个组的第一列，就用**当前**（已被前面列补偿过的）权重 `W[:, j:j+g]` 计算该组 scale/zero——组参数会"跟着补偿走"。
  - 细节坑：它读的是全矩阵 `W`，而块内补偿写在副本 `W1` 上。只有当组边界与块边界对齐（默认 `blocksize=128`、`groupsize=128`）时，`W[:, i1:i1+g]` 恰好等于块开始时的 `W1`；若 groupsize 不整除 blocksize，组参数会少看到块内的补偿。
  - 若开 act-order 且非 static：组按**置换后顺序**划分，同一组的列在原矩阵里是分散的 → 推理需要 `g_idx[j] = 列 j 属于第几组`。
- **static 分组**：`groups[perm[idx] // groupsize]` 用原始列号查回预先算好的组参数。
- `Losses1 = (w−q)²/d²` 是 OBS 误差估计，只用于打印。
- 最后 [L160](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L160) 把（反量化后的）`Q` 写回 `layer.weight`——模型仍是 fp16，但每个值都落在格点上，可以无损 pack。

### 3.4 GPTQ 驱动：`llama.py`

**(a) 逐层、子模块顺序** —— [llama.py#L72-L125](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L72-L125)

```python
if args.true_sequential:
    sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj']
    ]
else:
    sequential = [list(full.keys())]
for names in sequential:
    ...
        gptq[name] = GPTQ(subset[name])
        gptq[name].quantizer = Quantizer()
        gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
    ...
    for j in range(args.nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
    ...
        gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups)
        quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
...
for j in range(args.nsamples):
    outs[j] = layer(inps[j].unsqueeze(0), ...)[0]
inps, outs = outs, inps
```

- `--true-sequential`：**块内也顺序化**。先量化 q/k/v，再重新前向得到 `o_proj` 的输入（此时 q/k/v 已是量化后的），再量化 o_proj……这比 wanda/SparseGPT 的"块内同时"更精确（回顾 guide 10 §3.6 的小细节）。
- 块结束再前向一次，用**量化后的块**输出作为下一块输入（误差感知的逐层量化）。

**(b) RTN 基线** —— [llama.py#L182-L193](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L182-L193)：`--nearest` 时直接 `find_params` + `quantize`，没有任何补偿；对比 GPTQ 只差 `fasterquant` 里的那两行误差传播。

### 3.5 AWQ：从入口到逐层搜索

**(a) `run_awq` 主循环** —— [awq/quantize/pre_quant.py#L102-L249](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/pre_quant.py#L102-L249)

```python
samples = get_calib_dataset(data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
...
class Catcher(nn.Module):
    def forward(self, inp, **kwargs):
        inps.append(inp)
        layer_kwargs.update(kwargs)
        raise ValueError  # early exit to break later inference
...
for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
    ...
    def cache_input_hook(m, x, y, name, feat_dict):
        x = x[0]
        x = x.detach().cpu()
        feat_dict[name].append(x)
    ...
            named_linears[name].register_forward_hook(functools.partial(cache_input_hook, name=name, feat_dict=input_feat))
    inps = layer(inps, **layer_kwargs)[0]          # 用"未量化"的层算下一层输入
    ...
    input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
    if auto_scale:
        scales_list = auto_scale_block(layer, layer_kwargs, w_bit=w_bit, q_config=q_config, input_feat=input_feat)
        apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
        awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")
    if mse_range:
        clip_list = auto_clip_block(layer, w_bit=w_bit, q_config=q_config, input_feat=input_feat)
        apply_clip(layer, clip_list)
        awq_results["clip"] += ...
return awq_results
```

- Catcher 技巧与 wanda/GPTQ 相同；这里一次性把所有校准样本（默认 128 段 × 512 token，来自 pileval）拼成一个 batch 送入。
- 用 `functools.partial` 绑定 `name`，等价于 wanda 的闭包工厂。
- **与 GPTQ 的重要区别**：`inps = layer(inps)` 在**搜索 scale 之前**、用原始层算出——AWQ 的下一层输入**不**包含本层量化误差（但包含了等价缩放，等价缩放本身不改变输出）。
- AWQ 的产出是 `awq_results = {"scale": [...], "clip": [...]}`，**不改权重的量化**——量化（伪量化或真量化）是之后单独一步（`pseudo_quantize_model_weight` / `real_quantize_model_weight`）。`apply_awq`（[L252-L254](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/pre_quant.py#L252-L254)）可以把缓存的结果套到新加载的 fp16 模型上。

**(b) 每个 Llama 块搜索哪几组 scale** —— [auto_scale.py#L215-L256](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L215-L256)

```python
elif isinstance(module, (LlamaDecoderLayer, Qwen2DecoderLayer)):
    # attention input
    scales_list.append(_auto_get_scale(
        prev_op=module.input_layernorm,
        layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
        inp=input_feat["self_attn.q_proj"],
        module2inspect=module.self_attn,
        kwargs=module_kwargs,
    ))
    # attn out
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn.v_proj, layers=[module.self_attn.o_proj], inp=input_feat["self_attn.o_proj"]))
    # fc1
    scales_list.append(_auto_get_scale(
        prev_op=module.post_attention_layernorm,
        layers=[module.mlp.gate_proj, module.mlp.up_proj],
        inp=input_feat["mlp.gate_proj"],
        module2inspect=module.mlp,
    ))
    # fc2
    scales_list.append(_auto_get_scale(
        prev_op=module.mlp.up_proj, layers=[module.mlp.down_proj], inp=input_feat["mlp.down_proj"]))
```

每一组是 **(prev_op, 共享同一输入的若干 Linear, 观察输出的模块)**：

| prev_op（吸收 1/s） | 被缩放的 Linear（乘 s） | 计算误差的模块 | 为什么可以折叠 |
|---|---|---|---|
| `input_layernorm` | q/k/v_proj | 整个 `self_attn` | RMSNorm 输出 = γ⊙norm(x)，γ/s 即可 |
| `v_proj` | o_proj | o_proj 自身 | attention 对 V 是线性加权和，V 的第 j 个输出通道 /s_j ⇒ o_proj 输入第 j 通道 /s_j |
| `post_attention_layernorm` | gate/up_proj | 整个 `mlp` | 同 RMSNorm |
| `up_proj` | down_proj | down_proj 自身 | `silu(gate)·up`：up 的输出通道 /s ⇒ 乘积通道 /s（gate 那路不能缩放，因为 silu 非线性） |

- `v_proj → o_proj` 仅当两者权重形状相同（即**非 GQA**）时才做：GQA 下 v_proj 输出维 `n_kv·hd` 小于 o_proj 输入维 `n_heads·hd`，一个 V 通道被多个头复用，无法一一对应（注释指向 PR #67）。
- `module2inspect`：q/k/v 三者共享 scale，误差要在整个 attention 的输出上衡量（包含 softmax 的非线性影响），所以传 `module.self_attn` 与 `module_kwargs`（attention mask、position 等）。

**(c) 网格搜索 `_search_module_scale`** —— [auto_scale.py#L109-L156](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L109-L156)

```python
x = x.to(next(block.parameters()).device)
with torch.no_grad():
    org_out = block(x, **kwargs)                       # fp16 参考输出
...
x_max = get_act_scale(x)                               # (C_in,) = mean |x|
...
org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
for ratio in range(n_grid):
    ratio = ratio * 1 / n_grid                         # α ∈ {0, 0.05, …, 0.95}
    scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
    scales = scales / (scales.max() * scales.min()).sqrt()
    for fc in linears2scale:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))                 # W·diag(s)
        fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1)) # Q(W·diag(s))·diag(s)^-1
    out = block(x, **kwargs)
    ...
    loss = ((org_out - out).float().pow(2).mean().item())
    ...
    block.load_state_dict(org_sd)                      # 每次试完恢复原权重
```

- `get_act_scale`（[L29-L30](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L29-L30)）：`x.abs().view(-1, C_in).mean(0)`——每个输入通道在所有 token 上的平均绝对值。
- 搜索时**不真的去缩放输入**，而是把 `1/s` 乘回量化后的权重：`(Q(Ws)/s)·x = Q(Ws)·(x/s)`，数学等价，省得改上一层。
- 归一化 `s / sqrt(max·min)`：让 s 的几何中心在 1 附近，避免整体放大缩小（整体缩放对量化没有意义，只会改变数值范围）。
- `ratio=0` ⇒ `s≡1` ⇒ 就是 RTN，所以搜索至少能找回 RTN 的误差。
- 文件里还有一个 `get_weight_scale`（[L18-L25](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L18-L25)），在当前代码中**未被使用**（论文中曾尝试 `s = s_x^α · s_w^{−β}`，最终版只用激活）。

**(d) 折叠 scale：`apply_scale`** —— [auto_scale.py#L449-L480](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L449-L480)

```python
if isinstance(prev_op, nn.Linear):
    assert len(layers) == 1
    scale_fc_fc(prev_op, layers[0], scales)
elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm)):
    scale_ln_fcs(prev_op, layers, scales)
elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation, nn.SiLU)):
    new_module = ScaledActivation(prev_op, scales)
    set_op_by_name(module, prev_op_name, new_module)
    scale_gelu_fc(prev_op, layers[0], scales)
...
if input_feat_dict is not None:
    for layer_name in layer_names:
        inp = input_feat_dict[layer_name]
        inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))
```

- `scale_ln_fcs`（[L34-L51](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L34-L51)）：`ln.weight.div_(scales)`（有 bias 也除）；`fc.weight.mul_(scales.view(1, -1))`——按**列**（输入通道）乘。
- `scale_fc_fc`（[L55-L72](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py#L55-L72)）：`fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))` 按**行**（上一层输出通道）除；`fc2.weight.mul_(scales.view(1, -1))` 按列乘。`[-scales.size(0):]` 这个切片是为了兼容 fused qkv 之类"上一层输出更宽、只有尾部对应"的情形。
- `ScaledActivation`（[qmodule.py#L68-L75](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py#L68-L75)）：`act(x) / scales`——无法折叠时（如 Bloom/MPT 的 GELU 后接 fc2）显式插入一个除法。
- 最后把缓存的 `input_feat` 也除以 s，使后续 clip 搜索看到的是缩放后的输入（与已乘 s 的权重配套）。
- 现在 fp16 模型在数学上与原模型等价（忽略 fp16 舍入），但权重分布已被"调整"得更易量化。

**(e) 截断搜索 `auto_clip_layer`** —— [auto_clip.py#L11-L63](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py#L11-L63)

```python
# w           [co, ci]      -> [co, 1, n_group, group size]
# input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]       # 下采样到 ~512 token
w = w.reshape(w.shape[0], 1, -1, group_size)
...
    org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
    ...
    org_out = (input_feat * w).sum(dim=-1)            # co, n_token, n_group：每组的部分和
    for i_s in range(int(max_shrink * n_grid)):
        max_val = org_max_val * (1 - i_s / n_grid)
        min_val = -max_val
        cur_w = torch.clamp(w, min_val, max_val)
        q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
        cur_out = (input_feat * q_w).sum(dim=-1)
        err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
        cur_best_idx = err < min_errs
        ...
```

- 巧妙的形状：`(co,1,G,g) * (1,n,G,g)` 广播后对最后一维求和，得到**每个输出通道、每个 token、每个组**的部分点积 `(co,n,G)`。因为组之间的量化互不影响，误差可以按组独立最小化，于是一次性对所有组并行搜索最优截断 `max_val`。
- 截断比例从 1.0 到 0.5（`max_shrink=0.5`，20 格中的前 10 格）。
- `auto_clip_block`（[L66-L83](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py#L66-L83)）**跳过 q/k**：注释说 q·kᵀ 是二次的，单看线性输出误差衡量不准。

**(f) 伪量化与真量化** —— [quantizer.py#L61-L103](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py#L61-L103)

```python
if q_group_size > 0:
    assert org_w_shape[-1] % q_group_size == 0
    w = w.reshape(-1, q_group_size)                     # (C_out*C_in/g, g)：每行一个组
assert w.dim() == 2
if zero_point:
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
else:  # we actually never used this
    assert min_val is None
    ...
w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
```

- `reshape(-1, g)` 是 group 量化最简洁的写法：因为 `C_in % g == 0` 且行优先存储，每 g 个连续元素恰是同一输出行内连续的一组。
- 与 GPTQ 的 `find_params` 不同，AWQ **不强制** `min ≤ 0 ≤ max`，并把 zero clamp 到 `[0, maxq]`。
- `zero_point=False` 分支第一行就会因 `min_val` 未定义而 `NameError`——注释也说明"从未使用"，AWQ 实际只支持非对称。
- `real_quantize_model_weight`（[L126-L165](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py#L126-L165)）用 `get_scale_zp=True` 拿到 scales/zeros 后构造 `WQLinear`。

**(g) 推理模块 `WQLinear`** —— [qmodule.py#L78-L224](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py#L78-L224)

```python
if w_bit not in [4]:
    raise NotImplementedError("Only 4-bit are supported for now.")
...
scale_zeros = zeros * scales
...
intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / qscales[:, idx // group_size]).to(torch.int)[:, None])
...
awq_linear.qweight = pack_intweight(intweight.contiguous(), interleave=4, kstride=64)
...
scaled_zeros[:, : scales.shape[1]] = -(qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))).to(dtype)
```

- 存 `scaled_zeros = −s·z` 而不是 z：反量化变成 `ŵ = s·q + (−s·z)`，一次 FMA，kernel 更快。
- `forward`（L202-L224）：token 数 < 8 走 `gemv_forward_cuda_new`（decode），否则走 `gemm_forward_cuda_new`（prefill）。

### 3.6 推理引擎如何消费量化权重：vLLM（简要）

目录：[`vllm/model_executor/layers/quantization/`](https://github.com/vllm-project/vllm/tree/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization)。核心是两层抽象：

**(a) `QuantizationConfig` ↔ `QuantizeMethodBase`** —— [base_config.py#L23-L93](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/base_config.py#L23-L93)

```python
class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        """Create weights for a layer. ..."""
    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        ...
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """... for example, to transpose weights for computation."""
```

三段式生命周期：**create_weights**（按 checkpoint 格式注册 `qweight/scales/qzeros` 等空参数）→ 加载 safetensors 填充 → **process_weights_after_loading**（重排成 kernel 需要的布局，如 Marlin 的 tile 格式）→ 每次前向 **apply**。`QuantizationConfig.get_quant_method(layer, prefix)`（[L200-L214](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/base_config.py#L200-L214)）为每一层挑方法。

**(b) Linear 层如何接入** —— [vllm/model_executor/layers/linear.py#L287-L293](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/linear.py#L287-L293) 与 [L417](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/linear.py#L417)

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := resolve_quant_method(quant_config, self, prefix=prefix):
    self.quant_method = quant_method
...
output = self.quant_method.apply(self, x, bias)
```

模型代码完全不感知量化：同一个 `LlamaForCausalLM` 定义，换 `quant_config` 就换了 Linear 的权重格式和 kernel。方法名到 Config 的映射在 [__init__.py#L156-L169](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/__init__.py#L156-L169)（如 `"gptq"`/`"gptq_marlin"` → `AutoGPTQConfig`，`"awq"`/`"awq_marlin"` → `AutoAWQConfig`，`"fp8"` → `Fp8Config`）。

**(c) GPTQ checkpoint 格式** —— [auto_gptq.py#L95-L157](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/auto_gptq.py#L95-L157) 与 [L322-L436](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/auto_gptq.py#L322-L436)

```python
TYPE_MAP = {
    (4, True): scalar_types.uint4b8,
    (8, True): scalar_types.uint8b128,
}
...
self.pack_factor = 32 // weight_bits  # packed into int32
...
qweight = PackedvLLMParameter(
    data=torch.empty(input_size_per_partition // self.quant_config.pack_factor, output_size_per_partition, dtype=torch.int32),
    input_dim=0, output_dim=1, packed_dim=0, ...)
qzeros ... torch.empty(scales_and_zp_size, output_size_per_partition // self.quant_config.pack_factor, dtype=torch.int32)
scales ... torch.empty(scales_and_zp_size, output_size_per_partition, dtype=params_dtype)
```

- `qweight: (C_in/8, C_out) int32`——**沿输入维打包**，与 GPTQ 原仓库 `Quant3Linear` 的 `intweight.t()` 布局一致（也与 2.1 的 `pack_int4` 一致）。
- `scales: (C_in/g, C_out)`：注意相对 PyTorch 的 `(C_out, C_in)` 是**转置**的。
- `uint4b8`：对称 4bit 用无符号存储、偏置 8——就是 `quant.py` 里 `zero = (maxq+1)/2`。这里只支持 `sym=True`。
- 实际计算交给 `choose_mp_linear_kernel` 选出的 mixed-precision kernel（Marlin、Machete 等），`process_weights_after_loading` 时重排。
- **act-order**：[utils/gptq_utils.py#L23-L50](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/utils/gptq_utils.py#L23-L50)

```python
if desc_act and group_size == -1:
    # With one group per output channel, activation ordering is a no-op.
    desc_act = False
if desc_act:
    raise ValueError(
        "GPTQ group activation ordering (desc_act=True) is no longer "
        "supported. Use a checkpoint with static activation ordering "
        "or desc_act=False.")
```

  在该 commit 中，vLLM **拒绝** `desc_act=True` 且分组的 checkpoint（需要 `g_idx` gather 的那种），只接受 `static_groups` 式的"静态 act-order"——正好对应 3.3(a) 讲的权衡。

**(d) AWQ checkpoint 格式** —— [auto_awq.py#L72-L127](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/auto_awq.py#L72-L127) 与 [L835-L876](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/auto_awq.py#L835-L876)

```python
# AWQ uses a non-standard packing order within int32 values.
# For 4-bit: standard order stores values at bit positions [0,4,8,12,16,20,24,28]
# for indices [0,1,2,3,4,5,6,7], while AWQ stores them for indices
# [0,4,1,5,2,6,3,7]. This permutation reverses that ordering.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
...
# --- Convert qweight: (K, N // pack) packed_dim=1 → (K // pack, N) packed_dim=0
unpacked = (qw.unsqueeze(-1) >> shifts) & mask  # (K, N_packed, pack_factor)
unpacked = unpacked[:, :, reverse_order]
```

- AutoAWQ 格式 `qweight: (C_in, C_out/8)`——**沿输出维打包**，且 nibble 交错顺序为 `[0,4,1,5,2,6,3,7]`；`qzeros: (C_in/g, C_out/8)`（非对称，零点也打包）；`scales: (C_in/g, C_out)`。
- `_convert_awq_to_standard_format` 把它解包、恢复顺序、再沿输入维重新打包成 GPTQ 式布局，于是两种格式可以共用同一套 mixed-precision kernel。
- （AutoAWQ 是社区对 llm-awq 的打包实现；llm-awq 自带的 `WQLinear` 用 int16 + interleave 布局，与此不同。）

**(e) fp8 与 KV cache**：`Fp8LinearMethod.apply`（[fp8.py#L418-L461](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/fp8.py#L418-L461)）在常规路径直接调 `self.fp8_linear.apply_weights`（fp8 GEMM，激活也量化成 fp8 = W8A8）；batch-invariant 模式下有 "反量化到 bf16 再 `F.linear`" 的回退，可以清楚看到 per-tensor（`weight_scale.numel()==1`）与 per-row scale 的广播方式。`BaseKVCacheMethod`（[kv_cache.py#L43-L56](https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/model_executor/layers/quantization/kv_cache.py#L43-L56)）给 Attention 层挂 `k_scale/v_scale`：写入 cache 前量化、读出时反量化；checkpoint 里没有就默认 1.0。

---

## 4. 方法对比表

| 方法 | 需要校准数据 | 用到的统计量 | 是否改变"未量化"权重 | 典型配置 | 速度 | 4bit 精度 |
|---|---|---|---|---|---|---|
| RTN | 否 | 无 | 否 | W8 per-channel；W4 g128 | 秒级 | 一般（≥7B 尚可，小模型掉点明显）|
| GPTQ | 是（~128×2048 token） | `H = XXᵀ` | **是**（误差补偿到后续列）| W4 g128 (+act-order/static) | 分钟~小时 | 好 |
| AWQ | 是（~128×512 token） | `mean|x|`，块输出误差 | 等价缩放 + clip（无补偿）| W4 g128 非对称 | 分钟级 | 好，泛化更稳（不易过拟合校准集）|
| SmoothQuant（参考） | 是 | `max|x|`, `max|w|` | 等价缩放（迁移激活离群到权重）| W8A8 | 快 | — |
| fp8 (E4M3) | 可选（静态 scale） | `amax` | 否 | per-tensor / per-channel / per-block | 快 | 近无损（8bit）|
| SparseGPT（对照） | 是 | `H` | 是 | 50% / 2:4 | 同 GPTQ | — |

GPTQ vs SparseGPT：同一个 `H`、同一个 Cholesky、同一个块循环；区别只在 `q = quantize(w)` vs `q = w·(1−mask)`。二者甚至可以联合（先剪再量，SparseGPT 论文有 50% + 4bit 实验）。

---

## 5. 常见困惑 / 坑

1. **"GPTQ 量化完模型还是 fp16？"** `fasterquant` 写回的是反量化后的 fp16 权重（都在格点上）。真正省显存要 pack 成 int32 并换 kernel（GPTQ 原仓 `llama_pack3`，或 AutoGPTQ/GPTQModel 生态 + vLLM Marlin）。
2. **对称量化的零点**：GPTQ sym 用无符号存储 + 零点 8（`uint4b8`），而有些实现用有符号 int4、零点 0。两者等价但**格式不兼容**，读 checkpoint 时必须知道是哪种。
3. **scale 张量的形状/转置**：PyTorch `W` 是 `(C_out, C_in)`，但 GPTQ/AWQ checkpoint 的 `scales` 通常是 `(C_in/g, C_out)`。自己写反量化时最容易错在这里。
4. **AWQ 打包顺序 `[0,4,1,5,2,6,3,7]`**：直接按顺序解包会得到乱码权重（模型能跑但输出垃圾）。
5. **act-order + group 的 `g_idx`**：开 `--act-order` 不开 `--static-groups` 时，每列的组号不连续，推理端需要 `g_idx`；vLLM 当前版本直接拒绝这类 checkpoint。
6. **GPTQ 动态分组与 blocksize**：组参数由 `W[:, j:j+g]` 计算，它看不到当前块内的补偿；`groupsize` 应整除 `blocksize`（默认都是 128）。
7. **`percdamp` 与 Cholesky 失败**：报 "not positive-definite" 时把 `--percdamp` 从 0.01 调到 0.1；校准样本太少（N < C_in）时 H 秩亏。
8. **校准集分布**：GPTQ 用的是"误差补偿"，校准集太窄会过拟合（在别的领域退化）；AWQ 只搜一个标量 α 和 clip，相对更稳健。
9. **AWQ 下一层输入不含量化误差**：`run_awq` 用原始层计算 `inps`，而 GPTQ 用量化后的层——这是两者"逐层"语义的差别。
10. **GQA 模型的 AWQ**：`v_proj→o_proj` 那组 scale 会被跳过（形状不等），o_proj 只能直接量化。
11. **llm-awq 的对称量化分支是坏的**（`assert min_val is None` 会 NameError），`--no_zero_point` 不要用。
12. **KV cache fp8 的 scale**：没有校准的 `k_scale/v_scale` 默认 1.0，若 K/V 数值超过 E4M3 的 ±448 就会饱和；长上下文、带离群值的模型应提供校准过的 scale。
13. **W4A16 并不加速 prefill**：prefill 是计算瓶颈，反量化反而增加开销；它主要加速 decode（带宽瓶颈）。想加速 prefill 需要 W8A8/fp8。

---

## 6. 自测题

1. 4bit 对称量化、`xmax = 1.5`：按 GPTQ `find_params`，scale 和 zero 是多少？`w = −1.5` 会被量化成哪个整数？（答：`scale = 3/15 = 0.2`，`zero = 8`，`q = clamp(round(−7.5)+8) = 0 或 1`——注意 `torch.round` 是"四舍六入五成双"，`round(−7.5) = −8`，所以 q=0，ŵ=−1.6）
2. 写出 GPTQ 与 SparseGPT 在 `fasterquant`/`fasterprune` 内循环中唯一不同的那一行。
3. 为什么 GPTQ 可以让所有输出行按同一顺序量化，而 OBQ 要为每行单独选顺序？这对计算量意味着什么？
4. act-order 为什么按 `diag(H)` 降序？`diag(H)` 的物理含义是什么？
5. AWQ 中 `s` 为什么要除以 `sqrt(max(s)·min(s))`？如果不除会怎样？
6. 在 Llama 中，`up_proj → down_proj` 可以折叠 scale，`gate_proj → down_proj` 为什么不行？
7. 一个 `(4096, 11008)` 的 down_proj 用 W4 g128 GPTQ 量化，vLLM 中 `qweight`、`scales`、`qzeros` 的形状分别是什么？（提示：down_proj 的 `C_in=11008, C_out=4096`）
8. W4A16、W8A8-int8、W8A8-fp8 分别主要加速 prefill 还是 decode？为什么？

## 7. 动手练习

1. **RTN 粒度实验**：对一个真实 Llama 层权重，比较 per-tensor / per-channel / g128 / g32 的 4bit RTN 重建误差 `‖W−Ŵ‖` 与输出误差 `‖(W−Ŵ)X‖`。
2. **复现 GPTQ ≥ RTN**：用 2.2 的 `gptq_quant`，构造相关性强的 `X`（如 `X = A @ Z`，A 为随机低秩混合），观察 GPTQ 相对 RTN 的增益随相关性增大而增大；再打开 `actorder=True` 对比。
3. **GPTQ ↔ SparseGPT 互改**：把 guide 10 的 `sparsegpt_prune` 中 `q = where(mask, 0, w)` 改成 `q = quantize(w)`，验证得到的结果与 `gptq_quant`（无 group 时）一致。
4. **AWQ α 扫描曲线**：用 2.3 的 `awq_search_scale`，打印每个 α 的误差，画出 U 形曲线；在 X 中人为放大 1% 通道，观察最优 α 右移。
5. **格式转换**：写一个函数，把 2.1 的 `pack_int4` 输出（GPTQ 布局）转换成 AutoAWQ 布局（沿输出维打包 + `[0,4,1,5,2,6,3,7]` 交错），再用 vLLM `_convert_awq_to_standard_format` 的逻辑（numpy 重写）转回来，检查往返一致。
6. **折叠验证**：对一个 `RMSNorm → Linear` 小网络，随机 s，执行 `fold_into_rmsnorm`，验证 fp32 下输出与折叠前一致（误差 < 1e-5）。
