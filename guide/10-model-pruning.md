# 10 · 模型剪枝（Model Pruning）源码导读

> 对应源码：
> - `repos/09-model-pruning/wanda`（locuslab/wanda @ `8e8fc87b`）：Magnitude / Wanda / SparseGPT 三种非结构化与 N:M 剪枝都在这里
> - `repos/09-model-pruning/LLM-Pruner`（horseee/LLM-Pruner @ `128a07d9`）：面向 LLaMA 的结构化剪枝 + LoRA 恢复
> - `repos/09-model-pruning/Torch-Pruning`（VainF/Torch-Pruning @ `e80127d7`）：通用 DepGraph 依赖图
>
> 本文所有行号都基于上述 commit，链接是 permalink，点开即对应原文。

---

## 0. 要解决的问题 / 一句话直觉

**问题**：一个 7B 模型有 70 亿个参数，推理时既要存下它们（显存），又要把它们全读一遍（带宽）。很多参数其实"可有可无"，能不能删掉一部分，让模型更小更快，而精度几乎不掉？

**一句话直觉**：

- **Magnitude 剪枝**：绝对值小的权重贡献小，删掉。
- **Wanda**：权重小 *不一定* 不重要——如果它连接的那个输入通道经常很大（LLM 中有"离群激活"），乘积 `|W|·|x|` 才是真正贡献。所以按 `|W_ij| · ||X_j||` 打分。
- **SparseGPT**：删掉一个权重会造成误差，那就 *用剩下的权重去补偿*。补偿方式由二阶信息（Hessian `H = XXᵀ`）决定，这就是经典的 OBS（Optimal Brain Surgeon）。
- **LLM-Pruner / DepGraph**：上面三种只是把权重置 0（矩阵形状不变），需要稀疏硬件/内核才能加速。结构化剪枝直接 *删掉整行/整列/整个头*，矩阵真的变小了。但删 `q_proj` 的一个输出通道，就必须同步删 `o_proj` 的对应输入通道……这些"耦合"关系由**依赖图**自动找出。

### 三种稀疏粒度

| 粒度 | 删掉的东西 | 形状是否改变 | 加速需要 | 典型精度损失 |
|---|---|---|---|---|
| 非结构化 (unstructured) | 任意单个元素 | 否（只是变 0） | 专用稀疏 kernel，50% 稀疏在 GPU 上几乎不加速 | 最小 |
| 半结构化 N:M（如 2:4） | 每连续 M 个元素中删 N 个 | 否，但可压缩存储 | Ampere+ 的 Sparse Tensor Core，2:4 理论 2× 矩阵乘 | 中等 |
| 结构化 (structured) | 整行/整列/整个注意力头/整层 | **是** | 无需特殊硬件，普通 dense GEMM 直接变快 | 最大，通常需要再训练恢复 |

---

## 1. 核心原理（带张量形状）

### 1.1 符号约定

对一个 `nn.Linear(in_features=C_in, out_features=C_out)`：

- 权重 `W ∈ ℝ^{C_out × C_in}`（PyTorch 存储顺序：**行 = 输出通道，列 = 输入通道**）
- 校准输入把 batch 和序列展平：`X ∈ ℝ^{C_in × N}`，`N = 样本数 × seqlen`
- 输出 `Y = W X ∈ ℝ^{C_out × N}`

剪枝的目标（逐层重建）：找一个带掩码的 `Ŵ`，使

```
min_{Ŵ}  || W X − Ŵ X ||_F²     s.t.  Ŵ 满足稀疏约束
```

注意这个目标**按行可分**：第 i 行输出 `y_i = W_i X` 只依赖第 i 行权重 `W_i ∈ ℝ^{1×C_in}`。所以可以对每一行独立做决策——这正是 Wanda "按输出行比较"和 SparseGPT "所有行共享同一个 H" 的根源。

### 1.2 Magnitude

```
score_ij = |W_ij|
```

全局（整个矩阵）排序，删掉最小的 `s%`。

### 1.3 Wanda（Pruning by Weights AND activations）

```
score_ij = |W_ij| · ||X_j||_2          ||X_j||_2 = sqrt(Σ_n X_{j,n}²)，形状 (C_in,)
```

- `||X_j||` 是一个长度为 `C_in` 的向量，对**每一列**是常数。
- **比较组 = 每个输出行**：在每一行 `W_i` 内部排序，删掉该行分数最小的 `s · C_in` 个。论文实验证明 per-output 比 per-layer 全局排序好得多（对 LLM 尤其明显）。
- 为什么合理？若只剪一个权重 `W_ij`，输出 `y_i` 的变化是 `W_ij · X_{j,:}`，其范数恰好就是 `|W_ij| · ||X_j||`。这其实是 SparseGPT 分数在 "H 只取对角线" 时的近似（见 1.4）。
- 不需要任何权重更新、不需要求逆，只需一次前向收集 `||X_j||²`。

### 1.4 SparseGPT（OBS 的可扩展版）

对一行 `w ∈ ℝ^{C_in}`，误差 `E(w) = ||(w−ŵ) X||² = δwᵀ H δw`，其中

```
H = 2 X Xᵀ ∈ ℝ^{C_in × C_in}          (所有行共享同一个 H！)
```

**OBS 公式**（删除第 q 个权重并最优补偿其余权重）：

```
误差增量    ε_q = w_q² / [H⁻¹]_qq
最优补偿    δw  = − (w_q / [H⁻¹]_qq) · H⁻¹[:, q]          (形状 (C_in,))
```

对角线近似 `[H⁻¹]_qq ≈ 1/H_qq = 1/(2||X_q||²)` 时，`ε_q ∝ w_q² ||X_q||²`，开方就是 Wanda 分数。

**问题**：每删一个权重 H⁻¹ 就要更新（去掉第 q 行/列），对 `C_in=4096` 的每一行做这件事太贵。SparseGPT 的技巧：

1. **按列顺序处理**：从左到右逐列 i 决定剪不剪，并且只更新"还没处理的列" `i+1:`。这样所有行用的是同一串 `H⁻¹` 子矩阵序列 `H_{[i:],[i:]}⁻¹`，可以一次性预计算。
2. **Cholesky 技巧**：令 `H⁻¹ = Uᵀ U`（U 上三角），则 "依次消去前 i−1 个变量后的 H⁻¹ 的第 i 行" 恰好等于 `U[i, i:] · U[i,i]`。所以只要算一次 `U = chol(H⁻¹)ᵀ`，就有：
   - `d = U[i,i]`，于是 `[H_i⁻¹]_ii = d²`
   - 误差 `(w_i − q_i)² / d²`
   - 补偿 `W[:, i:] −= ((w_i − q_i)/d) · U[i, i:]`
3. **分块 (blocksize=128)**：块内逐列更新，块结束后把累积误差一次性用矩阵乘 `Err1 @ Hinv[i1:i2, i2:]` 推给后面所有列（懒更新，GEMM 友好）。GPTQ 用的是完全相同的骨架（见 guide 12）。
4. **mask 选择**：每进入一个块，用 `w²/d²` 在块内选出要剪的位置（自适应 mask）；2:4 模式下每 M 列选一次。

### 1.5 结构化剪枝：耦合与 Taylor 重要性

LLaMA MLP：`down( silu(gate(x)) * up(x) )`，形状：`gate,up: (I, d)`，`down: (d, I)`。
删掉中间维的第 k 个通道 ⇒ 必须同时删 `gate.weight[k,:]`、`up.weight[k,:]`、`down.weight[:,k]`。这三个切片组成一个**组（group）**。

Attention：删掉第 h 个头 ⇒ 删 `q/k/v_proj.weight[h*hd:(h+1)*hd, :]` 和 `o_proj.weight[:, h*hd:(h+1)*hd]`。

组重要性（LLM-Pruner 的一阶 Taylor）：删掉参数 `W_k` 的损失变化

```
ΔL ≈ | ∂L/∂W · W |   （逐元素，one-step Taylor）
I(group) = reduce_{组内所有切片}  Σ_{切片元素} |W ⊙ ∇W|
```

`reduce` 可以是 sum / mean / max / prod 等。可选加二阶项 `½ W ⊙ (∇W)² ⊙ W`（用梯度平方近似 Fisher/Hessian 对角线）。

**恢复**：结构化剪枝后困惑度上升明显，LLM-Pruner 用 LoRA 在少量数据（如 Alpaca 50k）上微调几小时恢复。

---

## 2. 最小实现（自己写的 PyTorch 片段）

### 2.1 Wanda + 2:4，作用在一个 Linear 上（含 hook 校准）

```python
import torch, torch.nn as nn

@torch.no_grad()
def wanda_prune_linear(layer: nn.Linear, calib_inputs, sparsity=0.5, n=0, m=0):
    """calib_inputs: list of tensors, each (B, T, C_in). n:m 例如 2:4 => n=2, m=4（每4个删2个）。"""
    C_out, C_in = layer.weight.shape
    sq_norm = torch.zeros(C_in, device=layer.weight.device)    # Σ_n x_{j,n}^2
    ntok = 0

    def hook(mod, inp, out):
        nonlocal ntok
        x = inp[0].reshape(-1, C_in).float()                   # (N, C_in)
        sq_norm.add_((x ** 2).sum(0))                          # 累加每个输入通道的平方和
        ntok += x.shape[0]
    h = layer.register_forward_hook(hook)
    for x in calib_inputs:
        layer(x)
    h.remove()

    W = layer.weight.data
    score = W.abs() * sq_norm.sqrt().unsqueeze(0)              # (C_out, C_in) ⊙ (1, C_in)
    mask = torch.zeros_like(W, dtype=torch.bool)               # True = 要剪掉
    if n:                                                      # N:M —— 每行、每 m 个连续列为一组
        s = score.view(C_out, C_in // m, m)
        idx = s.topk(n, dim=-1, largest=False).indices          # 每组最小的 n 个
        mask.view(C_out, C_in // m, m).scatter_(-1, idx, True)
    else:                                                      # 非结构化 —— 每个输出行内部排序
        k = int(C_in * sparsity)
        idx = score.argsort(dim=1)[:, :k]
        mask.scatter_(1, idx, True)
    W[mask] = 0
    return mask

# 用法
lin = nn.Linear(64, 32)
data = [torch.randn(2, 16, 64) for _ in range(4)]
m24 = wanda_prune_linear(lin, data, n=2, m=4)
assert (lin.weight.view(32, 16, 4) != 0).sum(-1).max() <= 2   # 每 4 个最多留 2 个
```

要点：均值还是总和不影响**同一行内**的排序（对所有列乘了同一个常数），所以 wanda 源码里那个 running mean 只是数值上更稳。

### 2.2 迷你 SparseGPT（单层，非结构化）

```python
@torch.no_grad()
def sparsegpt_prune(W, X, sparsity=0.5, blocksize=32, percdamp=0.01):
    """W: (C_out, C_in) float32；X: (C_in, N) 校准输入。返回剪枝且已补偿的 W。"""
    W = W.clone()
    C_in = W.shape[1]
    H = 2 * X @ X.T / X.shape[1]                               # (C_in, C_in)
    H += percdamp * H.diag().mean() * torch.eye(C_in)          # 阻尼，保证正定
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    for i1 in range(0, C_in, blocksize):
        i2 = min(i1 + blocksize, C_in)
        W1, U1 = W[:, i1:i2].clone(), U[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        d = U1.diag()                                          # (bs,)
        score = W1 ** 2 / d ** 2                               # OBS 误差 w^2/[H^-1]_qq
        thr = score.flatten().kthvalue(max(1, int(score.numel() * sparsity))).values
        mask1 = score <= thr
        for i in range(i2 - i1):
            w = W1[:, i]
            q = torch.where(mask1[:, i], torch.zeros_like(w), w)
            err = (w - q) / U1[i, i]                           # (C_out,)
            W1[:, i:] -= err[:, None] * U1[i, i:][None, :]     # 块内补偿后续列
            W1[:, i] = q                                       # 本列定型
            Err1[:, i] = err
        W[:, i1:i2] = W1
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]                      # 懒更新：一次性推给后面的块
    return W

W = torch.randn(16, 64); X = torch.randn(64, 512)
Wp = sparsegpt_prune(W, X)
mag = W.clone(); mag[W.abs() < W.abs().median()] = 0
print(((W - Wp) @ X).norm(), ((W - mag) @ X).norm())         # SparseGPT 的重建误差应更小
```

注意 `W1[:, i] = q` 写在补偿之后：补偿那一步对第 i 列本身的作用是 `w - err*d = q`，所以两者一致；显式赋值只是避免浮点残差。

---

## 3. 源码导读（按阅读顺序）

### 3.1 入口：`wanda/main.py` —— 稀疏类型如何变成 `prune_n, prune_m`

[main.py#L49-L53](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/main.py#L49-L53)

```python
prune_n, prune_m = 0, 0
if args.sparsity_type != "unstructured":
    assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
    prune_n, prune_m = map(int, args.sparsity_type.split(":"))
```

- `"2:4"` → `prune_n=2, prune_m=4`。注意：这里的 N 是**被剪掉**的个数（每 4 个剪 2 个），恰好和 NVIDIA 的 "2:4 = 每 4 个保留 2 个" 数量相同，因为都是 50%。`"4:8"` 同理。
- 随后 [main.py#L68-L75](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/main.py#L68-L75) 按 `--prune_method` 分发到 `prune_wanda / prune_magnitude / prune_sparsegpt / prune_ablate`。

### 3.2 `find_layers`：找出一个 decoder block 里所有 Linear

[lib/prune.py#L11-L30](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L11-L30)

递归 `named_children()`，返回形如 `{"self_attn.q_proj": Linear, ..., "mlp.down_proj": Linear}` 的字典。所有方法都是 **逐 decoder 层、逐 Linear** 地处理。

### 3.3 Magnitude 基线

[lib/prune.py#L105-L125](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L105-L125)

```python
W_metric = torch.abs(W)
if prune_n != 0:
    W_mask = (torch.zeros_like(W)==1)
    for ii in range(W_metric.shape[1]):
        if ii % prune_m == 0:
            tmp = W_metric[:,ii:(ii+prune_m)].float()
            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
else:
    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
    W_mask = (W_metric<=thresh)
W[W_mask] = 0
```

- N:M 分支：沿**输入维（dim=1）**每 `prune_m` 列取一个窗口 `tmp: (C_out, m)`，`topk(..., largest=False)` 找每行最小的 n 个，`ii + 局部下标` 换算成全局列号，`scatter_` 写入掩码。2:4 稀疏之所以沿输入维分组，是因为 Sparse Tensor Core 在 GEMM 的 K 维（即归约维 = 输入维）上压缩。
- 非结构化分支：**整层全局**一个阈值（与 Wanda 的逐行比较形成对照）。

### 3.4 校准数据：`prepare_calibration_input` 与 `Catcher` 技巧

[lib/prune.py#L58-L95](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L58-L95)

```python
inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
...
class Catcher(nn.Module):
    def forward(self, inp, **kwargs):
        inps[cache['i']] = inp
        cache['i'] += 1
        cache['attention_mask'] = kwargs['attention_mask']
        cache['position_ids'] = kwargs['position_ids']
        raise ValueError
layers[0] = Catcher(layers[0])
for batch in dataloader:
    try:
        model(batch[0].to(device))
    except ValueError:
        pass
layers[0] = layers[0].module
```

逐行解释：
- 用一个"假"层替换第 0 个 decoder 层。模型跑完 embedding 后进入 `Catcher.forward`，**把第 0 层的输入（即 embedding 输出，形状 `(1, seqlen, hidden)`）存下来**，再故意抛异常提前终止——后面 32 层根本不用算。
- 顺手拿到 `attention_mask`、`position_ids`，后续逐层前向时复用。
- 得到 `inps: (nsamples=128, seqlen, hidden)`。这就是"逐层（layer-wise）剪枝"的起点：之后每次只把**一层**放在 GPU 上跑。
- 注意 `128` 是硬编码的，等于默认 `--nsamples 128`。

### 3.5 激活统计：`WrappedGPT.add_batch`

[lib/layerwrapper.py#L5-L35](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/layerwrapper.py#L5-L35)

```python
self.rows = layer.weight.data.shape[0]       # C_out
self.columns = layer.weight.data.shape[1]    # C_in
self.scaler_row = torch.zeros((self.columns), device=self.dev)
...
def add_batch(self, inp, out):
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    tmp = inp.shape[0]                                    # 本批样本数 (=1)
    if isinstance(self.layer, nn.Linear):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))        # (B*T, C_in)
        inp = inp.t()                                     # (C_in, B*T)
    self.scaler_row *= self.nsamples / (self.nsamples+tmp)
    self.nsamples += tmp
    inp = inp.type(torch.float32)
    self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
```

- `inp.t()` 之后每一**行**是一个输入通道 j 在所有 token 上的取值，`torch.norm(dim=1)**2` 得到 `Σ_tokens x_j²`，形状 `(C_in,)`。
- 前两行 `*=`、`+= /nsamples` 是增量求平均：`scaler_row = (1/N_samples) Σ_samples ||X_j^{(s)}||²`。名字叫 `scaler_row` 但其实是**按列（输入通道）**的统计量。
- 为什么用 float32：bf16/fp16 下平方累加会溢出或丢精度。

### 3.6 Wanda 主循环：`prune_wanda`

[lib/prune.py#L127-L210](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L127-L210)

**(a) 挂 hook 收集统计** —— [L146-L162](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L146-L162)

```python
for name in subset:
    wrapped_layers[name] = WrappedGPT(subset[name])
def add_batch(name):
    def tmp(_, inp, out):
        wrapped_layers[name].add_batch(inp[0].data, out.data)
    return tmp
handles = []
for name in wrapped_layers:
    handles.append(subset[name].register_forward_hook(add_batch(name)))
for j in range(args.nsamples):
    with torch.no_grad():
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
for h in handles:
    h.remove()
```

- `register_forward_hook(fn)` 的 `fn(module, inputs_tuple, output)`；`inp[0]` 取出 Linear 的真正输入。
- `add_batch(name)` 是**闭包工厂**：如果直接在循环里写 `lambda` 引用 `name`，所有 hook 都会捕获最后一个 `name`（Python 后期绑定坑），所以要包一层函数。
- 跑完 128 个样本后摘掉 hook。

**(b) 打分与选择** —— [L164-L202](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L164-L202)

```python
W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
if prune_n != 0:
    ...同 magnitude 的 N:M 写法...
else:
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    ...
        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
        W_mask.scatter_(1, indices, True)
subset[name].weight.data[W_mask] = 0
```

- 第一行就是 Wanda 的全部"数学"：`(C_out, C_in) * (1, C_in)` 广播。
- `torch.sort(dim=-1)`：**沿每一行**排序 → per-output 比较组。每一行恰好剪 `C_in * ratio` 个，所以每个输出神经元的稀疏度都一样。
- `use_variant` 分支（L178-L196）是论文附录的变体：不固定每行剪多少个，而是剪到"累计分数占该行总分 α"为止，再二分搜索 α 让总稀疏度命中目标。

**(c) 关键：用剪枝后的层重新前向，得到下一层的输入** —— [L204-L207](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L204-L207)

```python
for j in range(args.nsamples):
    with torch.no_grad():
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
inps, outs = outs, inps
```

为什么要再算一遍？因为下一层应该看到的是"**已剪枝的**上一层"产生的激活，这样误差不会被忽略地逐层累积（顺序/贪心逐层剪枝）。`inps, outs = outs, inps` 是双缓冲交换，省一份显存。

> 小细节：同一个 decoder block 内的 7 个 Linear 是用**剪枝前**同一次前向的统计量打分的（hook 在剪枝前就摘掉了）。也就是说块内不是顺序的，块间才是顺序的。

### 3.7 SparseGPT：`lib/sparsegpt.py`

**(a) 累积 Hessian** —— [lib/sparsegpt.py#L27-L38](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L27-L38)

```python
inp = inp.t()                                     # (C_in, N)
self.H *= self.nsamples / (self.nsamples + tmp)
self.nsamples += tmp
inp = math.sqrt(2 / self.nsamples) * inp.float()
self.H += inp.matmul(inp.t())                     # (C_in, C_in)
```

结果就是 `H = (2/N) Σ X Xᵀ`，形状 `(C_in, C_in)`（4096×4096 = 64MB fp32；MLP 的 down_proj 是 11008² ≈ 485MB）。和 Wanda 比：Wanda 只存这个矩阵的**对角线**。

**(b) 预处理 H：死通道、阻尼、Cholesky 逆** —— [L52-L66](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L52-L66)

```python
dead = torch.diag(H) == 0
H[dead, dead] = 1
W[:, dead] = 0
...
damp = percdamp * torch.mean(torch.diag(H))
diag = torch.arange(self.columns, device=self.dev)
H[diag, diag] += damp
H = torch.linalg.cholesky(H)
H = torch.cholesky_inverse(H)
H = torch.linalg.cholesky(H, upper=True)
Hinv = H
```

- `dead`：某输入通道在校准集里恒为 0，则 H 对应行列全 0、不可逆。把对角置 1 并把这一列权重直接置 0（反正它对输出没贡献）。
- `damp`：加 `1% × 平均对角值` 的岭项，防止数值上不正定（N 不够大或通道高度相关时常见）。
- 三连：`cholesky(H)` → `cholesky_inverse` 得到 `H⁻¹` → 再对 `H⁻¹` 做**上三角** Cholesky 得 `U`（`H⁻¹ = UᵀU`）。原理见 1.4：`U` 的第 i 行就是"顺序消元到第 i 列时"所需的逆 Hessian 行（差一个 `U[i,i]` 因子），一次分解替代了 `C_in` 次 rank-1 降阶更新。

**(c) 分块 + 自适应 mask** —— [L70-L88](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L70-L88)

```python
for i1 in range(0, self.columns, blocksize):
    i2 = min(i1 + blocksize, self.columns)
    W1 = W[:, i1:i2].clone()                  # (C_out, bs)
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    Hinv1 = Hinv[i1:i2, i1:i2]                # (bs, bs)
    if prune_n == 0:
        ...
            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh
```

- `tmp = w² / U_ii²` 就是 OBS 误差 `w_q² / [H⁻¹]_qq`。注意此时 `W1` 已经被前面块的误差补偿更新过，所以 mask 是"看着补偿后的权重"选的，这叫 **adaptive mask selection**。
- 阈值在 `(C_out × 128)` 的整个块里选（不是逐行），块内稀疏度恰为 `sparsity`。

**(d) 逐列 OBS 更新** —— [L90-L106](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L90-L106)

```python
for i in range(count):
    w = W1[:, i]                     # (C_out,)  这一列跨所有输出行
    d = Hinv1[i, i]                  # 标量 U_ii
    if prune_n != 0 and i % prune_m == 0:
        tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
        mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    q = w.clone()
    q[mask1[:, i]] = 0
    Q1[:, i] = q
    Losses1[:, i] = (w - q) ** 2 / d ** 2
    err1 = (w - q) / d
    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    Err1[:, i] = err1
```

- 一次处理**一整列**（所有 `C_out` 行同时），因为所有行共享 H——这是 SparseGPT 能 scale 到 175B 的关键。
- N:M：每到组首 `i % m == 0` 时，在接下来 m 列里按 OBS 分数选 n 个剪掉（逐行）。
- `err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)`：外积 `(C_out,1)×(1,bs−i)`，即 OBS 补偿 `δw = −(w_q/[H⁻¹]_qq) H⁻¹[q, :]` 的 Cholesky 形式。
- 被剪的元素 `q=0`，没剪的 `q=w`、`err=0`，所以只有被剪的位置会推动后面列变化。

**(e) 块间懒更新** —— [L108-L111](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/sparsegpt.py#L108-L111)

```python
W[:, i1:i2] = Q1
Losses += torch.sum(Losses1, 1) / 2
W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])   # (C_out,bs)@(bs, C_in-i2)
```

块内的 rank-1 更新只作用在 128 列上（访存小），对剩余所有列的影响累计到 `Err1`，一次大 GEMM 补上。GPTQ 称之为 "lazy batch updates"。

**(f) 驱动函数** —— [lib/prune.py#L213-L300](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L213-L300)：与 `prune_wanda` 结构完全相同（Catcher → hook `add_batch` → `fasterprune(...)` [L288](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L288) → 重新前向 → 交换缓冲）。对比阅读能清楚看到：**两种方法只在"统计什么"(`||X_j||²` vs `XXᵀ`) 和"如何选+是否补偿"上不同**。

### 3.8 LLM-Pruner：结构化剪枝流程 `hf_prune.py`

**(a) 选重要性度量** —— [hf_prune.py#L82-L91](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L82-L91)：`random / l1 / l2 / taylor`，taylor 对应 `llama_pruner.TaylorImportance(group_reduction=..., taylor=...)`。

**(b) 构造 MetaPruner（block-wise 模式）** —— [hf_prune.py#L95-L121](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L95-L121)

```python
kwargs = {
    "importance": imp,
    "global_pruning": args.global_pruning,
    "iterative_steps": args.iterative_steps,
    "ch_sparsity": args.pruning_ratio,
    "ignored_layers":[],
    "channel_groups": {},
    "consecutive_groups": {
        layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
    },
    "customized_pruners": {
        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
    },
    "root_module_types": None,
    "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                      [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
}
pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
```

- `forward_prompts`（[L77-L80](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L77-L80)）只是两句随便的 token，注释说得很清楚：**只用于追踪计算图来建依赖图**，输出值无所谓。
- `root_instances`：只以 `q_proj`（代表注意力块）和 `gate_proj`（代表 MLP 块）为"根"建组。从根出发沿依赖图扩展，`q_proj` 的输出通道组会牵连到同块的 k/v/o，`gate_proj` 会牵连 up/down。
- `consecutive_groups[q_proj] = head_dim`：把连续 `head_dim` 个通道当作一个单元一起打分、一起删 → **以注意力头为粒度**剪枝，不会把一个头剪掉一半。
- `block_*_layer_start/end`：论文发现首尾几层非常敏感，所以只剪中间层：默认 `[3, 31)`（[hf_prune.py#L291-L294](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L291-L294)）。默认 `--taylor param_first`、`--grouping_strategy sum`（[L297-L299](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L297-L299)）。
- **不剪 hidden size**（残差流维度），因为那会牵连所有层、embedding 和 lm_head。channel-wise 模式（L167-L217）才剪 hidden 维，那时 `root_module_types=[LlamaRMSNorm, LlamaAttention]`。

**(c) 迭代：反向得梯度 → `pruner.step()`** —— [hf_prune.py#L125-L157](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/hf_prune.py#L125-L157)

```python
example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
if args.taylor in ['param_mix', 'param_second']:
    for j in range(args.num_examples):
        ...
        loss.backward()
        for module_param in model.parameters():
            module_param.grad = module_param.grad * module_param.grad / args.num_examples
            if hasattr(module_param, 'acc_grad'):
                module_param.acc_grad += module_param.grad
            ...
loss = model(example_prompts, labels=example_prompts).loss
loss.backward()
pruner.step()
...
for layer in model.model.layers:
    layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
```

- 一次语言模型 loss 的反向就给了所有参数的 `∇W`，Taylor 重要性需要它。
- `param_second / param_mix`：逐样本反向，把 `grad²/n` 累加到 `acc_grad`，作为 Hessian 对角的经验 Fisher 近似。
- 剪完头之后，**模型代码里写死的 `num_heads` 必须手动改**，否则 `view(bsz, q_len, num_heads, head_dim)` 会形状不匹配——这是结构化剪枝最容易踩的坑。

**(d) Taylor 重要性的实现** —— [LLMPruner/pruner/hf_llama_pruner.py#L215-L343](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/pruner/hf_llama_pruner.py#L215-L343)

```python
for dep, idxs in group:                       # group = [(依赖, 被剪下标), ...]
    layer = dep.target.module
    prune_fn = dep.handler
    ...
    salience = layer.weight * layer.weight.grad          # (C_out, C_in) 一阶 Taylor
    if self.taylor in ['param_second']:
        salience = layer.weight * layer.weight.acc_grad * layer.weight
    elif self.taylor in ['param_mix']:
        salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight
    # Linear out_channels
    if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
        if self.taylor == 'vectorize':
            local_norm = salience.sum(1).abs()           # (C_out,)
        elif 'param' in self.taylor:
            local_norm = salience.abs().sum(1)
    # Linear in_channels
    elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
        ...  local_norm = salience.abs().sum(0)          # (C_in,)
        local_norm = local_norm[idxs]
```

逐行：
- 组中的每个成员是"某个层的某一维被剪"：被剪**输出**通道（`q/k/v/gate/up` 的行）就按行求和 `sum(1)`；被剪**输入**通道（`o_proj/down_proj` 的列）按列求和 `sum(0)`。这样每个成员都给出一个"每通道重要性"向量。
- `vectorize`（向量级）：先求和再取绝对值 = 把整行当作一个参数向量做 Taylor `|∇Wᵀ·W|`；`param_*`（元素级）：先取绝对值再求和。论文 Table 中两者各有胜负。
- [L332-L341](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/pruner/hf_llama_pruner.py#L332-L341)：成员向量长度可能不同（例如 GQA 下 k_proj 较短），按最短长度 reshape 求和对齐，`torch.stack` 后用 `_reduce`（[L221-L238](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/pruner/hf_llama_pruner.py#L221-L238)：sum/mean/max/prod/first/second）汇总为组重要性。

**(e) MetaPruner 如何用重要性挑出要剪的下标** —— [LLMPruner/torch_pruning/pruner/algorithms/metapruner.py#L234-L284](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/torch_pruning/pruner/algorithms/metapruner.py#L234-L284)

```python
for group in self.DG.get_all_groups(ignored_layers=..., root_module_types=..., root_instances=...):
    if self._check_sparsity(group):
        ...
        imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
        current_channels = self.DG.get_out_channels(module)
        target_sparsity = self.get_target_sparsity(module)
        n_pruned = current_channels - int(self.layer_init_out_ch[module] * (1 - target_sparsity))
        ...
        if consecutive_groups > 1:
            imp = imp.view(-1, consecutive_groups).sum(1)      # (num_heads,)
        imp_argsort = torch.argsort(imp)
        ...
        elif consecutive_groups > 1:
            pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
            group_size = consecutive_groups
            pruning_idxs = torch.cat(
                [torch.tensor([j+group_size*i for j in range(group_size)])
                for i in pruning_groups], 0)
        ...
        group = self.DG.get_pruning_group(module, pruning_fn, pruning_idxs.tolist())
        if self.DG.check_pruning_group(group):
            yield group
```

- `get_target_sparsity` 配合 `iterative_steps` 实现**渐进剪枝**（每步剪一点）。
- `consecutive_groups`：把 `(num_heads*head_dim,)` 的通道重要性 `view(-1, head_dim).sum(1)` 成每头一个分数，选最小的若干个头，再展开成该头所有通道下标。
- 最后把下标交给依赖图 `get_pruning_group` 得到**所有耦合层上的对应下标**，`group.prune()` 一次性物理删除（见 [metapruner.py#L174-L187](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/torch_pruning/pruner/algorithms/metapruner.py#L174-L187) 的 `step`）。

**(f) 自定义剪枝函数示例：整个 Attention 当作一个"层"** —— [hf_llama_pruner.py#L39-L63](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/LLMPruner/pruner/hf_llama_pruner.py#L39-L63)

```python
for sub_layer in [layer.o_proj]:
    keep_idxs = list(set(range(sub_layer.out_features)) - set(idxs))
    ...
    sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[keep_idxs])
for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
    ...
    sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[:, keep_idxs])
...
prune_in_channels = prune_out_channels
```

这是 channel-wise 模式下剪 **hidden 维**用的：hidden 维的一个通道对 attention 而言是 `o_proj` 的输出行 + `q/k/v_proj` 的输入列。`prune_in_channels = prune_out_channels` 表示该模块的输入输出维是同一个维度（**层内依赖**，见 3.9）。

**(g) LoRA 恢复** —— [post_training.py#L122-L133](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/post_training.py#L122-L133)

```python
config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=args.lora_target_modules.split(","),
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

默认 `lora_r=8`、`target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"`（[L241-L244](https://github.com/horseee/LLM-Pruner/blob/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2/post_training.py#L241-L244)）。LoRA 作用在**已经变窄**的 Linear 上，训练完可以 merge 回去，最终模型仍是小的 dense 模型。

### 3.9 Torch-Pruning 的 DepGraph（新版代码）

LLM-Pruner 内嵌的是 Torch-Pruning 的旧版快照；读新版更清晰。

**(a) 构图入口** —— [torch_pruning/dependency/graph.py#L69-L135](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L69-L135)

```python
assert torch.is_grad_enabled(), "Dependency graph relies on autograd for tracing. ..."
self.module2node = self._trace(model, example_inputs, forward_fn, output_transform=output_transform)
self._build_dependency(self.module2node)
shape_infer.init_shape_information(self)
index_mapping.update_index_mapping(self)
```

依赖图是**借助 autograd 图**追踪出来的——所以不能在 `no_grad` 下构图。

**(b) `_trace`：用 forward hook 把 `grad_fn` 映射回 module** —— [graph.py#L432-L464](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L432-L464)

```python
def _record_grad_fn(module, inputs, outputs):
    ...
    gradfn2module[outputs.grad_fn] = module
hooks = [
    m.register_forward_hook(_record_grad_fn)
    for m in model.modules()
    if (isinstance(m, registered_types) and m not in self.IGNORED_LAYERS_IN_TRACING)
]
```

每个可剪模块的输出张量都带一个 `grad_fn`（autograd 节点）。记下 `grad_fn → module`，然后从模型输出的 `grad_fn` 沿 `next_functions` 反向遍历（`_trace_computational_graph`，[L513](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L513)），就得到模块之间（含 add、cat、split、view 等无参算子）的连接关系。

**(c) `_build_dependency`：两条规则** —— [graph.py#L403-L430](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L403-L430)

```python
for in_node in node.inputs:
    handler = self.get_pruner_of_module(in_node.module).prune_out_channels
    trigger = self.get_pruner_of_module(node.module).prune_in_channels
    dep = Dependency(trigger=trigger, handler=handler, source=node, target=in_node)
    node.dependencies.append(dep)
for out_node in node.outputs:
    trigger = self.get_pruner_of_module(node.module).prune_out_channels
    handler = self.get_pruner_of_module(out_node.module).prune_in_channels
    ...
# Rule 2) - Intra-layer Dependency
# This is implictly implemented by assigning
# prune_out_channels=prune_in_channels in tp.pruner.function.BasePruningFunc
```

- **规则 1（层间）**：A 的输出连到 B 的输入 ⇒ "剪 A 的 out" 必然触发 "剪 B 的 in"，反之亦然。
- **规则 2（层内）**：对 BatchNorm/LayerNorm/逐元素算子这类输入输出是同一维的模块，`prune_in_channels` 和 `prune_out_channels` 是**同一个函数对象**，于是 "in 被剪" 这个 trigger 会匹配到它自己 "out 被剪" 的那条边，依赖得以继续传播。说明见 [function.py#L39-L59](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/pruner/function.py#L39-L59)。
- 残差 `x + f(x)`：add 节点的两个输入被耦合到一起，于是整个残差流上的所有层的 hidden 维被连成一个巨大的组——这就是为什么剪 hidden size 牵一发动全身。

**(d) `Dependency`：一条边 = (trigger, handler)** —— [torch_pruning/dependency/dependency.py#L9-L80](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/dependency.py#L9-L80)

`is_triggered_by(fn)` 就是 `fn == self.trigger`；`__call__` 调 `self.handler(self.target.module, idxs)` 去真正改权重。`index_mapping` 处理 `cat/split` 时的下标平移（例如 fused `qkv_proj` 被 split 成 3 份，q 的第 5 个通道在 qkv 里是第 5 个，k 的第 5 个是 `d+5`）。

**(e) `get_pruning_group`：从根出发 DFS 收集所有被触发的剪枝操作** —— [graph.py#L180-L274](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L180-L274)

```python
group = Group()
root_node = self.module2node[module]
group.add_dep(dep=Dependency(pruning_fn, pruning_fn, source=root_node, target=root_node), idxs=idxs)
def _fix_dependency_graph_non_recursive(dep, idxs, *args):
    processing_stack = [(dep, idxs)]
    while len(processing_stack) > 0:
        dep, idxs = processing_stack.pop(-1)
        node, fn = dep.target, dep.handler
        visited_node.add(node)
        for new_dep in node.dependencies:
            if new_dep.is_triggered_by(fn):
                new_indices = idxs
                for mapping in new_dep.index_mapping:
                    if mapping is not None:
                        new_indices = mapping(new_indices)
                ...
                    group.add_dep(new_dep, new_indices)
                    processing_stack.append((new_dep, new_indices))
```

一个显式栈的 DFS：当前操作是 `fn` 作用在 `node` 上；遍历 `node` 的所有出边，凡是 trigger 等于 `fn` 的都被激活，映射下标后入组、入栈。最终 `group` 里是一串 `(Dependency, idxs)`，**这就是一个"耦合结构"**。

**(f) `get_all_groups` 与 `Group.prune`** —— [graph.py#L276-L326](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py#L276-L326) 以每个根模块的全部输出通道建组，`visited_layers` 去重（同一组不会因不同根被重复产出）。[group.py#L32-L64](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/group.py#L32-L64) 的 `prune()` 依次调用每个 `dep(idxs)`。

**(g) 真正删权重：`LinearPruner`** —— [function.py#L212-L235](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/pruner/function.py#L212-L235)

```python
def prune_out_channels(self, layer, idxs):
    keep_idxs = list(set(range(layer.out_features)) - set(idxs))
    keep_idxs.sort()
    layer.out_features = layer.out_features-len(idxs)
    layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)   # 删行
    ...
def prune_in_channels(self, layer, idxs):
    ...
    layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)   # 删列
```

**(h) 用于 LLM 的新版示例** —— [examples/LLMs/prune_llm.py#L295-L375](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/examples/LLMs/prune_llm.py#L295-L375)：通过 `num_heads={q_proj: n_heads, k_proj: n_kv_heads, ...}`、`prune_num_heads=True`、`prune_head_dims=False`（注释：不剪 head_dim 就不用动 RoPE）、`ignored_layers=[model.lm_head]` 声明 LLM 结构；剪完后（L347-L371）手动回写 `config.hidden_size / num_attention_heads / intermediate_size` 等。和 LLM-Pruner 手动改 `num_heads` 是同一件事。

---

## 4. 方法对比表

| 方法 | 粒度 | 打分 | 需要的统计量 | 权重更新 | 校准成本 | 是否需要再训练 |
|---|---|---|---|---|---|---|
| Magnitude | 非结构化 / N:M | `|W|` | 无 | 无 | 0 | 高稀疏时必须 |
| Wanda | 非结构化 / N:M | `|W|·‖X_j‖`，**逐输出行**比较 | `diag(XXᵀ)`，`(C_in,)` | 无 | 一次前向，分钟级 | 一般不需要 |
| SparseGPT | 非结构化 / N:M | `w²/[H⁻¹]_qq`，块内比较 | `XXᵀ`，`(C_in,C_in)` | OBS 补偿（Cholesky） | 前向 + 每层 Cholesky，数小时 @175B | 一般不需要 |
| LLM-Pruner | 结构化（头 / MLP 中间维 / hidden） | 组 Taylor `Σ|W⊙∇W|` | 一次（或少量）反向梯度 | 物理删除 | 反向 1 次 + LoRA 数小时 | **需要** LoRA 恢复 |
| Torch-Pruning DepGraph | 结构化（通用） | 可插拔（Magnitude、Taylor、Hessian…） | 取决于 importance | 物理删除 | 构图 + 打分 | 视剪枝率 |

---

## 5. 常见困惑 / 坑

1. **"2:4" 在 wanda 里是剪掉 2 个还是保留 2 个？** `prune_n` 是 `topk(largest=False)` 选出的**被剪**个数。2:4 恰好两者相等；若你写 `1:4`，在该代码里会被解释为"每 4 个剪 1 个"（且 main.py 还会因为 `sparsity_ratio != 0.5` 断言失败）。
2. **非结构化 50% 稀疏不会让 PyTorch 推理变快**。权重只是变成 0，仍然是 dense 张量。要么用 2:4 + `torch.sparse.to_sparse_semi_structured` / cuSPARSELt，要么用专门的稀疏 kernel。
3. **Wanda 的 `scaler_row` 名字误导**：它是每个**输入通道（列）**的统计量，长度 `C_in`；"按行比较"指的是在 `W` 的每一行内排序。
4. **为什么 Wanda 要逐行比较？** 若整层全局比较，某些输出行可能被剪光（该神经元死掉），而另一些几乎不剪。LLM 的不同输出行尺度差异很大，逐行能保证每个输出都有均衡的输入。
5. **SparseGPT 为什么要 `percdamp`？** `XXᵀ` 在 N < C_in 或通道共线时奇异，Cholesky 会失败（报 `not positive-definite`）。1% 平均对角阻尼是经验值；仍报错时调大到 0.1。
6. **SparseGPT 的 `H` 显存**：down_proj 的输入是 `intermediate_size`（11008/14336），fp32 H 约 0.5–0.8 GB/层，加上 Cholesky 临时张量要留足显存。
7. **hook 闭包的后期绑定**：`for name in ...: m.register_forward_hook(lambda *a: f(name, *a))` 会全部指向最后一个 name；必须用工厂函数（源码里的 `add_batch(name)`）或默认参数绑定。
8. **逐层顺序剪枝 ≠ 块内顺序**：wanda/SparseGPT 在一个 decoder block 内的所有 Linear 用的是同一次前向的输入；只有 block 之间才用"已剪枝"的输出。
9. **结构化剪枝后忘记改 `num_heads` / `config`**：`view(..., num_heads, head_dim)` 立刻报错，或者更糟——形状碰巧对上但语义错乱。LLM-Pruner 在 `hf_prune.py#L156-L157`，Torch-Pruning 示例在 `prune_llm.py#L347-L371` 手动修复。
10. **GQA 下剪头**：q 头数与 kv 头数不同，必须按 kv 组剪（一个 kv 头连同它服务的所有 q 头一起删），否则 `num_key_value_groups` 不再是整数。Torch-Pruning 通过 `num_heads` 字典分别声明。
11. **DepGraph 必须在 grad 开启下构建**（`graph.py#L121` 的断言），因为它靠 autograd 图追踪。
12. **Taylor 重要性用 fp16 梯度**：LLM-Pruner 把模型 `half()` 后反向，`W*grad` 可能下溢；样本太少时重要性噪声也很大。

---

## 6. 自测题

1. 对 `W ∈ ℝ^{4096×11008}`（down_proj），Wanda 需要额外存多大的统计量？SparseGPT 呢？（答：`(11008,)` vs `(11008, 11008)`）
2. 证明：若 `H` 是对角阵，SparseGPT 的分数 `w_q²/[H⁻¹]_qq` 与 Wanda 分数的平方成正比。
3. 为什么 SparseGPT 能让所有输出行共享同一个 `H⁻¹` 序列？如果每行的剪枝顺序不同会发生什么？
4. `sparsegpt.py` 里 `W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))` 之后，`W1[:, i]` 等于什么？
5. 在 LLaMA 中，剪掉 MLP 中间维第 k 个通道需要修改哪些张量的哪些切片？剪掉 hidden 维第 k 个通道呢？
6. LLM-Pruner 中 `vectorize` 与 `param_first` 两种 Taylor 聚合有什么区别？哪个可能出现正负抵消？
7. DepGraph 的"层内依赖"是如何仅用 `prune_in_channels = prune_out_channels` 一行实现的？

## 7. 动手练习

1. **验证 Wanda ≈ 对角 SparseGPT**：用 2.2 的代码，把 `H` 替换成 `torch.diag(H.diag())`，并去掉补偿步骤，比较选出的 mask 与 2.1 的 Wanda mask 的重合率（应接近 100%，差异来自块内全局 vs 逐行比较）。
2. **重建误差对比**：随机生成带离群通道的 `X`（例如把 5 个输入通道放大 50 倍），比较 Magnitude / Wanda / SparseGPT 的 `‖(W−Ŵ)X‖`。观察 Magnitude 在离群激活下崩坏。
3. **2:4 改造 SparseGPT**：在 2.2 中加入 `prune_n, prune_m` 分支（参考源码 L94-L96），并检查每 4 列中恰有 2 个零。
4. **手写一个 MLP 组剪枝**：对一个 `LlamaMLP`-like 模块（gate/up/down），用 `Σ|W⊙∇W|` 计算中间维 I 个通道的组重要性（三个切片相加），删掉最低的 25%，构造新的更小的 `nn.Linear` 并验证输出形状与原模块一致。
5. **读 DepGraph**：`pip` 环境中（自行准备）对一个两层 MLP 加残差构图，`print(DG.get_pruning_group(fc1, tp.prune_linear_out_channels, idxs=[0,1]))`，看看组里有哪些依赖；再对残差流维度试一次，体会"牵一发动全身"。
