# 08 · LoRA 与 PEFT 源码导读

> 读完本篇你应该能回答：LoRA 的 `A`、`B` 分别是什么形状、为什么 `B` 初始化为 0？`α/r` 这个缩放从哪来、rsLoRA 改了什么？PEFT 是怎样“不改模型代码”就把 LoRA 塞进任意 HuggingFace 模型的？merge/unmerge 在源码里就是哪两行？QLoRA、DoRA、PiSSA 在 PEFT 里分别落在哪个函数？vLLM 怎样在一个 batch 里同时服务多个 LoRA？

本篇引用的仓库与 commit（所有链接为固定 commit 的 permalink）：

| 仓库 | 本地路径 | commit |
|---|---|---|
| microsoft/LoRA | `repos/07-finetune/LoRA` | `c4593f060e6a368d7bb5af5273b8e42810cdef90` |
| huggingface/peft | `repos/07-finetune/peft` | `eac46efeef6e5f63ecc38de5c9148513df602b29` |
| huggingface/accelerate | `repos/07-finetune/accelerate` | `ccfdff7dbbd6a8c455853a9e5eb6396ea51ffd45` |
| vllm-project/vllm | `repos/03-inference/vllm` | `f5fefbffe51e1cc91e0d6507d272f337473989b3` |

---

## 1. 要解决的问题 / 一句话直觉

**问题**：全量微调一个 7B 模型，要为每个参数存梯度和 Adam 的两个状态，显存约是推理时的 4 倍以上；而且每个下游任务都要存一份完整模型副本。

**一句话直觉**：冻结原权重 `W`，只学一个**低秩增量** `ΔW = B·A`（秩 r 很小，例如 8）。微调后的权重是 `W + (α/r)·B·A`。可训练参数从 `d_out·d_in` 降到 `r·(d_in + d_out)`；推理时还能把 `ΔW` 直接加回 `W`（merge），**零额外延迟**。

背后的假设：微调带来的权重变化本身是“低内在秩”的——不需要满秩矩阵就能表达任务适配。

---

## 2. 核心原理（带形状）

### 2.1 前向

对一个线性层 `y = x W^T + b`：

```
x       : [N, d_in]            # N = batch*seq
W       : [d_out, d_in]        # 冻结
A       : [r, d_in]            # 可训练，"降维"（shrink）
B       : [d_out, r]           # 可训练，"升维"（expand）
s       = α / r                # 缩放（rsLoRA: α / sqrt(r)）

y = x W^T + b + s · (dropout(x) A^T) B^T
           └─[N,d_out]─┘   └──[N,r]──┘ └─[N,d_out]
```

计算顺序很重要：先算 `x A^T`（`[N, r]`，很小），再乘 `B^T`；**不要**先算 `B A`（`[d_out, d_in]`，和 W 一样大）——那只在 merge 时做一次。

### 2.2 初始化：为什么 B=0

- `A` 用 Kaiming-uniform（与 `nn.Linear` 默认相同）随机初始化，`B = 0`。
- 于是训练开始时 `ΔW = B A = 0`，模型输出与原模型**完全一致**，微调从预训练点出发。
- 为什么不两个都为 0？若 `A=B=0`，`∂L/∂B = s·(xA^T)^T·g = 0`、`∂L/∂A = s·B^T…= 0`，梯度全为零，永远学不动。一个随机、一个为零，就能打破对称：第一步 `B` 有梯度（因为 `xA^T ≠ 0`），之后 `A` 也会有梯度。

### 2.3 缩放 α/r 与 rsLoRA

`s = α/r` 的作用：改变 r 时不必重新调学习率（论文的说法是“调 α 大致等价于调学习率”）。rsLoRA（Rank-Stabilized LoRA）指出当 r 增大时 `α/r` 会让更新幅度过小，建议 `s = α/√r`。PEFT 用 `use_rslora=True` 切换（见 4.4）。

### 2.4 merge / unmerge

```
merge  : W ← W + s · B A          # [d_out, d_in]，之后 forward 只走 base 层
unmerge: W ← W − s · B A
```

注意数值：在 bf16/fp16 或量化权重上 merge→unmerge 不一定精确还原（舍入误差），QLoRA 的 4bit 权重 merge 后还要重新量化。

### 2.5 注入在哪里

原论文只对注意力的 `W_q, W_v` 加 LoRA；后来的实践（QLoRA 等）常对所有线性层（`q,k,v,o,gate,up,down`）加。PEFT 用 `target_modules` 指定，`"all-linear"` 表示所有 Linear/Conv1D（排除输出层）。

### 2.6 变体速览

| 变体 | 改了什么 | 数学 |
|---|---|---|
| QLoRA | 底座 W 以 4bit NF4 存储并冻结，LoRA 参数用 bf16/fp32 | `y = dequant(W_q) x + s·B A x` |
| DoRA | 把权重分解为“幅度×方向”，方向用 LoRA 更新，幅度单独学一个向量 m | `W' = m ⊙ (W + sBA) / ‖W + sBA‖_c` |
| PiSSA | 用 W 的 top-r 奇异分量初始化 A、B，残差作为新的冻结 W | `W = W_res + s·B A`，`B A ≈ U_r S_r V_r^T / s` |
| rsLoRA | 缩放改为 `α/√r` | — |

---

## 3. 最小实现（PyTorch）

一个带 merge/unmerge 的 `LoRALinear`，以及一个“按名字替换子模块”的注入函数（PEFT 的核心机制就是这个）。

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, rslora=False):
        super().__init__()
        self.base = base                                           # 原层：W [out, in], b [out]
        for p in self.base.parameters():
            p.requires_grad_(False)                                # 冻结
        self.r = r
        self.scaling = alpha / math.sqrt(r) if rslora else alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))   # [r, in]
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))  # [out, r]  B=0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))      # 与 nn.Linear 默认一致
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self):                                        # [out, in]
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):                                          # x: [..., in]
        y = self.base(x)
        if self.merged:
            return y
        return y + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

def inject_lora(model: nn.Module, target_names=("q_proj", "v_proj"), **kw):
    """把名字以 target_names 结尾的 nn.Linear 替换成 LoRALinear（PEFT 的做法同理）。"""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(parent, child_name, LoRALinear(child, **kw))
    for n, p in model.named_parameters():                          # 只训练 lora_ 参数
        p.requires_grad_("lora_" in n)
    return model

if __name__ == "__main__":
    torch.manual_seed(0)
    class Attn(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        def forward(self, x):
            return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
    m = Attn(); x = torch.randn(4, 32); y0 = m(x)
    inject_lora(m, r=4, alpha=8)
    assert torch.allclose(m(x), y0)                                # B=0 => 初始输出不变
    with torch.no_grad():
        m.q_proj.lora_B.normal_()                                  # 假装训练过
    y1 = m(x); m.q_proj.merge(); y2 = m(x); m.q_proj.unmerge(); y3 = m(x)
    assert torch.allclose(y1, y2, atol=1e-5) and torch.allclose(y1, y3, atol=1e-5)
    print("trainable:", sum(p.numel() for p in m.parameters() if p.requires_grad),
          "total:", sum(p.numel() for p in m.parameters()))
```

---

## 4. 源码导读（按阅读顺序）

阅读路线：原版 loralib（最短、最直观）→ PEFT 的 `LoraLayer.update_layer` / `Linear.forward/merge` → PEFT 怎样找到并替换目标模块 → 变体（QLoRA / DoRA / PiSSA / rsLoRA）→ accelerate 与 PEFT 的交互 → vLLM 多 LoRA 服务。

### 4.1 microsoft/LoRA：`loralib/layers.py`

#### (1) `LoRALayer`：混入类，存超参

- L12–L30：https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py#L12-L30

```python
self.r = r
self.lora_alpha = lora_alpha
if lora_dropout > 0.:
    self.lora_dropout = nn.Dropout(p=lora_dropout)
else:
    self.lora_dropout = lambda x: x
self.merged = False                 # 当前 W 里是否已经加上了 BA
self.merge_weights = merge_weights  # eval() 时是否自动 merge
```

#### (2) `Linear`：继承 `nn.Linear`，额外两个参数

- L90–L152：https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py#L90-L152

`__init__`（L92–L117）：

```python
nn.Linear.__init__(self, in_features, out_features, **kwargs)     # 自己就是那个 W
if r > 0:
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))     # [r, d_in]
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))    # [d_out, r]
    self.scaling = self.lora_alpha / self.r
    self.weight.requires_grad = False                                       # 冻结 W
self.reset_parameters()
if fan_in_fan_out:
    self.weight.data = self.weight.data.transpose(0, 1)
```

`fan_in_fan_out`：GPT-2 的 `Conv1D` 把权重存成 `[d_in, d_out]`（与 `nn.Linear` 相反），加载这种 checkpoint 时用它转置。后面的 `T(w)` 小函数就是为此准备的。

`reset_parameters`（L119–L125）：

```python
nn.Linear.reset_parameters(self)
if hasattr(self, 'lora_A'):
    # initialize B the same way as the default for nn.Linear and A to zero
    # this is different than what is described in the paper but should not affect performance
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)
```

⚠️ 这里的**注释和代码不一致**：注释说“B 用 Linear 默认、A 为 0”，代码实际是 **A 用 Kaiming、B 为 0**（与论文一致，`MergedLinear.reset_parameters` L196–L201 的注释就写对了）。读源码时以代码为准。

`train(mode)`（L127–L142）——把 merge/unmerge 挂在 `model.train()` / `model.eval()` 上：

```python
nn.Linear.train(self, mode)
if mode:                                           # 切到训练：如果已 merge，要减回去
    if self.merge_weights and self.merged:
        if self.r > 0:
            self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        self.merged = False
else:                                              # 切到推理：merge 进 W
    if self.merge_weights and not self.merged:
        if self.r > 0:
            self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        self.merged = True
```

`forward`（L144–L152）：

```python
if self.r > 0 and not self.merged:
    result = F.linear(x, T(self.weight), bias=self.bias)                         # [N, d_out]
    result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
    return result                                                                # x@A^T: [N,r]; @B^T: [N,d_out]
else:
    return F.linear(x, T(self.weight), bias=self.bias)
```

注意 dropout 只作用在 LoRA 支路的输入上，不影响冻结的主干。

#### (3) `MergedLinear`：给融合的 qkv 只加部分 LoRA

- L155–L244：https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py#L155-L244

GPT-2 的 `c_attn` 是一个 `[d, 3d]` 的融合层。想只对 q、v 加 LoRA，就传 `enable_lora=[True, False, True]`：

```python
self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))            # [2r, d]
self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))  # [2d, r]
self.lora_ind = ...  # 长度 out_features 的 bool，标记哪些输出列有 LoRA
```

`merge_AB`（L208–L216）用 `F.conv1d(..., groups=sum(enable_lora))` 一次算出两个 `B_i A_i`（分组卷积 = 分块矩阵乘），`zero_pad` 再把结果填回 `[3d, d]` 中对应的行。这个“多切片共用一层”的问题在 vLLM 里以 `n_slices` / `output_slices` 的形式再次出现（4.7）。

#### (4) `loralib/utils.py`：只训练 LoRA、只保存 LoRA

- `mark_only_lora_as_trainable` L13 起 / `lora_state_dict` L33 起：https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/utils.py#L13-L49

```python
for n, p in model.named_parameters():
    if 'lora_' not in n:
        p.requires_grad = False
...
return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
```

靠参数名里有 `lora_` 来识别——PEFT 至今仍沿用 `prefix = "lora_"`（`tuners/lora/model.py` L238）。

loralib 的局限：要求你**改模型代码**，把 `nn.Linear` 换成 `lora.Linear`。PEFT 解决的正是这一点。

### 4.2 PEFT `LoraLayer`：多 adapter 的参数容器

- `class LoraLayer` L108–L156：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/layer.py#L108-L156

与 loralib 最大的不同：**组合（composition）而非继承**，且**每个超参都是 dict（按 adapter 名索引）**：

```python
self.base_layer = base_layer               # 被包住的原始 nn.Linear（或 bnb.Linear4bit 等）
self.r = {}
self.lora_alpha = {}
self.scaling = {}
self.lora_dropout = nn.ModuleDict({})
self.lora_A = nn.ModuleDict({})            # adapter_name -> nn.Linear(in, r, bias=False)
self.lora_B = nn.ModuleDict({})            # adapter_name -> nn.Linear(r, out)
self.merged_adapters = []
self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
```

所以一个层可以同时挂 `"default"`、`"math"`、`"code"` 多个 adapter，用 `set_adapter` 切换激活哪个。`base_layer` 也可能本身又是一个 tuner 层，`BaseTunerLayer.get_base_layer`（`tuners/tuners_utils.py` L1881–L1891）会循环 `while hasattr(base_layer, "base_layer")` 剥到最底层。

#### `update_layer`：真正创建 A/B 的地方

- L219–L333：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/layer.py#L219-L333

```python
lora_variant = self.resolve_lora_variant(config=config)        # DoRA 等变体在这里挂上
...
self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)   # weight: [r, in]
self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)  # weight: [out, r]
...
if use_rslora:
    self.scaling[adapter_name] = lora_alpha / math.sqrt(r)    # rsLoRA
else:
    self.scaling[adapter_name] = lora_alpha / r
...
if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
    with gather_params_ctx(self.get_base_layer().weight):
        self.pissa_init(adapter_name, init_lora_weights)
elif ... "corda" / "astra" / "olora" / "mica" / "loftq" / "eva" / "orthogonal" / "lora_ga"
elif init_lora_weights:
    self.reset_lora_parameters(adapter_name, init_lora_weights)
```

注意 `nn.Linear(in, r)` 的 `.weight` 形状是 `[r, in]`，与 loralib 的 `lora_A` 完全相同；`lora_A(x)` 就等于 `x @ A^T`。`gather_params_ctx` 是为了在 DeepSpeed ZeRO-3 下（权重被切片）临时收集完整权重做 SVD 等初始化。

#### `reset_lora_parameters`：默认初始化

- L335–L386：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/layer.py#L335-L386

```python
if init_lora_weights is True:
    # initialize A the same way as the default for nn.Linear and B to zero
    nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
elif init_lora_weights.lower() == "gaussian":
    nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
...
nn.init.zeros_(self.lora_B[adapter_name].weight)
```

Embedding 反过来（`lora_embedding_A` 为 0，`lora_embedding_B` 正态），与 loralib `Embedding.reset_parameters` 一致——因为 embedding 的“输入”是 one-hot 查表，A 在前面。后半段还会在张量并行（TP）下 broadcast LoRA 权重，保证各 rank 初始值一致。

### 4.3 PEFT `Linear`：forward / merge / unmerge / get_delta_weight

- `class Linear` L982 起：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/layer.py#L982-L1187

`lora_variants`（L1008–L1022）是一张“配置字段 → 变体类”的表：`("use_dora",): variants.DoraLinearVariant`、`("arrow_config",)`、`("use_bdlora",)`、`("alora_invocation_tokens",)` 等。`resolve_lora_variant` 根据 config 选一个。

#### `forward`（L1146–L1187）

```python
if self.disable_adapters:                         # with model.disable_adapter(): 走原模型
    if self.merged:
        self.unmerge()
    result = self.base_layer(x, *args, **kwargs)
elif adapter_names is not None:                   # 同一 batch 不同样本用不同 adapter
    result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, ...)
elif self.merged:                                 # 已 merge：只走 base
    result = self.base_layer(x, *args, **kwargs)
else:
    result = self.base_layer(x, *args, **kwargs)  # [N, out]
    for active_adapter in self.active_adapters:   # 可以同时激活多个 adapter（相加）
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = self._cast_input_dtype(x, lora_A.weight.dtype)          # 例如 base 是 4bit/bf16, LoRA 是 fp32
        if active_adapter not in self.lora_variant:                 # vanilla LoRA
            result = result + lora_B(lora_A(dropout(x))) * scaling
        else:
            result = self.lora_variant[active_adapter].forward(self, active_adapter=active_adapter, x=x, result=result, ...)
    result = result.to(torch_result_dtype)
```

核心就一行：`result = result + lora_B(lora_A(dropout(x))) * scaling`，与第 2.1 节公式逐项对应。

#### `_mixed_batch_forward`（L925–L969）：PEFT 的“多 LoRA 推理”

```python
result = self.base_layer(x, *args, **kwargs)
unique_adapters = set(adapter_names)
for adapter in unique_adapters:
    sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
for i, active_adapter in enumerate(unique_adapters):
    if active_adapter == "__base__":
        continue
    sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
    lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
    result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
```

先用共享的 base 层算完整个 batch，再按 adapter 名把样本分组，各组走自己的 A/B。这和 MoE 的“按专家分组”是同一个模式（见第 07 篇）；vLLM 的 punica kernel 把这个 Python 循环融合成一个 GPU kernel（4.7）。

#### `get_delta_weight`（L1116–L1144）与 `merge`（L1024–L1091）/ `unmerge`（L1093–L1114）

```python
output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]   # [out, in]
```

`merge` 的非 safe 路径：

```python
delta_weight = self.get_delta_weight(active_adapter)
base_layer.weight.data += delta_weight
...
self.merged_adapters.append(active_adapter)
```

`safe_merge=True` 时先在 `clone()` 上加，`torch.isfinite(...).all()` 检查无 NaN 后再写回。`unmerge` 按 `merged_adapters.pop()` 的**逆序**减回去。`merged` 属性就是 `bool(self.merged_adapters)`（tuners_utils.py L1977–L1978）。`lora_bias=True` 时还会把 `lora_B.bias * scaling` 加到 base bias 上。

`get_delta_weight` 里还有个细节：CPU 上的 fp16/bf16 会先转 fp32 再相乘（CPU 半精度 matmul 慢）。

### 4.4 PEFT 如何把 LoRA 注入模型：`get_peft_model` → `inject_adapter` → `_create_and_replace`

调用链：

```
peft.get_peft_model(model, LoraConfig(...))           # mapping_func.py L105
  └─ PeftModelForCausalLM(...) → LoraModel(model, config)   # LoraModel 继承 BaseTuner
       └─ BaseTuner.__init__ → self.inject_adapter(...)       # tuners_utils.py L344
            ├─ for key, module in named_modules: _check_target_module_exists(config, key)
            ├─ parent, target, target_name = _get_submodules(model, key)
            ├─ self._create_and_replace(config, adapter_name, target, target_name, parent, current_key=key)
            │     ├─ 已经是 LoraLayer → target.update_layer(...)          # 加第二个 adapter
            │     └─ 否则 → new_module = _create_new_module(...)           # 按类型分派
            │              self._replace_module(parent, target_name, new_module, target)  # setattr 替换
            └─ self._mark_only_adapters_as_trainable(model)                # tuners_utils.py L1143
```

#### (1) 配置：`LoraConfig`

- `tuners/lora/config.py` L514 起，字段 L699–L760、`use_dora` L911：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/config.py#L699-L760

```python
r: int = field(default=8, ...)
target_modules: Optional[Union[list[str], str]] = field(default=None, ...)   # 列表=后缀匹配；字符串=正则；"all-linear"
exclude_modules: ...
lora_alpha: int = field(default=8, ...)
lora_dropout: float = field(default=0.0, ...)
bias: Literal["none", "all", "lora_only"] = ...
use_rslora: bool = field(default=False, ...)          # α/√r
modules_to_save: Optional[list[str]] = ...            # 例如分类头，整层训练并保存
init_lora_weights: bool | Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]",
                                  "corda", "astra", "loftq", "orthogonal", "mica"] ...
rank_pattern: Optional[dict] = ...                     # 不同层不同 r
use_dora: bool = field(default=False, ...)
```

#### (2) 匹配：`check_target_module_exists`

- `tuners/tuners_utils.py` L2344–L2420：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/tuners_utils.py#L2344-L2420

```python
if isinstance(config.target_modules, str):
    target_module_found = match_target_against_key(config.target_modules, key)    # 字符串：正则全匹配
elif key in config.target_modules:
    target_module_found = True
else:
    target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)  # 列表：后缀
```

`key` 是 `named_modules()` 给出的完整路径，例如 `model.layers.3.self_attn.q_proj`。`target_modules=["q_proj","v_proj"]` 会命中所有层的 q/v。`exclude_modules` 和 `modules_to_save` 在前面优先排除。`layers_to_transform` 用正则 `.*?\.[^.]*\.(?P<idx>\d+)\.` 抽取**第一个**数字作为层号——注释特别说明这样不会误取 MoE 的专家编号（`...layers.1.experts.0...`）。

#### (3) 遍历与替换：`inject_adapter` / `_get_submodules`

- `inject_adapter` L799 起（主循环 L948–L979）：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/tuners_utils.py#L948-L979

```python
for key, module in named_modules:
    ...
    result = self._check_target_module_exists(peft_config, key)
    ...
    else:
        targeted_module_names.append(key)
        parent, target, target_name = _get_submodules(model, key)
        ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
        with ctx():
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
```

`_get_submodules`（`utils/other.py` L1059–L1063）只是字符串切分：

```python
parent = model.get_submodule(".".join(key.split(".")[:-1]))   # model.layers.3.self_attn
target_name = key.split(".")[-1]                               # "q_proj"
target = model.get_submodule(key)                              # 原 nn.Linear
```

#### (4) `LoraModel._create_and_replace`

- `tuners/lora/model.py` L294–L468：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/model.py#L294-L468

```python
r = lora_config.rank_pattern.get(r_key, lora_config.r)                # 每层可以有不同 r / alpha
alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
kwargs = {"r": r, "lora_alpha": alpha, ..., "loaded_in_8bit": ..., "loaded_in_4bit": ...}
...
if is_existing_lora_layer:
    target.update_layer(adapter_name, r, lora_alpha=alpha, target_name=current_key, config=lora_config)
else:
    new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
    if adapter_name not in self.active_adapters:
        new_module.requires_grad_(False)          # 额外加的 adapter 默认不训练
    self._replace_module(parent, target_name, new_module, target)
```

`_replace_module`（L470–L497）核心就是 `setattr(parent, child_name, new_module)`，然后把新建的 `lora_*` 子模块搬到原权重所在的 device。这就是“不改模型源码”的秘密：**Python 对象的属性可以随时被替换**，模型 forward 里写的 `self.q_proj(x)` 会自动调用新的 LoRA 层。

#### (5) 分派：`_create_new_module` 与 `dispatch_default`

- `_create_new_module` L499–L565：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/model.py#L499-L565

```python
if is_bnb_available():
    from .bnb import dispatch_bnb_8bit
    dispatchers.append(dispatch_bnb_8bit)
if is_bnb_4bit_available():
    from .bnb import dispatch_bnb_4bit
    dispatchers.append(dispatch_bnb_4bit)
dispatchers.extend([dispatch_eetq, dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_hqq, dispatch_inc,
                    dispatch_torchao, dispatch_megatron, dispatch_transformer_engine, dispatch_default])
for dispatcher in dispatchers:
    new_module = dispatcher(target, adapter_name, config=lora_config, **kwargs)
    if new_module is not None:  # first match wins
        break
```

“责任链”：量化层优先匹配（因为 `bnb.nn.Linear4bit` 也是 `nn.Linear` 的子类，若 default 先匹配就会出错），最后才是 `dispatch_default`（`layer.py` L2811–L2853）——按 `nn.Embedding / Conv2d / Conv3d / Conv1d / MultiheadAttention / nn.Linear / Conv1D` 选择对应 LoRA 类；遇到 GPT-2 的 `Conv1D` 会自动把 `fan_in_fan_out` 设为 True。

#### (6) 冻结：`_mark_only_adapters_as_trainable`

- `tuners/tuners_utils.py` L506 起：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/tuners_utils.py#L506-L530

除了 adapter 参数（通过遍历 tuner 层确定，而不是仅靠名字匹配，注释解释了避免误伤名字里恰好含 `lora_` 的基座模块）和 `modules_to_save` 包装的模块外，一律 `p.requires_grad = False`；再按 `bias` 配置放开 bias。

#### (7) 卸载：`merge_and_unload`

- `_unload_and_optionally_merge` L679–L740 与 `merge_and_unload` L742 起：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/tuners_utils.py#L679-L760

```python
elif hasattr(target, "base_layer"):
    if merge:
        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
    self._replace_module(parent, target_name, target.get_base_layer(), target)
```

先 merge，再把 LoRA 包装层替换回 `base_layer`，得到一个结构与原模型完全相同的普通模型（可直接用 vLLM 等加载）。docstring 提醒：“this is not an in-place operation”，要接住返回值。

### 4.5 变体：QLoRA、DoRA、PiSSA、rsLoRA

#### (1) QLoRA：`tuners/lora/bnb.py` 的 `Linear4bit`

- `class Linear4bit` L313 起，`forward` L513–L565，`dispatch_bnb_4bit` L571–L591：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/bnb.py#L513-L591

PEFT 里的 QLoRA 并没有独立的“QLoRA 类”：它 = `bitsandbytes` 4bit 底座（transformers 用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 加载）+ 普通 LoRA 支路。`dispatch_bnb_4bit` 在 `loaded_in_4bit` 且 `target` 是 `bnb.nn.Linear4bit` 时创建包装层。`forward`：

```python
result = self.base_layer(x, *args, **kwargs)     # bnb 内部 dequant NF4 -> compute_dtype 再 matmul
# As per Tim Dettmers, for 4bit, we need to defensively clone here.
result = result.clone()
...
requires_conversion = not torch.is_autocast_enabled()
if requires_conversion:
    expected_dtype = result.dtype
    x = self._cast_input_dtype(x, lora_A.weight.dtype)
output = lora_B(lora_A(dropout(x))) * scaling    # LoRA 支路在高精度中计算
if requires_conversion:
    output = output.to(expected_dtype)
result = result + output
```

`merge`（L355 起）要先 `dequantize_bnb_weight`，加上 delta 后再重新构造 `Params4bit` 量化——并发出警告 “Merge lora module to 4-bit linear may get different generations due to rounding errors.”

训练前通常调用 `prepare_model_for_kbit_training`（`src/peft/utils/other.py` L158 起，https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/utils/other.py#L190-L250）：冻结全部参数（L193–L195）、把非 4bit 的 fp16/bf16 参数（LayerNorm 等）转成 fp32（L204–L209）、开启 gradient checkpointing 并 `enable_input_require_grads()`（L232–L233）——后者让 embedding 输出 `requires_grad=True`，否则冻结模型 + 重入式 checkpoint 时梯度传不回 LoRA。

#### (2) DoRA：`tuners/lora/dora.py` + `variants.py`

- `DoraLinearLayer` L62 起，`get_weight_norm` L86–L91，`update_layer` L101–L130，`forward` L132–L166：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/dora.py#L86-L166
- `DoraLinearVariant` L137 起（`init` / `merge_safe` / `merge_unsafe` / `unmerge` / `forward`）：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/variants.py#L137-L255

初始化时幅度向量 `m` = 初始权重的逐行范数：

```python
weight = transpose(weight, self.fan_in_fan_out)
weight = weight + scaling * lora_weight
weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)     # [out]：每个输出通道一行的 L2 范数
...
self.weight = nn.Parameter(weight_norm, requires_grad=True)         # 这就是 m（lora_magnitude_vector）
```

`forward` 计算“增量输出”（加在 base 结果上）：

```python
weight_norm = self.get_weight_norm(weight=weight, lora_weight=lora_weight.detach(), scaling=scaling)
weight_norm = weight_norm.detach()                    # 论文 4.3 节：把 ||V+ΔV||_c 当常数，不回传梯度
mag_norm_scale = (magnitude / weight_norm).view(1, -1)        # [1, out] = m / ||W + sBA||
lora_result = lora_B(lora_A(x))
...
result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
```

推导：DoRA 输出 = `(m/‖W+sBA‖) ⊙ (xW^T + s·xA^TB^T)`。base 层已经算了 `xW^T`（即 `base_result`，去掉 bias），所以只需再加 `(scale−1)·base_result + scale·s·lora_result`。`DoraLinearVariant.forward`（variants.py L227–L255）把它加到 `result` 上。注意每次 forward 都要算整个 `W + sBA` 的范数，开销比 LoRA 大，所以 config 的帮助文本建议推理时 merge。merge 公式（L188–L190）：`new_weight = (m / weight_norm) * (orig_weight + delta_weight)`，且 `weight_norm` 被缓存，因为 unmerge 时无法从已 merge 的权重重新算出它（L184–L187 注释）。

#### (3) PiSSA：`LoraLayer.pissa_init`

- L432–L465：https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/tuners/lora/layer.py#L432-L465

```python
V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)     # weight: [out, in]
Vr = V[:, : self.r[adapter_name]]          # [out, r]
Sr = S[: self.r[adapter_name]]
Sr /= self.scaling[adapter_name]           # 预先除以 s，使 s·B·A 恰好等于主奇异分量
Uhr = Uh[: self.r[adapter_name]]           # [r, in]
lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr  # [r, in]
lora_B = Vr @ torch.diag(torch.sqrt(Sr))   # [out, r]
...
weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A   # 残差 W_res 作为新的冻结权重
self.get_base_layer().weight.data = weight
```

与普通 LoRA 的区别：A、B 都非零（初始输出依然不变，因为 `W_res + sBA = W`），训练直接更新“最重要的”奇异方向，收敛更快。`pissa_niter_K` 用 `svd_lowrank` 快速近似。代价：**底座权重被改了**，保存/分享时要用 `subtract_mutated_init`（model.py L1059）等把它转换回“标准 LoRA”形式。

#### (4) rsLoRA

就是 `update_layer` 里的那一行 `self.scaling[adapter_name] = lora_alpha / math.sqrt(r)`（见 4.2）。vLLM 加载时也读取 `use_rslora`：`vllm/lora/peft_helper.py` 的 `_scaling`（L117–L120）同样返回 `alpha / math.sqrt(rank)`。

### 4.6 accelerate 与 PEFT 如何配合训练

accelerate 不认识 LoRA，它只做“设备放置 / DDP / FSDP / DeepSpeed / 混合精度”的包装；PEFT 模型对它而言就是普通 `nn.Module`，只是大部分参数 `requires_grad=False`。典型流程：

```python
model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_cfg)   # 可选 4bit
model = prepare_model_for_kbit_training(model)                                   # QLoRA 时
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"))
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model, optimizer, dl = accelerator.prepare(model, optimizer, dl)                 # DDP/FSDP 包装
```

两个交互点值得读源码：

1. **只保存 adapter**：accelerate 的 `_get_model_state_dict` / `_set_model_state_dict`（`src/accelerate/utils/fsdp_utils.py` L57–L94，https://github.com/huggingface/accelerate/blob/ccfdff7dbbd6a8c455853a9e5eb6396ea51ffd45/src/accelerate/utils/fsdp_utils.py#L57-L94）在 `adapter_only and is_peft_model(model)` 时调用 PEFT 的 `get_peft_model_state_dict` / `set_peft_model_state_dict`（或 FSDP2 下用 `ignore_frozen_params=True`）。`is_peft_model` 定义在 `utils/modeling.py` L73–L79：`isinstance(extract_model_from_parallel(model), PeftModel)`。
2. **FSDP 的包装粒度**：FSDP1 默认把一个 wrap 单元内的参数 flatten 成一个 FlatParameter，冻结和可训练参数混在一起需要 `use_orig_params=True`（accelerate 的 `FullyShardedDataParallelPlugin.use_orig_params`，dataclasses.py L1632 起的文档；FSDP2 中此选项已过时，见 L1940–L1942 的警告）。PEFT 提供 `fsdp_auto_wrap_policy`（`src/peft/utils/other.py` L1310 起），其 `lambda_policy_fn` 把“叶子模块且 `weight.requires_grad`”的模块（即 LoRA 的 A/B）单独包成 FSDP 单元，与 transformer block 的包装策略 `_or_policy` 组合。

### 4.7 vLLM：多 LoRA 服务（punica 风格）

问题：一个推理服务同时托管很多 LoRA（每个用户一个），同一个 batch 里不同请求用不同 LoRA。逐个 merge 不现实（merge 后就不能服务别的 LoRA 了），只能**保持 unmerged**，并且要把“不同 token 乘不同 A/B”高效地做成 kernel。

#### (1) 层替换：`from_layer`

- `vllm/lora/utils.py` L107–L125：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/utils.py#L107-L125

```python
for lora_cls in _all_lora_classes:
    if lora_cls.can_replace_layer(source_layer=layer, lora_config=lora_config, ...):
        instance_layer = lora_cls(layer)
        instance_layer.create_lora_weights(max_loras, lora_config, model_config)
        return instance_layer
```

`LoRAModelManager._create_lora_modules`（`model_manager.py` L467 起）对模型逐个模块调用 `from_layer` 并 `replace_submodule`——和 PEFT 的“替换子模块”是同一招。

#### (2) 预分配的“堆叠”权重槽：`create_lora_weights`

- `BaseLinearLayerWithLoRA` L70 起，`create_lora_weights` L104–L155，`set_lora` L162–L188：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/layers/base_linear.py#L104-L188

```python
self.lora_a_stacked = tuple(torch.zeros(max_loras, 1, lora_a_out_size, self.input_size, ...)  # [L, 1, max_r, in]
                            for _ in range(self.n_slices))
self.lora_b_stacked = tuple(torch.zeros(max_loras, 1, lora_b_out_size, lora_config.max_lora_rank, ...)  # [L, 1, out, max_r]
                            for _ in range(self.n_slices))
```

- `max_loras` 个**槽位**一次性分配好（`vllm/config/lora.py`：`max_loras` 默认 1、`max_lora_rank` 默认 16）。
- `set_lora` 把某个 LoRA 的 A/B 拷进槽位 `index`：`self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(lora_a)`——rank 小于 `max_lora_rank` 的部分保持为 0（零填充不影响结果）。
- `n_slices`：QKV / gate_up 这种融合层有多个切片（`MergedColumnParallelLinearWithLoRA`、`QKVParallelLinearWithLoRA` 等，`layers/column_parallel_linear.py` L184、L388），每个切片一套 A/B——正是 loralib `MergedLinear` 要解决的问题。
- TP 下：ColumnParallel 切 B 的输出维，RowParallel 切 A 的输入维（`slice_lora_a/slice_lora_b`）。

缩放去哪了？`LoRALayerWeights.optimize`（`vllm/lora/lora_weights.py` L36–L42）在加载时执行 `self.lora_b *= self.scaling; self.scaling = 1`，把 `α/r` 预先乘进 B。所以 `add_lora_linear` 调用时传的 scale 是 `1.0`。

槽位管理：`LoRAModelManager.activate_adapter`（`model_manager.py` L346 起）找一个空槽 `index`，对每个 LoRA 层调用 `module.set_lora(index, lora_a, lora_b)`；`_set_adapter_mapping`（L424 起）把“每个 token 属于哪个槽”的映射交给 punica wrapper（`update_metadata`）。

#### (3) 前向：`_apply_lora_to_output` → `add_lora_linear` = shrink + expand

- `base_linear.py` L198–L242：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/layers/base_linear.py#L198-L242
- `PunicaWrapperGPU.add_lora_linear` L206 起，`add_shrink` L91–L122：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/punica_wrapper/punica_gpu.py#L91-L265

```python
output = self._get_quant_method().apply(self.base_layer, x, bias)       # 共享的 base GEMM（可量化）
lora_output = self.punica_wrapper.add_lora_linear(
    output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices)
```

`add_lora_linear` 的 docstring 写明了语义：

```
for i in range(len(lora_a_stacked)):
    y[i] += x[i].unsqueeze(0) @ lora_a_stacked[indices[i], layer_idx, :, :] @ lora_b_stacked[indices[i], layer_idx, :, :] * scale
```

实现分两步：`buffer = torch.empty((len(output_slices), x.size(0), r), dtype=torch.float32)`；`add_shrink`（`x @ A^T` → `[N, r]`，fp32 中间缓冲）；`add_expand`（`buffer @ B^T` 累加到 `y`）。这就是 punica 论文的 SGMV（segmented gather matrix-vector）思想：shrink/expand 各是一个 kernel，内部按 LoRA 分段。

#### (4) Triton kernel：按 LoRA id 排序 token

- `LoRAKernelMeta.prepare_tensors` L107–L160：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L107-L160

```python
_, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)   # 按 LoRA 槽号排序 token
lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)                       # 每个 LoRA 段的起点
```

- `_lora_shrink_kernel` L101–L217：https://github.com/vllm-project/vllm/blob/f5fefbffe51e1cc91e0d6507d272f337473989b3/vllm/lora/ops/triton_ops/lora_shrink_op.py#L101-L217

```python
slice_id = tl.program_id(axis=1)
lora_idx = tl.program_id(axis=2)                        # grid 第 3 维 = 第几个活跃 LoRA
lora_id = tl.load(lora_ids + lora_idx)
if lora_id == -1:                                       # -1 = 不用 LoRA 的 token
    return
lora_m_size = tl.load(num_tokens_per_lora + lora_idx)   # 该 LoRA 有几个 token
cta_m_offset = pid_m * BLOCK_M
if cta_m_offset >= lora_m_size:
    return
lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
ram = tl.load(cta_lora_seq_indices + offset_m)          # 本 CTA 处理的原始 token 行号（gather）
do_shrink_kernel(..., lora_id, ..., ram, ...)           # 用槽 lora_id 的 A 与这些行做 GEMM
```

把它和第 07 篇 vLLM `fused_moe_kernel` 对照：**“按专家排序 token + 每段一个 GEMM + 读 A 时 gather”**，几乎一模一样——多 LoRA 服务就是一种“每个 token 只选 1 个专家（LoRA）、专家是低秩矩阵”的 MoE。

---

## 5. 不同实现对比

| 维度 | loralib | PEFT | vLLM LoRA |
|---|---|---|---|
| 接入方式 | 改模型代码，`nn.Linear`→`lora.Linear`（继承） | 运行时 `setattr` 替换子模块（组合 `base_layer`） | 运行时 `replace_submodule`，`from_layer` 分派 |
| 多 adapter | 不支持 | `ModuleDict` 按名存多个，`set_adapter` 切换，`adapter_names` 混合 batch | `max_loras` 个预分配槽位，同一 batch 任意混合 |
| scaling | `α/r` | `α/r` 或 `α/√r`（rsLoRA），`rank_pattern/alpha_pattern` | 加载时乘进 B（`optimize()`） |
| merge | `train()/eval()` 自动 | `merge()/unmerge()/merge_and_unload()`，`safe_merge` | 不 merge（要同时服务多个） |
| 多 adapter 计算 | — | Python 循环按 adapter 分组 | Triton shrink/expand kernel，按 LoRA id 排序 token |
| 量化底座 | 无 | bnb 4/8bit、GPTQ、AWQ、HQQ、torchao… 各有 dispatcher | 支持在量化 base 上叠加（base 走 quant_method） |
| 初始化 | A Kaiming、B 0 | `True`/gaussian/pissa/olora/loftq/eva/corda/… | 只加载已训练权重 |
| 变体 | 无 | DoRA、aLoRA、Arrow、BdLoRA、… 通过 `LoraVariant` 插件 | 主要是普通 LoRA |

---

## 6. 常见困惑 / 坑

1. **“lora_A 是 r×in 还是 in×r？”** loralib 与 PEFT 都是 `A: [r, d_in]`、`B: [d_out, r]`（PEFT 用 `nn.Linear(in, r)` 存 A，其 `.weight` 就是 `[r, in]`）。论文写成 `W + BA`、`h = W x + B A x`（列向量约定）。代码里是行向量 `x @ A^T @ B^T`。
2. **loralib 注释写反了**：`Linear.reset_parameters` 的注释说 “B 用 Linear 默认、A 为 0”，代码其实是 A Kaiming、B 为 0。
3. **改了 r 却没改 α**：`α/r` 下 r 翻倍、α 不变 → 缩放减半，有效学习率变了。常见经验是 `α = 2r` 或固定 α 只调 r；rsLoRA 下的关系又不同。
4. **merge 后继续训练**：PEFT 的 `forward` 在 `merged=True` 时只走 base，LoRA 参数不会得到梯度；要训练先 `unmerge()`。loralib 在 `model.train()` 时会自动 unmerge。
5. **QLoRA merge 精度**：4bit 权重 merge 要先反量化、再加 delta、再量化，结果与 unmerged 推理不完全一致（PEFT 会发出警告）。更稳妥的是在 bf16 底座上 `merge_and_unload()` 后再整体量化。
6. **`modules_to_save` 不是 LoRA**：分类头等会整层复制并全量训练，保存时一起存——adapter 文件可能因此变大。
7. **gradient checkpointing + 冻结 embedding**：需要 `enable_input_require_grads()`，否则重入式 checkpoint 拿不到需要梯度的输入，LoRA 梯度为 None（`prepare_model_for_kbit_training` 帮你做了）。
8. **PiSSA / OLoRA / LoftQ 会修改底座权重**：分享 adapter 时，别人加载的原始底座与你训练时的残差底座不同，需要转换（`subtract_mutated_init` / `path_initial_model_for_weight_conversion`）。
9. **DoRA 推理慢**：每次 forward 都要算 `‖W + sBA‖`（且需反量化底座），务必 merge 后部署。
10. **多 LoRA 服务的 rank 上限**：vLLM 槽位按 `max_lora_rank` 分配，rank 更大的 adapter 放不进去；rank 小的会零填充，计算量仍按 `max_lora_rank` 算。

---

## 7. 自测题 + 动手练习

**自测题**

1. 一个 `d_in = d_out = 4096` 的线性层，r=16，LoRA 可训练参数是多少？占原层参数的百分之几？
2. 如果把 `B` 随机初始化、`A` 置 0，训练能进行吗？初始输出会变吗？和标准初始化相比有什么差别？
3. PEFT 中 `target_modules=["proj"]` 会不会命中 `o_proj`？会不会命中 `model.proj`？`target_modules="proj"`（字符串）呢？（提示：读 `check_target_module_exists` 中列表后缀匹配与字符串正则全匹配的区别）
4. 为什么 `_create_new_module` 的 dispatcher 顺序中 bnb 必须排在 `dispatch_default` 之前？
5. DoRA `forward` 里为什么返回 `(mag_norm_scale - 1) * base_result + ...` 而不是直接 `mag_norm_scale * (base_result + ...)`？
6. vLLM 为什么把 `α/r` 预先乘进 `lora_b`？如果一个 LoRA 用了 rsLoRA，vLLM 怎么知道？
7. PEFT `_mixed_batch_forward` 与 vLLM `_lora_shrink_kernel` 都在做“按 adapter 分组”，二者的差异在哪里（从 kernel 启动次数、是否需要 `.tolist()`/Python 循环的角度回答）？

**动手练习**

1. 运行第 3 节代码；把 `inject_lora` 改成支持正则 `target_modules`（模仿 PEFT 的 `re.fullmatch`），并实现 `merge_and_unload()`：merge 后把 `LoRALinear` 替换回 `base`。
2. 在一个小模型（例如 `repos/00-basics/nanoGPT` 的 GPT）上用你的 `LoRALinear` 只微调 `c_attn`，比较 r=1/4/16 的 loss 曲线；再开启 rsLoRA 对比。注意 nanoGPT 的 `c_attn` 是融合 qkv，可参考 loralib `MergedLinear` 只对 q、v 加 LoRA。
3. 实现 DoRA 版 `DoRALinear`：增加 `m = ‖W‖_row` 参数，forward 用 `m / ‖W + sBA‖_row.detach()` 缩放，验证初始输出与原层一致，merge 公式与 PEFT `merge_unsafe` 一致。
4. 实现 PiSSA 初始化：对 `W` 做 SVD，按 `pissa_init` 的公式设置 A/B 与残差 W，验证 `W_res + s·B·A ≈ W`。
5. 写一个多 LoRA 的参考实现：给定 `x: [N, d]`、`token_lora: [N]`（-1 表示无 LoRA）、`A: [L, r, d]`、`B: [L, d_out, r]`，先按 `token_lora` 排序再分段计算，然后与“逐 token 用 `A[token_lora[i]]`”的朴素版本对齐——这就是 vLLM `LoRAKernelMeta.prepare_tensors` + shrink/expand 的语义。
