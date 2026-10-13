# 11 · 模型合并（Model Merging）源码导读

> 对应源码：`repos/10-model-merge/mergekit`（arcee-ai/mergekit @ `9eeb5398`）
>
> 注意：这个 commit 的 mergekit 是**较新的重构版本**（`TensorGroup` / `TensorBatch` / `GroupMergeMethod` / `@merge_method` 装饰器），和网上很多老教程里的 `MergeMethod.make_task()` 写法不同。本文以仓库实际代码为准，行号与 permalink 都指向该 commit。

Permalink 前缀：`https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/`

> 阅读约定：引用的源码片段为便于阅读，会用 `...` 省略无关行、把跨多行的函数参数合并为一行，偶尔在行尾加中文注释；一切以 permalink 指向的原文为准。

---

## 0. 要解决的问题 / 一句话直觉

**问题**：你手上有同一个底座（如 Llama-2-13B）微调出来的多个模型：一个会写代码、一个会做数学、一个对话好。能不能**不训练**、只做权重的算术，就得到一个"都会"的模型？

**一句话直觉**：

- 同底座微调出的模型，权重都在底座附近一个很小的邻域里（微调改动小），而且损失面在这个邻域里近似"连通、凸"。所以权重**平均**通常仍是好模型（Model Soup）。
- 把"微调学到的东西"看成一个向量：**任务向量 τ = θ_ft − θ_base**。能力可以相加（`θ_base + τ_code + τ_math`）、可以相减（`θ − τ_toxic` 去毒）——这就是 **Task Arithmetic**。
- 直接相加会互相干扰：τ 里大量是噪声小值（冗余），不同任务在同一参数上方向相反（符号冲突）。**TIES** = 剪掉小值（Trim）+ 每个参数投票选符号（Elect）+ 只平均同号的（Disjoint merge）。**DARE** = 随机丢掉 90% 的 τ 再放大补偿，发现几乎不掉点，说明 τ 极度冗余。
- **SLERP**：两个模型之间沿"球面"插值，而不是直线插值，保持权重范数不塌缩。
- **Passthrough / Frankenmerge**：不做算术，直接把 A 的第 0–23 层和 B 的第 20–39 层拼起来，做出一个更深的模型。
- **MoE 合并（mergekit-moe）**：把几个稠密模型的 MLP 当作专家，attention 共用底座，用提示词的隐状态构造路由器。

---

## 1. 核心原理（带张量形状）

对**每一个权重张量名**（如 `model.layers.7.mlp.up_proj.weight`，形状 `(I, d)`）独立做合并。设 K 个微调模型 `θ_k`，底座 `θ_0`，形状都相同 `S`。

### 1.1 Linear / Model Soup

```
θ = Σ_k w_k θ_k / Σ_k w_k            （normalize=True 时）
```

不需要 base。均匀权重即 "uniform soup"。

### 1.2 Task Arithmetic

```
τ_k = θ_k − θ_0                     形状 S
θ = θ_0 + λ · Σ_k w_k τ_k
```

### 1.3 TIES-Merging（Trim, Elect Sign, Disjoint Merge）

1. **Trim**：每个 `τ_k` 只保留绝对值最大的 `density` 比例（如 20%），其余置 0 → `τ̂_k`。
2. **Elect**：对每个参数位置 p，选主导符号
   `γ_p = sign( Σ_k τ̂_k[p] )`（mergekit `sum` 模式：按带权和的符号，等价于"按总质量投票"）
3. **Disjoint Merge**：只平均与 `γ_p` 同号的那些 τ：
   `τ_m[p] = Σ_{k∈A_p} w_k τ̂_k[p] / Σ_{k∈A_p} w_k`，`A_p = {k : sign(τ̂_k[p]) = γ_p}`
4. `θ = θ_0 + λ τ_m`

形状：把 K 个 τ 堆成 `(K, *S)`，mask 也是 `(K, *S)`，沿 dim=0 求和。

### 1.4 DARE（Drop And REscale）

```
m ~ Bernoulli(p)           p = density（保留概率），形状 S
τ̃ = τ ⊙ m / p             （原论文：除以 p 保持期望不变）
```

然后可以接 Task Arithmetic（`dare_linear`）或 TIES 的符号选举（`dare_ties`）。mergekit 的 rescale 实现是"**让 L1 范数与丢弃前相等**"，期望意义下与 `1/p` 等价（见 3.4）。

### 1.5 Model Breadcrumbs 与 DELLA

- **Breadcrumbs**：除了剪掉最小的值，还剪掉**最大的 γ 比例**（离群大改动，可能是过拟合），保留中间部分。
- **DELLA (MagPrune)**：DARE 的改进——丢弃概率不是均匀的，**幅值越大保留概率越高**：按行排名归一化到 `[0,1]`，保留概率 `p_i ∈ [density−ε, density+ε]` 线性分布。

### 1.6 SLERP（球面线性插值）

把两个张量拉平成向量 `a, b`，

```
cos Ω = (a/‖a‖)·(b/‖b‖)
slerp(t) = sin((1−t)Ω)/sin Ω · a + sin(tΩ)/sin Ω · b
```

当 `|cos Ω|` 接近 1（几乎共线）时退化为线性插值 `(1−t)a + t b`。只支持**两个**模型。`t` 可以随层变化（"gradient"），例如 attention 前几层偏 A、后几层偏 B。

### 1.7 Passthrough / Frankenmerge

输出模型第 j 层 = 某个输入模型第 `layer_range[0] + offset` 层的原样拷贝。可以产生比原模型**更深**的模型（如两个 40 层 13B 拼成 44 层）。

### 1.8 MoE 合并

输出 Mixtral 结构：每层 `E` 个专家，专家 e 的 `w1/w2/w3` = 第 e 个源模型该层的 `gate/down/up_proj`；attention、norm、embedding 取自 base。路由器权重 `(E, d)`：第 e 行 = 专家 e 的正向提示在该层的平均隐状态（减去负向提示），再 L2 归一化。

---

## 2. 最小实现（自己写的 PyTorch 片段）

### 2.1 Task Arithmetic / TIES / DARE 合并 3 个 state dict

```python
import torch

def trim_topk(t: torch.Tensor, density: float) -> torch.Tensor:
    """保留绝对值最大的 density 比例（整张量全局排序）。"""
    if density >= 1: return t
    k = max(1, int(density * t.numel()))
    thr = t.abs().flatten().kthvalue(t.numel() - k + 1).values     # 第 k 大的绝对值
    return t * (t.abs() >= thr)

def dare(t: torch.Tensor, density: float) -> torch.Tensor:
    m = torch.bernoulli(torch.full_like(t, density))
    return t * m / density                                         # 论文版 rescale

@torch.no_grad()
def merge(base: dict, models: list, weights, method="ties", density=0.2, lam=1.0):
    out = {}
    w = torch.tensor(weights, dtype=torch.float32)                  # (K,)
    for name, b in base.items():
        if not b.is_floating_point():
            out[name] = b.clone(); continue
        b32 = b.float()
        taus = torch.stack([m[name].float() - b32 for m in models])   # (K, *S)
        if method in ("ties", "dare_ties"):
            sp = trim_topk if method == "ties" else dare
            taus = torch.stack([sp(t, density) for t in taus])
        if method == "dare_linear":
            taus = torch.stack([dare(t, density) for t in taus])
        wv = w.view(-1, *([1] * b.dim()))                            # (K,1,1,...)
        taus = taus * wv                                             # 先乘权重
        if method in ("ties", "dare_ties"):
            elected = torch.where(taus.sum(0) >= 0, 1.0, -1.0)       # (*S,) 主导符号
            agree = (taus.sign() == elected).float()                 # (K,*S)，0 值不算同意
            num = (taus * agree).sum(0)
            den = (wv * agree).sum(0).clamp_min(1e-8)                # 只数同号模型的权重
            merged_tau = torch.where(agree.sum(0) > 0, num / den, torch.zeros_like(num))
        else:                                                        # task arithmetic / dare_linear
            merged_tau = taus.sum(0)
        out[name] = (b32 + lam * merged_tau).to(b.dtype)
    return out

# 玩具验证
torch.manual_seed(0)
base = {"w": torch.randn(8, 8), "step": torch.tensor(3)}
fts = [{"w": base["w"] + 0.1 * torch.randn(8, 8), "step": torch.tensor(3)} for _ in range(3)]
merged = merge(base, fts, weights=[1, 1, 1], method="ties", density=0.3)
print((merged["w"] - base["w"]).abs().mean())
```

注意 TIES 分支除以的是"同号模型的权重和"（disjoint mean），而 task arithmetic 分支不除（直接求和）——这与 mergekit 中 `ties` 默认 `normalize=True`、`task_arithmetic` 默认 `normalize=False` 一致（见 3.2）。

### 2.2 SLERP

```python
def slerp(t: float, a: torch.Tensor, b: torch.Tensor, dot_thr=0.9995, eps=1e-8):
    va, vb = a.flatten().float(), b.flatten().float()
    na, nb = va / (va.norm() + eps), vb / (vb.norm() + eps)
    dot = (na * nb).sum().clamp(-1, 1)
    if dot.abs() > dot_thr:                       # 几乎共线：退化为 lerp
        return ((1 - t) * a.float() + t * b.float()).to(a.dtype)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    out = torch.sin((1 - t) * omega) / so * va + torch.sin(t * omega) / so * vb
    return out.view_as(a).to(a.dtype)
```

注意：系数作用在**原始未归一化**的 `va, vb` 上（与 mergekit 相同），归一化只用来算夹角。

---

## 3. 源码导读（按阅读顺序）

整体数据流：

```
YAML ──pydantic──▶ MergeConfiguration ──MergePlanner──▶ 每个输出张量一个 ExecuteMergeMethodTask
                                                         │  (依赖 GatherTensors → LoadTensor)
                                     Executor 拓扑排序 ◀──┘
                                           │
                     execute: 构造 TensorGroup ──▶ method._execute_resolved ──▶ merge_group / merge_batch
                                           │
                                     SaveTensor → TensorWriter(分片写 safetensors)
```

### 3.1 入口：从命令行到 `run_merge`

- 命令行脚本在 [pyproject.toml#L42-L47](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/pyproject.toml#L42-L47)：`mergekit-yaml = "mergekit.scripts.run_yaml:main"`，`mergekit-moe = "mergekit.scripts.moe:main"`。
- `run_yaml.main` 用 `MergeConfiguration.model_validate(yaml.safe_load(...))` 解析 YAML（[scripts/run_yaml.py#L27](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/scripts/run_yaml.py#L27)），然后调用 `run_merge`。

[mergekit/merge.py#L30-L84](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge.py#L30-L84)

```python
arch_info = get_architecture_info(merge_config, options)
loader_cache = LoaderCache()
...
targets = MergePlanner(
    merge_config, arch_info, options=options, out_model_config=cfg_out,
).plan_to_disk(out_path=out_path)
...
    exec = Executor(
        targets=targets,
        math_device=options.device,
        storage_device=options.device if options.low_cpu_memory else "cpu",
    )
for _task, value in exec.run(quiet=options.quiet):
    ...
```

三步：**识别架构**（知道一个 Llama 有哪些权重名）→ **规划**（生成任务图）→ **执行**（按拓扑序跑，`math_device` 上算、`storage_device` 上存）。

### 3.2 YAML 长什么样：`config.py`

[mergekit/config.py#L76-L120](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/config.py#L76-L120)

```python
class InputSliceDefinition(BaseModel):
    model: ModelReference
    layer_range: Tuple[int, int]
    parameters: Optional[Dict[str, ParameterSetting]] = None

class OutputSliceDefinition(BaseModel):
    sources: List[InputSliceDefinition]
    base_model: Optional[ModelReference] = None
    ...
class MergeConfiguration(BaseModel):
    modules: Optional[Dict[str, OutputModuleDefinition]] = None
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None
    merge_method: str
    base_model: Optional[ModelReference] = None
    dtype: Optional[str] = None
    ...
    parameters: Optional[Dict[str, ParameterSetting]] = None
```

三种写法：`models:`（整模型合并）、`slices:`（按层段，frankenmerge 必用）、`modules:`（多模块模型，如视觉+语言）。规划时都被归一化成 `modules → slices`（见 3.3）。

看一个真实例子 [examples/ties.yml](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/examples/ties.yml#L1-L22)：

```yaml
models:
  - model: psmathur/orca_mini_v3_13b
    parameters:
      density: [1, 0.7, 0.1] # density gradient
      weight: 1.0
  - model: garage-bAInd/Platypus2-13B
    parameters:
      density: 0.5
      weight: [0, 0.3, 0.7, 1] # weight gradient
  - model: WizardLM/WizardMath-13B-V1.0
    parameters:
      density: 0.33
      weight:
        - filter: mlp
          value: 0.5
        - value: 0
merge_method: ties
base_model: TheBloke/Llama-2-13B-fp16
parameters:
  normalize: true
  int8_mask: true
```

参数可以是：标量；**列表 = 随层插值的梯度**；`filter` 条件列表 = 按张量名子串选值。这些由 `evaluate_setting` 解析：

[config.py#L27-L73](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/config.py#L27-L73)

```python
if isinstance(setting, (float, int, bool, str)):
    return validate(setting)
elif isinstance(setting, list):
    ...
    if all(isinstance(e, (float, int, bool, str)) for e in setting):
        values = [validate(e) for e in setting]
        ...
        scaled = t * (len(setting) - 1)
        i0 = int(scaled)
        i1 = min(len(setting) - 1, i0 + 1)
        frac = scaled - i0
        result = (1 - frac) * values[i0] + frac * values[i1]
        return validate(result)
    else:
        for cond in setting:
            if (cond.filter is None) or (cond.filter == "*") or (tensor_name and cond.filter in tensor_name):
                res = evaluate_setting(tensor_name, cond.value, t, validate=validate)
                return res
```

- `t ∈ [0,1]` 是"当前层在 slice 中的相对位置"（3.3 中 `t = idx/(num_layers-1)`）。`[0, 0.3, 0.7, 1]` 被当作 4 个等距锚点，层间线性插值。
- `filter` 是**子串匹配**（`cond.filter in tensor_name`），第一个命中的生效，所以兜底项（无 filter）要放最后。

参数优先级：`ConfigReader.parameter_sources`（[config.py#L194-L206](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/config.py#L194-L206)）依次 yield 某模型在 slice 里的参数 → slice 参数 → module 参数 → 全局 `parameters`；`resolve_parameter`（[parameter_resolver.py#L19-L39](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/parameter_resolver.py#L19-L39)）取**第一个非 None** 的值，否则用 `ParameterSpec.default`，必填参数缺失就报错。

### 3.3 规划：YAML → 每个张量一个任务（`plan.py`）

**(a) 选方法** —— [plan.py#L61](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L61)：`self._method = merge_methods.get(config.merge_method)`，即从注册表里按名字取（见 3.5）。

**(b) 归一化配置** —— [plan.py#L93-L170](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L93-L170)

```python
# models -> modules.models
if self.config.models:
    ...
# slices -> modules.slices
...
# modules.models -> modules.slices
            num_layers = module_arch.num_layers(model_cfg)
            slices_in.append(InputSliceDefinition(layer_range=[0, num_layers], model=model_in.model, parameters=model_in.parameters))
        if base_model and not base_included:
            logging.info("Base model specified but not in input models - adding")
            ...
        module_out.slices = [OutputSliceDefinition(sources=slices_in)]
```

`models:` 写法 = "一个 slice，每个模型都取全部层 `[0, num_layers]`"。注意 base 不在 `models` 里时会被**自动加入**为一个输入。最后 [L163-L170](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L163-L170) 先用 `validate_inputs` 检查每个 slice 的输入数量/是否有 base 是否满足方法的 `InputContract`（比如 SLERP 必须恰好 2 个输入且含 base），在加载任何权重之前就失败。

**(c) 逐 slice、逐层** —— [plan.py#L275-L306](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L275-L306)

```python
slice_lengths = [s.layer_range[1] - s.layer_range[0] for s in definition.sources]
if not all(s == slice_lengths[0] for s in slice_lengths):
    raise RuntimeError("All inputs to a slice must contain the same number of layers")
...
for idx in range(num_layers):
    if num_layers > 1:
        t = idx / (num_layers - 1)
    else:
        t = 1
    self.plan_layer(definition.sources, layer_offset=idx, t=t, cfg_reader=cfg_reader, module_name=module_name)
```

`t` 就是 3.2 里梯度插值用的相对位置。

**(d) 一层 → 多个张量** —— [plan.py#L246-L273](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L246-L273)

```python
weights_out: List[WeightInfo] = module_arch.layer_weights(index=self._current_module_layers)
weights_in: List[List[WeightInfo]] = [
    self._model_arch(s.model).get_module(module_name).layer_weights(index=s.layer_range[0] + layer_offset)
    for s in sources
]
for idx, w_o in enumerate(weights_out):
    self.plan_tensor(weight=w_o, weights_in=[weights_in[j][idx] for j in range(len(weights_in))], ...)
self._current_module_layers += 1
```

**这是 frankenmerge 的关键**：输出层号用的是递增计数器 `_current_module_layers`，输入层号是 `layer_range[0] + offset`。所以 slice1 = A 的 `[0,24)`、slice2 = B 的 `[20,40)` 会产生输出层 0..43，其中输出层 24 来自 B 的第 20 层——权重名被**重新编号**。

**(e) 单个张量的任务** —— [plan.py#L172-L244](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L172-L244)

```python
global_params, tensor_params = resolve_parameters(
    tensor_merge_method.spec,
    tensor_name=weight.name,
    inputs={model: w.name for model, w in zip(models, weights_in)},
    sources=cfg_reader.parameter_sources,
    base_model=base_model,
    t=cfg_reader.t,
)
gather_tensors = GatherTensors(weight_info=ImmutableMap(data=dict(zip(models, weights_in))), dtype=self.config.dtype, ...)
...
tensor_task = ExecuteMergeMethodTask(
    method_name=tensor_merge_method.spec.name,
    gather_tensors=tensor_input_task,
    model_order=tuple(models),
    parameters=global_params,
    input_parameters=tensor_params,
    base_model=base_model,
    output_weight=weight,
)
self._tensors.append((weight, tensor_task))
```

- **参数在规划期就解析成具体数值**（`global_params`：SHARED 参数，如 `normalize/lambda/t`；`tensor_params[model]`：每个输入一份，如 `weight/density`）。执行期不再看 YAML。
- 若涉及词表维度（embedding/lm_head）且配置了 tokenizer，会插入 `PermutedVocabulary` 任务把不同模型的词表对齐（L213-L226）。
- `plan_module`（[L308-L335](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L308-L335)）在所有层之前/之后分别规划 `pre_weights`（embedding）和 `post_weights`（final norm、lm_head）。
- `plan_to_disk`（[L337-L367](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/plan.py#L337-L367)）给每个张量任务包一层 `SaveTensor`，再加一个 `FinalizeModel`，返回目标任务列表。

`resolve_parameters` 注意一点（[parameter_resolver.py#L56-L65](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/parameter_resolver.py#L56-L65)）：`NON_BASE` 作用域的参数（如 TIES 的 `weight/density`）**不会**为 base 模型解析，所以 base 不需要写 `weight`。

### 3.4 任务图与执行器（`graph.py`）

**(a) `Task`** —— [graph.py#L23-L68](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/graph.py#L23-L68)：一个冻结的 pydantic 模型，`arguments()` 返回依赖 `{参数名: Task}`，`execute(**kwargs)` 接收依赖的结果。因为是 frozen 且可哈希，**相同的任务自动去重**（比如两个输出张量依赖同一个 `LoadTensor`）。

`GatherTensors`（[io/tasks.py#L144-L175](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/io/tasks.py#L144-L175)）的依赖是每个模型一个 `LoadTensor`，执行结果是 `{ModelReference: tensor}`。

**(b) 调度** —— [graph.py#L305-L382](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/graph.py#L305-L382)

```python
for dep in task.arguments().values():
    to_explore.add(dep)
    edge_tups.append((dep, task))
...
def _compare_key(node: TaskHandle) -> Tuple[str, int]:
    ...
    return (task.group_label() or "", -task.priority())
graph = networkx.DiGraph(edge_tups)
schedule = [node for node in networkx.lexicographical_topological_sort(graph, key=_compare_key) ...]
# Calculate last use indices for memory optimization
for idx, task in reversed(list(enumerate(schedule))):
    for dep in task.arguments().values():
        if dep not in last_use_index:
            last_use_index[dep] = idx
```

- 拓扑排序时用 `group_label`（通常是 shard 文件名）做字典序，**让来自同一个 safetensors 分片的张量挨着处理**，减少反复打开文件。
- `last_use_index`：每个中间结果最后一次被用到的位置；执行器在 [graph.py#L494-L501](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/graph.py#L494-L501) 一过这个位置就 `del`。这就是 mergekit 能在**很小内存**下合并 70B 模型的原因：同一时刻只驻留少数几个张量。

**(c) 执行循环** —— [graph.py#L442-L504](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/graph.py#L442-L504)：取依赖结果 → 若 `task.uses_accelerator()` 则搬到 `math_device`（GPU）→ `task.execute(**arguments)` → 结果搬回 `storage_device`（CPU）→ 淘汰过期值。

### 3.5 方法注册表（`merge_methods/registry.py`）

[registry.py#L46-L133](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/registry.py#L46-L133)

```python
GeneralizedTaskArithmeticMerge(
    consensus_method=None, sparsification_method=None,
    default_normalize=False, default_rescale=False,
    method_name="task_arithmetic", ...),
GeneralizedTaskArithmeticMerge(
    consensus_method=ConsensusMethod.sum, sparsification_method=SparsificationMethod.magnitude,
    default_normalize=True, default_rescale=False,
    method_name="ties", ...),
GeneralizedTaskArithmeticMerge(
    consensus_method=ConsensusMethod.sum, sparsification_method=SparsificationMethod.random,
    default_normalize=False, default_rescale=True,
    method_name="dare_ties", ...),
...  # dare_linear / breadcrumbs / breadcrumbs_ties / della / della_linear
```

**这张表是理解 mergekit 的钥匙**：task_arithmetic、TIES、DARE、Breadcrumbs、DELLA 全是**同一个类**的不同配置，只在两根轴上变化：

| method_name | sparsification（怎么稀疏化 τ） | consensus（是否选符号） | 默认 normalize | 默认 rescale |
|---|---|---|---|---|
| `task_arithmetic` | 无 | 无 | False | False |
| `ties` | `magnitude` | `sum` | True | False |
| `dare_linear` | `random` | 无 | False | True |
| `dare_ties` | `random` | `sum` | False | True |
| `breadcrumbs` | `magnitude_outliers` | 无 | False | False |
| `breadcrumbs_ties` | `magnitude_outliers` | `sum` | False | False |
| `della` | `della_magprune` | `sum` | True | True |
| `della_linear` | `della_magprune` | 无 | False | True |

其它独立方法：`linear`、`slerp`、`nuslerp`、`multislerp`、`karcher`、`model_stock`、`passthrough`、`nearswap`、`sce`、`arcee_fusion`、`ram`（L46-L58）。

### 3.6 通用任务算术：`GeneralizedTaskArithmeticMerge.merge_group`

**(a) 参数声明** —— [generalized_task_arithmetic.py#L39-L89](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/generalized_task_arithmetic.py#L39-L89)

`int8_mask / normalize / rescale / lambda` 是 SHARED（全局一份）；`weight`（**无默认值 = 必填**）、`density`（默认 1.0）是 NON_BASE（每个非 base 模型一份）；breadcrumbs 额外有 `gamma=0.01`，della 额外有 `epsilon=0.15`。`contract=InputContract(base=BasePolicy.REQUIRED)`：这一族方法都必须有 base。

**(b) 合并主体** —— [generalized_task_arithmetic.py#L91-L145](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/generalized_task_arithmetic.py#L91-L145)

```python
base = group.base.tensor
entries = group.non_base
...
deltas = base.new_empty((len(entries), *base.shape))        # (K, *S)
for index, entry in enumerate(entries):
    delta = entry.tensor - base                             # 任务向量 τ_k
    if self.sparsification_method:
        delta = sparsify(
            delta,
            density=parameters["density"][entry.id],
            method=self.sparsification_method,
            rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
            **{key: parameters[key][entry.id] for key in ("gamma", "epsilon") if key in parameters},
        )
    deltas[index] = delta
weights = torch.tensor([parameters["weight"][entry.id] for entry in entries], ...)   # (K,)
while deltas.dim() > weights.dim():
    weights.unsqueeze_(-1)                                   # (K,1,1,...) 便于广播
deltas.mul_(weights)
if self.consensus_method:
    mask_dtype = torch.int8 if parameters["int8_mask"] else base.dtype
    mask = get_mask(deltas, method=self.consensus_method, mask_dtype=mask_dtype)
    mixed_delta = (deltas * mask).sum(dim=0)                 # 只加同号的
    divisor = (weights * mask).sum(dim=0)                    # 同号模型的权重和
    divisor[divisor == 0] = 1
else:
    mixed_delta = deltas.sum(dim=0)
    divisor = weights.sum(dim=0)
    divisor[divisor.abs() < 1e-8] = 1
if parameters["normalize"]:
    mixed_delta /= divisor
if parameters["lambda"] != 1:
    mixed_delta *= parameters["lambda"]
return (base + mixed_delta).to(base.dtype)
```

逐行要点：
- `parameters["density"][entry.id]`：per-input 参数是一个以模型为键的 `PerInputValues` 映射。
- **顺序**：先稀疏化每个 τ（Trim / Drop），再乘权重，再选符号。所以符号选举是"按加权后的质量"投票。
- `divisor[divisor == 0] = 1`：某位置所有模型都被剪成 0 时避免除零（此时 `mixed_delta` 本来也是 0）。
- `normalize=True` 时 TIES 得到的是"同号者的加权平均"（论文的 disjoint mean）；`normalize=False` 时是加权和（task arithmetic 语义）。
- `lambda`：论文中的整体缩放系数（通常 0.5–1.5 调参）。
- `int8_mask`：mask 用 int8 存，省内存（`(K,*S)` 的 mask 对大张量不小）。

**(c) 符号选举** —— [generalized_task_arithmetic.py#L148-L172](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/generalized_task_arithmetic.py#L148-L172)

```python
sign = delta.sign().to(mask_dtype)                          # (K,*S) ∈ {-1,0,1}
if method == "sum":
    sign_weight = delta.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
elif method == "count":
    majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
return sign == majority_sign
```

- `sum`（TIES 论文）：按 τ 的**总和**的符号；大幅值的模型话语权更大。`count`：一模型一票。
- `(x >= 0)*2 - 1`：把 bool 映射到 ±1，**平票/全 0 时取 +1**。
- 返回 `sign == majority_sign`：被剪成 0 的元素 `sign=0`，永远不等于 ±1，所以自动被排除在 disjoint mean 之外——这就是为什么分母只数"非零且同号"的模型。

### 3.7 稀疏化工具：`mergekit/sparsify.py`

**(a) 重缩放** —— [sparsify.py#L23-L53](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/sparsify.py#L23-L53)

```python
masked = tensor * mask
...
elif norm == RescaleNorm.l1:
    before_scale = tensor.abs().sum()
    after_scale = masked.abs().sum()
...
return masked * (before_scale / after_scale)
```

**与 DARE 论文的差别**：论文除以保留概率 `p`；这里让丢弃后的 L1 范数等于丢弃前。对随机丢弃，`E[after] = p · before`，所以两者期望一致；但对 magnitude 剪枝（若开 rescale），L1 匹配会给出远小于 `1/p` 的放大（因为保留的都是大值）。

**(b) TIES 的 Trim：`magnitude`** —— [sparsify.py#L56-L74](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/sparsify.py#L56-L74)

```python
k = int(density * tensor.numel())
assert k > 0, "not gonna zero out the whole tensor buddy"
mask = torch.zeros(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
w = tensor.abs().reshape(-1)
...
topk = torch.argsort(w, descending=True)[:k]
mask[topk] = 1
```

整张量（而非逐行）全局取 top-k。

**(c) Breadcrumbs：`magnitude_outliers`** —— [sparsify.py#L77-L111](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/sparsify.py#L77-L111)

```python
target_n = int(density * num_elems)
n_top = int(gamma * num_elems)
n_bot = max(0, num_elems - target_n - n_top)
indices = torch.sort(w, descending=False).indices
mask[indices[n_bot : n_bot + target_n]] = 1
```

按幅值升序排，跳过最小的 `n_bot` 个，保留接下来的 `target_n` 个，于是最大的约 `gamma` 比例也被扔掉。

**(d) DARE：`bernoulli`** —— [sparsify.py#L114-L130](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/sparsify.py#L114-L130)：`mask = torch.bernoulli(torch.full_like(tensor, density))`，CPU 上 fp16 不支持 bernoulli 所以先转 fp32。**有随机性**，可重复实验需设 `--random-seed`（`run_merge` 开头 [merge.py#L36-L37](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge.py#L36-L37)）。

**(e) DELLA：`della_magprune`** —— [sparsify.py#L133-L170](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/sparsify.py#L133-L170)

```python
if len(tensor.shape) < 2:
    tensor = tensor.unsqueeze(0)
magnitudes = tensor.abs()
sorted_indices = torch.argsort(magnitudes, dim=1, descending=False)
ranks = sorted_indices.argsort(dim=1).to(work_dtype) + 1          # 每行内的名次 1..n
min_ranks = ranks.min(dim=1, keepdim=True).values
max_ranks = ranks.max(dim=1, keepdim=True).values
rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
probs = (density - epsilon) + rank_norm * 2 * epsilon               # ∈ [d-ε, d+ε]
mask = torch.bernoulli(probs).to(work_dtype)
```

- `argsort().argsort()` 是求"名次"的经典技巧。
- **逐行**（dim=1）排名，与 magnitude 的全局排序不同。
- 要求 `density ± epsilon ∈ (0,1)`（L145-L148），否则报错。

### 3.8 Linear 与 SLERP：批处理内核

这两个方法用 `@merge_method` 装饰器定义，第一个参数注解为 `TensorBatch` → 走 `BatchedMergeMethod`（[easy_define.py#L97-L136](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/easy_define.py#L97-L136) 根据第一个参数的类型注解选择 `BatchedMergeMethod` 或 `FunctionalGroupMergeMethod`，并**从函数签名自动推导参数 spec**）。

`TensorBatch` 的约定（[base.py#L326-L353](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/base.py#L326-L353)）：每个输入形状 `[output, *weight_shape]`，**第 0 维是"多个同形状输出张量打包在一起"的批维**（`BatchedMergeMethod._execute` 在 [base.py#L589-L612](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/base.py#L589-L612) 打包、执行、再 `unbind(0)` 拆开）。所以内核里每个系数都要 reshape 成 `(B, 1, 1, ...)`。

**Linear** —— [linear.py#L15-L38](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/linear.py#L15-L38)

```python
def linear_merge(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: bool = True,
) -> torch.Tensor:
    first = batch.tensors[0]
    ...
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    weight = weight.to(dtype)                                   # (B, K)
    coefficient_shape = (first.shape[0],) + (1,) * (first.ndim - 1)
    if normalize:
        denominator = weight.sum(dim=1).reshape(coefficient_shape)
    result = torch.zeros_like(first, dtype=dtype)
    for tensor, coefficient in zip(batch.tensors, weight.unbind(1)):
        result.addcmul_(tensor, coefficient.reshape(coefficient_shape))
    if normalize:
        result.div_(denominator)
    return result.to(first.dtype)
```

- `weight` 的 scope 是 `INPUT`（**包括 base 在内**每个输入都有），打包后形状 `(B, K)`。
- 用 fp32 累加再转回原 dtype，避免 bf16 累加误差。
- 不需要 base（Model Soup）。

**SLERP** —— [slerp.py#L20-L78](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/slerp.py#L20-L78)

```python
a = v0.reshape(v0.shape[0], -1)                     # (B, numel)
b = v1.reshape(v1.shape[0], -1)
t = t.reshape(-1, 1).to(dtype)                      # (B, 1)
...  # 分块计算 norm_a, norm_b（避免一次性升 fp32 占满内存）
dot = torch.stack(dot_parts).sum(dim=0).clamp(-1, 1)        # (B, 1) 余弦
linear = dot.abs() > dot_threshold
theta = torch.acos(torch.where(linear, 0, dot))
sin_theta = torch.sin(theta)
coef_a = torch.where(linear, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
coef_b = torch.where(linear, t, torch.sin(t * theta) / sin_theta)
...
    result[:, start : start + chunk_size] = coef_a * x + coef_b * y
```

- 夹角是在**整个张量拉平后**算的（每个输出张量一个 θ），不是逐行。
- `dot_threshold=0.9995`：共线时 `sin θ≈0`，退化为 lerp；`torch.where(linear, 0, dot)` 让 `acos` 在不选的分支也不产生 NaN 梯度。
- 分块（`_CHUNK_ELEMENTS = 1024*1024`）只为了限制临时 fp32 显存。

SLERP 的约束 [slerp.py#L81-L101](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/slerp.py#L81-L101)：`InputContract(base=BasePolicy.REQUIRED, min_inputs=2, max_inputs=2)`，`t=0` 返回 base、`t=1` 返回另一个模型（`slerp(t, batch.tensors[base_index], batch.tensors[1 - base_index])`）。配合 [examples/gradient-slerp.yml](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/examples/gradient-slerp.yml#L1-L20)：

```yaml
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5 # fallback for rest of tensors
```

attention 从浅到深由 base 渐变到另一个模型，MLP 反过来。

### 3.9 Passthrough 与 frankenmerge

[passthrough.py#L16-L27](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/passthrough.py#L16-L27)

```python
@merge_method(name="passthrough", pretty_name="Passthrough", uses_accelerator=False,
              contract=InputContract(max_inputs=1))
def passthrough_merge_method(group: TensorGroup, scale: PerInput[Optional[float]] = None) -> torch.Tensor:
    entry = group.entries[0]
    value = scale[entry.id]
    return entry.tensor if value is None else entry.tensor * value
```

方法本身几乎什么都不做：`max_inputs=1`（每个 slice 只能有一个来源），可选 `scale` 缩放；`uses_accelerator=False` 不必搬到 GPU。**真正的"魔法"在规划器的层号重映射**（3.3(d)）。例子 [examples/orcamini-platy-44layer.yml](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/examples/orcamini-platy-44layer.yml#L1-L9)：

```yaml
slices:
  - sources:
    - model: psmathur/orca_mini_v3_13b
      layer_range: [0, 24]
  - sources:
    - model: garage-bAInd/Platypus2-13B
      layer_range: [20, 40]
merge_method: passthrough
```

24 + 20 = 44 层（原模型 40 层）；B 的 20–23 层与 A 的 20–23 层功能"重叠"，这是 frankenmerge 的常见做法（重叠衔接比硬切更平滑）。输出 config 的层数由 `_model_out_config` 把各 slice 的 `layer_range` 长度求和后写入 `num_hidden_layers` 等配置键（[merge.py#L302-L338](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge.py#L302-L338)）。

### 3.10 执行适配：`ExecuteMergeMethodTask.execute`

[task_adapter.py#L89-L155](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/task_adapter.py#L89-L155)

```python
entries = tuple(
    TensorEntry(id=model, tensor=tensor, is_base=model == self.base_model)
    for model in self.model_order
    if (tensor := tensors.get(model)) is not None
)
group = TensorGroup(entries=entries, metadata=TensorMetadata.from_weight_info(self.output_weight))
...
buffer = copy_non_floating_buffer(group.tensors, self.output_weight.name)
if buffer is not None:
    return buffer
group.validate_tensors()
parameters = dict(self.parameters.items())
for parameter in method.spec.input_parameters:
    inputs = entries if parameter.scope == ParameterScope.INPUT else group.non_base
    parameters[parameter.name] = PerInputValues(
        [(entry.id, self.input_parameters[entry.id][parameter.name]) for entry in inputs]
    )
(result,) = method._execute_resolved((group,), [parameters], dtype=..., out_dtype=...)
```

- 把 `{model: tensor}` 字典包装成 `TensorGroup`（[base.py#L227-L307](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/base.py#L227-L307)），标记哪个是 base；`validate_tensors` 检查所有输入形状一致（词表大小不同的 embedding 在这里报错，并提示配置 tokenizer）。
- 非浮点 buffer（如整数的 position ids）不合并，只在所有输入一致时原样拷贝。
- 可选权重缺失（L104-L124）按方法的 `optional_tensor_policy` 处理：SLERP 是 `PASSTHROUGH_SINGLETON`（只有一个模型有就直接用它）。
- `GroupMergeMethod._execute`（[base.py#L615-L655](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/base.py#L615-L655)）先 `align_dtype` 统一 dtype、关掉 autocast，再调 `merge_group`，最后校验输出形状必须与输入一致。

> 想在 Python 里直接合并内存中的 state dict（不走 YAML）？新版提供了 `mergekit.merge_methods.merge_state_dicts`（[api.py#L53-L155](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/api.py#L53-L155)），内部同样把每个 key 变成一个 `TensorGroup` 再调用 `method(groups, parameters=...)`。例如 `merge_state_dicts({"base": sd0, "a": sd1, "b": sd2}, "ties", base="base", parameters={"weight": {"a": 1.0, "b": 1.0}, "density": 0.2})`（per-input 参数可传以 id 为键的映射或标量广播，见 [base.py#L537-L564](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/merge_methods/base.py#L537-L564)）。

### 3.11 MoE 合并：`mergekit-moe`

**(a) 配置** —— [moe/config.py#L12-L42](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/config.py#L12-L42)

```python
class Expert(BaseModel):
    source_model: ModelReference
    positive_prompts: Optional[List[str]] = None
    negative_prompts: Optional[List[str]] = None
    noise_scale: Optional[float] = None
    residual_scale: Optional[float] = None

class MoEMergeConfig(BaseModel):
    base_model: ModelReference
    experts: List[Expert]
    gate_mode: str = (
        "hidden"  # possible values: "hidden", "cheap_embed", "random", "uniform_random"
    )
    ...
    experts_per_token: int = 2
    shared_experts: Optional[List[Expert]] = None
```

`is_bad_config`（L45-L86）会拒绝"所有专家提示词相同"或"所有专家同一个模型"的配置——那样路由毫无意义。

**(b) 路由器初始化** —— [moe/router.py#L71-L151](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/router.py#L71-L151)

```python
for expert in tqdm.tqdm(experts, desc="expert prompts"):
    hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))   # (num_layers, hidden)
    if expert.negative_prompts:
        hidden_states -= _do_it(tokenize_prompts(expert.negative_prompts, tokenizer))
    hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    gate_vecs.append(hidden_states)
gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
return gate_vecs.permute(1, 0, 2)          # (num_layer, num_expert, hidden_size)
```

`get_hidden_states`（[L18-L36](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/router.py#L18-L36)）用 **base 模型**跑一遍提示词，`output_hidden_states=True` 取每层输入（`hidden_states[:-1]`，形状 `(num_layers, B, T, d)`），对序列取均值（`hidden_avg`）或最后一个 token，再对 batch 平均。于是第 l 层路由器 `W_gate[l] ∈ ℝ^{E×d}` 的第 e 行就是"专家 e 擅长的输入在第 l 层长什么样"的单位方向；推理时 `logits = W_gate · h` 本质是**余弦相似度路由**。`cheap_embed` 模式只用 token embedding 平均，所有层相同。

`warn_degenerate_gates`（[L154-L185](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/router.py#L154-L185)）用每层 `(E,d)` 矩阵的条件数检测"提示词太像导致路由向量几乎共线"。

**(c) 写出 Mixtral 权重** —— [moe/mixtral.py#L94-L188](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/mixtral.py#L94-L188)

```python
for weight_info in tqdm.tqdm(MISTRAL_INFO.all_weights(base_cfg), desc="Weights"):
    if ".mlp." in weight_info.name:
        continue
    else:
        copy_tensor_out(weight_info, base_loader, writer, ...)     # 非 MLP 权重全部来自 base
for layer_idx in tqdm.trange(base_cfg.num_hidden_layers, desc="Expert weights"):
    ...
    for expert_idx, expert in enumerate(config.experts):
        expert_loader = loaders.get(expert.source_model)
        for source_param, expert_param in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
            tensor = expert_loader.get_tensor(source_name)
            tensor = noise_and_scale(tensor, expert, is_residual=source_param == "down_proj")
            ...
for layer_idx, weight in enumerate(tqdm.tqdm(router_weights, desc="Router weights")):
    writer.save_tensor(self._router_weight_name(layer_idx), weight.to(dtype=out_dtype).contiguous(), ...)
```

- attention/norm/embedding：**只来自 base**。所以专家模型最好和 base 同源（同一底座微调），否则 attention 与 MLP 不匹配。
- 每个专家的 MLP 三件套原样放进 `experts.{e}.w1/w2/w3`（再按 transformers 新格式打包成 `gate_up_proj` / `down_proj`）。
- `noise_and_scale`（[moe/common.py#L60-L68](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/moe/common.py#L60-L68)）可加噪声打破对称，`residual_scale` 只缩放 `down_proj`（写回残差流的那一个）。
- 路由器写到 `model.layers.{l}.mlp.gate.weight`（L91-L92）。
- 入口串联见 [scripts/moe.py#L54-L78](https://github.com/arcee-ai/mergekit/blob/9eeb539892c6753b032ea164e818b44a8c7a401b/mergekit/scripts/moe.py#L54-L78)。

---

## 4. 方法对比表

| 方法 | 需要 base | 输入数 | 对 τ 的处理 | 关键超参 | 直觉 / 适用 |
|---|---|---|---|---|---|
| Linear (Model Soup) | 否 | ≥1 | 不用 τ，直接平均 θ | `weight` | 同一次预训练的多次微调（不同种子/超参）|
| Task Arithmetic | 是 | ≥1 | `Σ w_k τ_k` | `weight`, `lambda` | 能力加减；简单基线 |
| TIES | 是 | ≥1 | Trim(top-k) → 选符号 → 同号均值 | `density`, `weight`, `lambda` | 多任务合并，减少干扰 |
| DARE (linear / ties) | 是 | ≥1 | 随机丢弃 + 重缩放 | `density`（保留率） | τ 极冗余时；常 0.1–0.5 |
| Breadcrumbs | 是 | ≥1 | 去掉最小 + 最大 γ | `density`, `gamma` | 怀疑大幅值改动是噪声/过拟合 |
| DELLA | 是 | ≥1 | 按幅值名次的自适应随机丢弃 | `density`, `epsilon` | DARE 的改进 |
| SLERP | 是 | **恰好 2** | 球面插值 θ | `t`（可分层梯度） | 两模型融合，最流行的社区做法 |
| Model Stock | 是 | ≥3（含 base）| 由 τ 之间夹角自动算插值比 | 无 | 免调参的 soup |
| Passthrough | 否 | 每 slice 1 个 | 原样拷贝层 | `layer_range`, `scale` | 加深模型（frankenmerge）|
| MoE (mergekit-moe) | 是 | E 个专家 | MLP 变专家，router 由提示隐状态 | `gate_mode`, `experts_per_token`, prompts | 把多个领域模型组装为稀疏 MoE |

---

## 5. 常见困惑 / 坑

1. **不同底座的模型不能直接算术合并。** Task Arithmetic/TIES 假设所有模型在同一个底座的邻域内（共享"盆地"）。两个独立预训练的模型神经元顺序都不对应（排列对称性），直接平均会崩。
2. **`density` 是保留比例，不是丢弃比例。** DARE 论文里的"drop rate p=0.9"对应 mergekit `density: 0.1`。
3. **mergekit 的 DARE rescale 是 L1 匹配，不是 `/density`。** 期望一致，但单次结果略有差异；`dare_*` 默认 `rescale=True`，`ties` 默认 False。
4. **TIES 的 `weight` 不是"比例"。** `normalize=True` 时同号位置会除以同号者的权重和，所以三个模型都写 `weight: 1.0` 就是均值；但 `task_arithmetic` 默认 `normalize=False`，三个 `weight: 1.0` 就是 **τ 之和**（可能过冲，需要 `lambda` 或小权重）。
5. **base 是否在 `models:` 列表里无所谓**：规划器会自动补进去（plan.py L143-L156），但 NON_BASE 参数（`weight/density`）对 base 不生效。
6. **梯度参数 `[a, b, c]` 是按层插值的**，`t = idx/(num_layers-1)`；在 frankenmerge 多 slice 情况下，`t` 是在**每个 slice 内部**重新从 0 到 1 计算的。
7. **`filter` 是子串匹配且按顺序取第一个命中**，兜底项必须放最后；`filter: mlp` 会命中所有含 "mlp" 的张量名。
8. **SLERP 只能两个模型**；多模型请用 `multislerp`/`karcher`/`nuslerp`。SLERP 的夹角按**整个张量**计算。
9. **词表不同**（如一个模型加了特殊 token）：embedding/lm_head 形状不一致，会在 `validate_tensors` 报错。需配置 `tokenizer_source: base/union` 或 `tokenizer:`，让 `PermutedVocabulary` 先对齐。
10. **Frankenmerge 输出层数改变后**，`num_hidden_layers` 必须更新（mergekit 自动写 config），但下游若有按层号的东西（如 LoRA 适配器）就不能复用了。
11. **MoE 合并后不训练路由也能跑，但效果依赖提示词质量**。提示词太相似 → 路由向量共线 → `warn_degenerate_gates` 警告，某些专家永远选不到。
12. **DARE/DELLA 有随机性**：要复现结果需固定 `--random-seed`。

---

## 6. 自测题

1. 写出 TIES 三步各自的张量形状变化：输入 K 个 `(4096, 4096)` 的 τ，mask 的形状是什么？`divisor` 呢？
2. 在 `get_mask` 中，某位置三个 τ 为 `[0.3, -0.5, 0]`，`sum` 和 `count` 两种模式分别选出什么符号？哪些模型被纳入平均？
3. 为什么 SLERP 要在 `|cos Ω| > 0.9995` 时退化为线性插值？
4. 解释 `orcamini-platy-44layer.yml` 里输出模型第 30 层的权重来自哪个模型的哪一层（答：Platypus2 第 26 层）。
5. `ties` 与 `dare_ties` 在 `registry.py` 中的配置差别有哪几项？
6. 规划器在哪一步把 YAML 的 `weight: [0, 0.3, 0.7, 1]` 变成某一层的具体数？执行期还会读 YAML 吗？
7. mergekit-moe 的路由向量为什么要先减去负向提示的隐状态、再 L2 归一化？

## 7. 动手练习

1. **验证 2.1**：用三个同形状随机 state dict 跑你的 `merge(method="ties")`，再与 `mergekit.merge_methods.merge_state_dicts(..., "ties", base=..., parameters={"weight": ..., "density": ...})` 比较（需自行准备装有 mergekit 的环境；DARE 有随机性，只比 TIES）。注意 mergekit 的 trim 用 `argsort[:k]`，与 `kthvalue` 阈值在并列值上可能有微小差异。
2. **符号冲突统计**：对两个真实微调模型（同底座），统计每层 τ 的符号冲突比例 `mean(sign(τ1) != sign(τ2))`（忽略 0），以及 trim 到 20% 后的冲突比例。
3. **DARE 冗余实验**：对一个微调模型，令 `θ = θ_0 + dare(τ, p)`，p 取 0.5/0.1/0.01，测困惑度，体会"丢 90% 几乎无损"。
4. **写一个自定义方法**：参考 `linear.py`，用 `@merge_method` 写一个 "TIES 但用 count 投票 + 中位数代替均值" 的方法，并在 `registry.py` 的模式下 `register()`（只在本地实验，不提交）。
5. **画任务图**：在 `MergePlanner.plan_to_disk` 之后打印每个 `SaveTensor.tensor_task.gather_tensors.weight_info`，观察一个 frankenmerge 配置中输出权重名与输入权重名的对应关系。
