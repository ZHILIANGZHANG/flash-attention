# LLM 底层实现源码导读

这个仓库是一个**读源码学大模型**的工作区。本仓库自己的内容是导读文档，所有被读的源码都以 git submodule 的形式固定在某个 commit，所以导读里引用的行号会一直对得上。

> 原来的 FlashAttention 源码完整保留在 [`flash-attention/`](flash-attention/)，在那个目录里仍然可以照常编译和测试。

```
.
├── README.md            ← 你在这里：总路线图
├── guide/               ← 每个主题一篇导读：原理 → 最小实现 → 逐行读真实源码 → 练习
├── repos/               ← 被读的开源库（git submodule，按主题分组）
├── flash-attention/     ← 原 FlashAttention 仓库（FA2 / FA3 / FA4-CuTeDSL）
└── notes/               ← 你自己的学习笔记
```

## 0. 准备

```bash
git clone https://github.com/zhiliangzhang/flash-attention.git llm-src && cd llm-src

# 拉取全部被读的库（浅克隆，约 2.2 GB）
git submodule update --init --depth 1 repos

# 只拉某一个主题 / 某一个库
git submodule update --init --depth 1 repos/05-linear-attention
git submodule update --init --depth 1 repos/03-inference/nano-vllm
```

导读里的每个源码位置都同时给出本地路径和 GitHub 固定链接（permalink），手机上也能直接点开看。

## 1. 怎么读

每篇导读都是同一个结构：

1. **要解决的问题**：先想清楚为什么需要它
2. **核心原理**：公式，每个张量都标出 shape
3. **最小实现**：20~100 行 PyTorch，把本质抽出来，可以先跑这段
4. **源码导读**：按阅读顺序，一站一站地读真实代码，每一站讲清楚"这几行在干什么、为什么这么写"
5. **对比表 / 常见困惑 / 自测题 + 动手练习**

建议的读法：先读最小实现，再打开真实源码对照。看不懂的地方直接问 Claude：在这个仓库里打开 Claude Code，说"带我读 guide/07 的第 3 站"就行。

## 2. 主线路线图

按依赖关系排序。前一阶段是后一阶段的基础，同一阶段内的主题可以按兴趣挑着读。

### 阶段一：一个 LLM 是怎么跑起来的

| # | 主题 | 导读 | 主要源码 | 先读最小的 |
|---|---|---|---|---|
| 01 | Transformer 全流程：BPE、RMSNorm、RoPE、SwiGLU、采样 | [guide/01](guide/01-transformer-basics.md) | nanoGPT, minbpe, nanochat, transformers/llama | `nanoGPT/model.py` |
| 02 | 注意力变体：MHA / MQA / GQA / **MLA** | [guide/02](guide/02-attention-mha-gqa-mqa-mla.md) | transformers, DeepSeek-V3, FlashMLA | `DeepSeek-V3/inference/model.py` |

### 阶段二：推理，围绕 KV cache 的一切

| # | 主题 | 导读 | 主要源码 | 先读最小的 |
|---|---|---|---|---|
| 03 | **KV cache**、PagedAttention、continuous batching、前缀缓存 | [guide/03](guide/03-kv-cache-and-paged-attention.md) | nano-vllm, mini-sglang, vllm, sglang | `nano-vllm`（约 1500 行的 vLLM） |
| 04 | FlashAttention 与 attention kernel：online softmax、tiling、split-KV | [guide/04](guide/04-flash-attention-and-kernels.md) | flash-attention, triton, flashinfer, xformers | `triton/python/tutorials/06-fused-attention.py` |
| 05 | **KV 压缩 / token 剪枝**：StreamingLLM、H2O、SnapKV、FastV、ToMe、LLMLingua | [guide/05](guide/05-kv-compression-and-token-pruning.md) | kvpress, streaming-llm, FastV, ToMe, LLMLingua | `kvpress` |
| 12 | 量化：RTN、GPTQ、AWQ、KV 量化 | [guide/12](guide/12-quantization.md) | gptq, llm-awq | `gptq/gptq.py` |
| 13 | 投机解码：rejection sampling、Medusa、EAGLE、MTP | [guide/13](guide/13-speculative-decoding.md) | Medusa, EAGLE, vllm | `Medusa` |

### 阶段三：模型结构的演进

| # | 主题 | 导读 | 主要源码 | 先读最小的 |
|---|---|---|---|---|
| 06 | **线性注意力 / SSM**：Linear Attn、RetNet、GLA、DeltaNet、Mamba、RWKV | [guide/06](guide/06-linear-attention-and-ssm.md) | flash-linear-attention, mamba, RWKV-LM | `fla/ops/*/naive.py` |
| 07 | **MoE**：路由、负载均衡、dispatch/combine、grouped GEMM、专家并行 | [guide/07](guide/07-moe.md) | DeepSeek-V3, megablocks, DeepEP, megatron, vllm | `DeepSeek-V3/inference/model.py` |

### 阶段四：训练与后训练

| # | 主题 | 导读 | 主要源码 | 先读最小的 |
|---|---|---|---|---|
| 08 | **LoRA** / PEFT：QLoRA、DoRA、多 LoRA 服务 | [guide/08](guide/08-lora-and-peft.md) | LoRA, peft, vllm/lora | `LoRA/loralib/layers.py` |
| 09 | 后训练：SFT、PPO、**DPO**、**GRPO**、**OPD**（on-policy distillation） | [guide/09](guide/09-post-training-sft-dpo-grpo-opd.md) | trl, verl, OpenRLHF, tinker-cookbook | `trl` 的 DPO loss |
| 14 | 分布式训练：DDP、ZeRO/FSDP、TP、PP、CP、EP | [guide/14](guide/14-distributed-training.md) | megatron-lm, deepspeed, torchtitan, accelerate | `torchtitan` |

### 阶段五：模型压缩与合并

| # | 主题 | 导读 | 主要源码 | 先读最小的 |
|---|---|---|---|---|
| 10 | **模型剪枝**：magnitude、Wanda、SparseGPT、2:4、LLM-Pruner、DepGraph | [guide/10](guide/10-model-pruning.md) | wanda, LLM-Pruner, Torch-Pruning | `wanda/lib/prune.py` |
| 11 | **模型合并**：task arithmetic、SLERP、TIES、DARE、MoE merge | [guide/11](guide/11-model-merge.md) | mergekit | `mergekit/mergekit/merge_methods/` |

> 编号是主题 ID，不代表阅读顺序。读的顺序按上面的阶段走。12 量化、13 投机解码放在阶段二，是因为它们都是推理优化。

## 3. 库清单（`repos/`）

| 目录 | 库 | 在本仓库里用来学什么 |
|---|---|---|
| `00-basics/` | [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat), [minbpe](https://github.com/karpathy/minbpe) | 最小可读的 GPT 训练 / 对话 / BPE |
| | [transformers](https://github.com/huggingface/transformers), [tokenizers](https://github.com/huggingface/tokenizers), [safetensors](https://github.com/huggingface/safetensors) | 工业级模型定义、attention 分发、cache、generate |
| `01-attention/` | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | MLA 参考实现（含 absorb）、MLA decode kernel |
| `02-attention-kernels/` | [triton](https://github.com/triton-lang/triton), [flashinfer](https://github.com/flashinfer-ai/flashinfer), [xformers](https://github.com/facebookresearch/xformers), [quack](https://github.com/Dao-AILab/quack), [liger-kernel](https://github.com/linkedin/Liger-Kernel) | attention / 融合算子 kernel |
| `03-inference/` | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [mini-sglang](https://github.com/sgl-project/mini-sglang), [vllm](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang) | KV cache 管理、调度、推理引擎 |
| `04-kv-compression-token-pruning/` | [kvpress](https://github.com/NVIDIA/kvpress), [streaming-llm](https://github.com/mit-han-lab/streaming-llm), [FastV](https://github.com/pkunlp-icler/FastV), [ToMe](https://github.com/facebookresearch/ToMe), [LLMLingua](https://github.com/microsoft/LLMLingua) | KV 淘汰、token 剪枝 / 合并、prompt 压缩 |
| `05-linear-attention/` | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [mamba](https://github.com/state-spaces/mamba), [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | 线性注意力、SSM |
| `06-moe/` | [megablocks](https://github.com/databricks/megablocks), [DeepEP](https://github.com/deepseek-ai/DeepEP) | dropless MoE、专家并行通信 |
| `07-finetune/` | [LoRA](https://github.com/microsoft/LoRA), [peft](https://github.com/huggingface/peft), [accelerate](https://github.com/huggingface/accelerate) | LoRA 原始实现与工业实现 |
| `08-rl-alignment/` | [trl](https://github.com/huggingface/trl), [verl](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | SFT / DPO / GRPO / PPO / 蒸馏 |
| `09-model-pruning/` | [wanda](https://github.com/locuslab/wanda), [LLM-Pruner](https://github.com/horseee/LLM-Pruner), [Torch-Pruning](https://github.com/VainF/Torch-Pruning) | 非结构化 / 结构化剪枝 |
| `10-model-merge/` | [mergekit](https://github.com/arcee-ai/mergekit) | 模型合并 |
| `11-quantization/` | [gptq](https://github.com/IST-DASLab/gptq), [llm-awq](https://github.com/mit-han-lab/llm-awq) | 权重量化 |
| `12-speculative-decoding/` | [Medusa](https://github.com/FasterDecoding/Medusa), [EAGLE](https://github.com/SafeAILab/EAGLE) | 投机解码 |
| `13-distributed/` | [megatron-lm](https://github.com/NVIDIA/Megatron-LM), [deepspeed](https://github.com/deepspeedai/DeepSpeed), [torchtitan](https://github.com/pytorch/torchtitan) | 各种并行策略 |
| 根目录 | [flash-attention](flash-attention/) | FA2 (CUDA) / FA3 (Hopper) / FA4 (CuTeDSL) |

## 4. 常用操作

```bash
# 跨库搜某个概念出现在哪里
grep -rn "def repeat_kv" repos/00-basics/transformers/src/transformers/models/llama/
grep -rln "reverse_kl\|on_policy" repos/08-rl-alignment --include=*.py

# 把某个库更新到上游最新（更新后导读里的行号可能会偏移）
git submodule update --remote --depth 1 repos/03-inference/vllm

# 想单步调试某个库
pip install -e repos/03-inference/nano-vllm
```

## 5. 学习记录

把自己的笔记、跑过的实验脚本放在 [`notes/`](notes/)。
