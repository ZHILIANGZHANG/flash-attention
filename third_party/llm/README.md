# 大模型源码阅读工作区（third_party/llm）

这里的每个子目录都是一个 **git submodule**，用 `--depth 1` 浅克隆，`.gitmodules` 里设了 `shallow = true`，所以本仓库只记录各库的 commit 指针，体积不会变大。

## 获取 / 更新

```bash
# 新 clone 本仓库后，拉取全部（约 0.9 GB）
git submodule update --init --depth 1 third_party/llm

# 只拉某一个
git submodule update --init --depth 1 third_party/llm/transformers

# 把某个库更新到上游最新
git submodule update --remote --depth 1 third_party/llm/vllm
```

## 收录的库

| 层次 | 库 | 看点 |
|---|---|---|
| 模型定义 | `transformers` | 模型结构、attention 后端分发、KV cache、mask、generate |
| 分词 / 权重 | `tokenizers`, `safetensors` | BPE 的 Rust 实现；零拷贝的权重格式 |
| 训练 | `accelerate`, `peft`, `trl` | 分布式封装、LoRA、SFT/DPO/GRPO |
| 大规模训练 | `megatron-lm`, `deepspeed` | TP/PP/CP 并行、ZeRO、MoE |
| 推理引擎 | `vllm`, `sglang` | PagedAttention、continuous batching、调度器、RadixAttention |
| Kernel | `triton`, `xformers`, `quack`, `liger-kernel` | Triton 编译器；memory-efficient attention；CuTeDSL kernel（与 FA4 同作者）；Triton 融合算子 |
| 入门 | `nanoGPT`, `nanochat` | 几百行的完整 GPT 训练 / 对话流程，适合先读 |

## 推荐阅读路线（以 FlashAttention 为主线）

### 第 1 步：先搞清楚一个 Transformer 长什么样
- `nanoGPT/model.py`：约 300 行，里面有 `CausalSelfAttention`，可以对照 `F.scaled_dot_product_attention`
- `transformers/src/transformers/models/llama/modeling_llama.py`
  - `LlamaRMSNorm`、`LlamaRotaryEmbedding`（RoPE）、`LlamaMLP`（SwiGLU）
  - `eager_attention_forward`：最朴素的 softmax(QKᵀ)V，**这就是 FlashAttention 要算的东西**
  - `LlamaAttention.forward`：通过 `ALL_ATTENTION_FUNCTIONS.get_interface(...)` 分发到 eager / sdpa / flash / flex

### 第 2 步：transformers 怎么调用 flash-attn
- `transformers/src/transformers/integrations/flash_attention.py`：`flash_attention_forward`
- `transformers/src/transformers/modeling_flash_attention_utils.py`
  - `lazy_import_flash_attention`：如何在 FA2 / FA3 / FA4 之间选择
  - `_upad_input` / `_get_unpad_data`：把 padding 的 batch 转成 varlen（`cu_seqlens`），对应本仓库 `flash_attn_varlen_func`
  - `prepare_fa_kwargs_from_position_ids`：packed sequence（多条样本拼接）
- 对照阅读：`flash_attn/cute/interface.py`（本仓库 FA4 入口）
- 其他后端：`integrations/sdpa_attention.py`、`integrations/flex_attention.py`、`masking_utils.py`
- KV cache：`transformers/src/transformers/cache_utils.py`

### 第 3 步：推理引擎里的 attention（paged KV、prefill/decode）
- vLLM
  - `vllm/vllm/v1/attention/backends/flash_attn.py`、`fa_utils.py`：vLLM 调 FlashAttention 的方式（`block_table` → 本仓库 `paged_kv.py`）
  - `vllm/vllm/v1/core/kv_cache_manager.py`、`block_pool.py`：分页 KV cache 的管理
  - `vllm/vllm/v1/core/sched/`：continuous batching 调度
  - `vllm/vllm/v1/worker/gpu_model_runner.py`：一次 forward 前要准备哪些元数据
- SGLang
  - `sglang/python/sglang/srt/layers/attention/flashattention_backend.py`（里面有 FA4 cute 的使用）
  - `sglang/python/sglang/srt/mem_cache/`：RadixAttention 前缀缓存

### 第 4 步：kernel 层
- `xformers/`：memory-efficient attention，FlashAttention 之前的思路
- `triton/python/tutorials/06-fused-attention.py`：用 Triton 写的 FlashAttention，比 CUDA/CuTe 版本好读很多
- `quack/quack/`：和 FA4 一样用 CuTeDSL，GEMM、softmax、cross-entropy，可以对照本仓库 `flash_attn/cute/`
- `liger-kernel/`：RMSNorm、RoPE、SwiGLU、CrossEntropy 的 Triton 融合实现

### 第 5 步：训练侧的 attention 与并行
- `megatron-lm/megatron/core/transformer/attention.py`、`dot_product_attention.py`、`multi_latent_attention.py`
- `megatron-lm/megatron/core/transformer/moe/`
- `deepspeed/deepspeed/runtime/zero/`：ZeRO 1/2/3
- `peft/src/peft/tuners/lora/`：LoRA 怎么注入到 attention 的 q/k/v 投影上

## 小技巧
- 跨库追踪调用：`grep -rn "flash_attn_varlen_func" third_party/llm --include=*.py`
- 只读源码不用安装；想单步调试可以 `pip install -e third_party/llm/transformers`
- 自己的笔记放 `agent_space/`（CLAUDE.md 约定的草稿区）
