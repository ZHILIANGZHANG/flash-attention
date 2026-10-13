# CLAUDE.md

This repository is an **LLM source-code reading workspace**. The owner understands LLMs conceptually and is using it to learn the low-level implementation of KV cache, MLA/GQA, MoE, linear attention, LoRA, DPO/GRPO/on-policy distillation, pruning, merging, quantization, speculative decoding, distributed training, etc.

## Layout

- `README.md` — the roadmap (源码导读主线), in Chinese.
- `guide/NN-*.md` — one deep-dive guide per topic: principle → minimal PyTorch implementation → line-by-line walkthrough of real source → exercises.
- `repos/<NN-topic>/<lib>` — reference libraries as **shallow git submodules** pinned to a commit. Treat them as read-only reference code; do not commit changes inside them.
- `flash-attention/` — the original FlashAttention repository, moved here unchanged. It has its own `flash-attention/CLAUDE.md` with build/test instructions; run its commands from inside that directory.
- `notes/` — the owner's own notes and experiment scripts.

## How to help

- Reply in Chinese unless asked otherwise; keep code identifiers in English.
- Act as a tutor: explain *why* the code is written that way, spell out tensor shapes, and connect the code back to the math. Prefer walking through real code in `repos/` over generic explanations.
- Every path, symbol and line number you cite must be verified by reading the file. Line numbers are valid for the pinned submodule commit; when citing for the web, use permalinks `https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<a>-L<b>` with `git -C <submodule> rev-parse HEAD`.
- If a submodule directory is empty, initialise it: `git submodule update --init --depth 1 <path>`.
- When adding a library: `git submodule add --depth 1 <url> repos/<NN-topic>/<name>`, then `git config -f .gitmodules submodule.<path>.shallow true`, and add it to the README table and the relevant guide. Run `git submodule add` sequentially (parallel runs fight over `.git/index.lock`).
- Put exploratory scripts in `notes/` (or the scratchpad), not inside `repos/`.
