# nano-vllm 精读：16 节课 + 分享稿

> 面向"想读懂推理引擎到底怎么转"的读者：把一个 1450 行的最小 vLLM 实现拆到能讲、能改、能验证。
> 分析对象：`GeeeekExplorer/nano-vllm`，commit `bb823b3`（2026-04-26），v0.2.0，MIT。

## 为什么读它

DeepInfer 那一套看的是"生产框架如何为单一模型做外科手术"；nano-vllm 相反，是**把推理引擎削到只剩必要部件**：没有 OpenAI 服务端、没有量化、没有投机解码、没有多模态，1450 行里只有调度、分页 KV、张量并行、FlashAttention、CUDA Graph 和一行采样。

读完它你会知道：请求进来以后每一步在哪个函数里、KV 块怎么被复用和抢占、TP 的两个进程怎么通信、CUDA Graph 为什么只给 decode。这些概念在 vLLM/SGLang/DeepInfer 里是同一套骨架，只是被工程放大到几万行。

## 目录

```
nano-vllm-study/
├── README.md                          # 本文件
├── docs/                              # 16 节课
│   ├── 00-导读-学习地图.md
│   ├── 01-项目全景与血统.md
│   ├── 02-请求生命周期.md
│   ├── 03-Sequence与状态机.md
│   ├── 04-Scheduler两队列调度.md
│   ├── 05-BlockManager与前缀缓存.md
│   ├── 06-抢占与重算.md
│   ├── 07-进程模型与张量并行.md
│   ├── 08-KV缓存显存分配.md
│   ├── 09-Attention分页与变长.md
│   ├── 10-并行线性层与权重加载.md
│   ├── 11-Qwen3模型图与融合.md
│   ├── 12-Sampler与采样.md
│   ├── 13-CUDA-Graph捕获.md
│   ├── 14-性能基准与实验方法.md
│   └── 15-与DeepInfer对照.md
├── labs/                              # 本机可跑的实验（纯 CPU，跑仓库原码）
│   ├── sim_prefix_cache.py            # 验证块管理/前缀缓存语义
│   ├── sim_scheduler.py               # 验证 prefill 优先、chunk 只给队首、LIFO 抢占
│   ├── sim_sampler.py                 # 验证 Gumbel-max 采样与温度
│   ├── run_labs.sh                    # 一键跑全部
│   └── RUN_OUTPUT.md                  # 本机真实输出（文中引用的数字都来自这里）
└── slides/
    ├── nano-vllm-深度解析.html         # 自包含 slides（49 页，双击打开）
    ├── nano-vllm-深度解析.pdf          # 同内容 PDF（16:9）
    ├── deck_template.html             # 模板（可改）
    ├── build_deck.py                  # 构建脚本：注入 slide 内容
    └── assets/                        # 预览截图
```

## 怎么用

- **只想快速了解（30 分钟）**：`docs/01` → `docs/02` → `docs/04` → `docs/05` → `docs/15`，然后翻 slides。
- **系统读源码（约 20 小时）**：按 `docs/00` 的三条路线走，每条课后都有自测题。
- **想动手改**：先跑 `labs/`（纯 CPU 就能跑，验证调度和前缀缓存），再按 `docs/08`、`docs/13` 在本机之外补 GPU 侧实验。

```bash
# 跑全部 CPU 实验（不需要 GPU / torch）
python3 -m venv ~/.venv-nanovllm-lab
~/.venv-nanovllm-lab/bin/pip install numpy xxhash
git clone https://github.com/GeeeekExplorer/nano-vllm ~/nano-vllm
bash labs/run_labs.sh

# 看 slides
open slides/nano-vllm-深度解析.html
```

## 姊妹仓库

- [`deepinfer-study`](https://github.com/Akashaf-coder/deepinfer-study)：DeepSeek DeepInfer（≈5.7 万行生产框架）的 15 节课精读。
- 建议顺序：**先读本仓库建立骨架**，再读 DeepInfer 看放大后的样子 —— `docs/15` 是两者的对照表。

## 前置知识

Python、PyTorch 基本用法；`docs/09`、`docs/13` 需要一点 CUDA 编程概念（kernel、流、图捕获）；`docs/10` 需要知道"列并行/行并行的区别"这一件事。

## 事实与标注

- 所有代码引用格式 `文件:行号`，基于 commit `bb823b3`。
- **本机（Apple M5，macOS）无法运行 nano-vllm**：依赖 `flash-attn`/`triton` 需要 CUDA，且 `pyproject.toml` 限定 `>=3.10,<3.13`（本机 3.14）。因此 kernel、CUDA Graph、端到端吞吐均为**静态分析**，不冒充实测。
- 可以验证的部分已确实跑过：`labs/` 里的三个脚本直接 `import` 仓库源码（`block_manager.py`、`scheduler.py` 只依赖 `xxhash`/`numpy`），`labs/RUN_OUTPUT.md` 是本机真实输出。`sim_sampler.py` 是 numpy 等价复现（本机无 torch），文中已明确标注。
- 性能数字全部标注来源：仓库 README 的对比表 vs 本机实验，两者不混用。

## 原始仓库

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm && pip install -e .
# 需要 CUDA 环境 + Qwen3-0.6B 权重
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python example.py
```

## 16 节课速览

| 课 | 一句话 |
| --- | --- |
| 00 | 三条路线 + 三遍读法 + 本机能做什么 |
| 01 | 1450 行的地图、血统、六个边界 |
| 02 | 一次 generate 的七个阶段与可观测点 |
| 03 | Sequence 的两个状态 + 跨进程瘦身 |
| 04 | 两队列、prefill 优先、chunk 只给队首 |
| 05 | 块分配、哈希链前缀缓存、尾块不共享 |
| 06 | ref_count、COW、LIFO 抢占与全量重算 |
| 07 | spawn 进程、SharedMemory、rank0 采样 |
| 08 | KV 显存公式、warmup 测峰、按模块注入 |
| 09 | Triton 写 KV、varlen prefill、分页 decode |
| 10 | 列/行并行、QKV 合并、weight_loader |
| 11 | Qwen3 图、qk-norm、残差融合、只算末位 |
| 12 | 除以指数的 argmax = Gumbel-max |
| 13 | 为什么要 CUDA Graph、为什么只给 decode |
| 14 | 基准怎么读、vLLM 对比的公平性边界 |
| 15 | 与 DeepInfer：同一套概念的两种极端 |

## 三个核心结论（一页纸）

1. **它就是 vLLM 的骨架**：分页 KV + 前缀缓存 + 连续批处理 + TP，没有一处是重新发明的；价值在于把 vLLM 里散在几十个文件的路径压到 1450 行。
2. **调度是"批级"而非"token 级"**：一次 step 要么全 prefill 要么全 decode，chunked prefill 只对队首生效 —— 这是它和 vLLM V1 最大的行为差异，也是 decode 饥饿的根因（`labs/sim_scheduler.py` 实测）。
3. **没有 swap/offload/淘汰**：KV 不够就把整条序列清空重算（LIFO 抢占）。省了实现，代价是被抢占的长序列可能长期饥饿。
