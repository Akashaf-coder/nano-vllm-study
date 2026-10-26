# 15 与 DeepInfer 对照：同一套概念的两种极端

## 本课目标

- 把 nano-vllm 的每个函数映射到生产框架里的对应机制；
- 理解"为什么 DeepInfer 只用 EP/DP 不做 TP，而 nano-vllm 只用 TP"；
- 知道两者的复杂度差在哪、哪些复杂度是"规模逼出来的"、哪些是"场景选择"；
- 拿到一条学习路径：先用 nano-vllm 建立骨架，再用 DeepInfer 看放大后的样子。

对照对象：`deepseek-ai/DeepInfer`（commit `9bd9cf2`），我另一份精读在 `deepinfer-study` 仓库。

## 1. 规模：39 倍

| | nano-vllm | DeepInfer |
| --- | --- | --- |
| 语言构成 | Python 单语言 | Python + C++（调度器）+ TileLang（kernel） |
| 行数 | **1,450 行 / 21 个文件** | **Python 54,929 行（270 文件）+ C++ 2,425 行（15 文件）≈ 5.7 万行** |
| 构建 | `pip install -e .` | `./build.sh` + vendor 子模块 + `DEEPINFER_ACCELERATOR` 环境变量 |
| 可运行性 | 单卡 CUDA 即可 | 需要 SM100（B200 级）或 Ascend 950；**本机两者都没有** |
| 分析方式 | 可跑原码验证调度/缓存 | 静态分析 + 仓库自带图表 |

两个数字放在一起就是全部：**39 倍代码量，换来的不是"更快的 decode 循环"，而是把单卡批处理变成了能在生产集群上跑的系统。**

## 2. 八个维度的对照

| 维度 | nano-vllm | DeepInfer | 差异的根源 |
| --- | --- | --- | --- |
| 调度 | 两队列，同步，批级；一次 step 全 prefill 或全 decode | ZOS 异步调度：四段流水线，把 CPU 调度藏进 GPU 执行窗口；chunk 裁剪 + LCFS 抢占 | nano 的目标是**读懂**，DeepInfer 的目标是**不让 GPU 等 CPU** |
| 前缀缓存 | 引擎内，块哈希链，只共享写满的块 | 引擎**外**（3FS + Entrypoint），引擎内只有 fork/COW | 生产环境要跨实例共享，单机内存装不下 |
| 重算 | 抢占即全量重算，无上界 | YoCo bounded replay：重算量硬约束（128 token + 6 层） | 生产不允许"最坏情况不可控" |
| 并行 | **只有 TP**（1–8 卡，按头切） | **只有 EP/DP**（专家并行 + 数据并行，无 TP） | 稀疏 MoE 模型按专家切更划算；稠密小模型切头更简单 |
| 精度 | bf16 单一 | FP4 权重/激活、state slot 管理 | 生产要榨干显存带宽 |
| 注意力 | 标准 MLA/GQA + FlashAttention（varlen + paged） | CSA2 稀疏注意力：压缩 + 稀疏 + SWA + 跨层共享 | 长上下文 + 生产成本 |
| 采样 | 12 行：温度 + Gumbel-max | fused sampler、batch-invariant、xgrammar 异步编译的结构化输出 | 生产要"同 batch 内结果不随 batch 组成变化" |
| 服务面 | 无（离线 `generate`） | 离线但完整：多模态 VE 独立部署、EPD 分离、双硬件平台抽象 | 生产要接业务 |

## 3. 概念映射表（读 DeepInfer 时带着这张表）

| nano-vllm | DeepInfer 里叫什么 | 关系 |
| --- | --- | --- |
| `Scheduler.schedule()` | ZOS 异步调度器（C++） | 同一职责，但从"同步返回一批"变成"流水线四段" |
| `BlockManager` 的块哈希链 | BlockManager + Entrypoint（外部前缀缓存） | 引擎内部分基本同构，缓存位置外移 |
| `Sequence.num_cached_tokens` | state slot / block table | 同类概念，DeepInfer 多了"状态槽"这一层 |
| `preempt()` + 全量重算 | YoCo bounded replay（128 + 6） | **同一条思路（丢弃重算）加了上界** |
| `tensor_parallel_size` + `all_reduce` | EPLB（专家负载均衡）+ symm_mem shuffle | 不同并行轴：nano 切头，DeepInfer 搬专家 |
| `Attention.forward`（FlashAttention） | CSA2 稀疏注意力（TileLang kernel） | nano 买现成，DeepInfer 自己写 |
| `Sampler`（Gumbel-max） | fused sampler + batch-invariant | 同一目标（采样），DeepInfer 追求"与 batch 组成无关" |
| `capture_cudagraph()` | CUDA Graph（双硬件平台各一份） | 同构，DeepInfer 要为两种硬件各写一次 |
| `--`（无） | DSpark 投机解码 | nano 完全没有这一层 |
| `--`（无） | 多模态 VE / EPD | nano 完全没有这一层 |

**读法建议**：先把左边一列读通（这份教材的目的），再去看右边 —— 每一个都能对上"我在 nano 里见过的最小版本"。

## 4. 为什么并行轴相反

这是最值得琢磨的一处对照。

**nano-vllm 只有 TP**：

- 模型是稠密的（Qwen3-0.6B，28 层 attention + MLP）；
- 单卡装得下，TP 只是为了"更快"；
- TP 的实现成本低：切输出/切输入 + 一次 `all_reduce`（`docs/10`），不需要路由、不需要负载均衡；
- 缺点：每层两次 `all_reduce`，小 batch 时通信占比高。

**DeepInfer 只有 EP/DP**：

- 模型是 MoE（V4.1-Flash），专家数量远超卡数；
- 单卡装不下整个模型，必须按专家切；
- EP 的代价是**路由与负载不均**，于是要有 EPLB（专家并行负载均衡）去做"逻辑专家 → 物理专家"的贪心重排 + `symm_mem` shuffle；
- 不用 TP 是因为 MoE 的 attention 部分相对小，切它收益不大，而 `all_reduce` 在高带宽互联下反而不如"搬专家"划算。

一句话：**并行轴由模型结构决定，不由框架偏好决定。** 稠密小模型切头，稀疏大模型切专家。

## 5. 复杂度都花在哪

把 39 倍拆开看：

| 复杂度来源 | nano-vllm | DeepInfer 里对应 | 为什么必须有 |
| --- | --- | --- | --- |
| CPU 与 GPU 的重叠 | 无（同步） | ZOS 异步四段流水线 | 生产里 CPU 调度耗时和 GPU 计算同量级，不重叠就浪费一半 |
| 缓存位置 | 引擎内 | 3FS + Entrypoint | 单机显存装不下跨请求的公共前缀 |
| 重算上界 | 无 | YoCo 128+6 | 尾延迟 SLA |
| 并行复杂度 | 1 个轴 | EP + DP + 负载均衡 | MoE + 多机 |
| kernel | 20 行 Triton | TileLang 双硬件 kernel 工程 | 要吃到 SM100 的新指令 |
| 采样 | 12 行 | fused + batch-invariant + 结构化输出 | 生产要"可复现 + 约束解码" |
| 平台抽象 | 无 | Platform 抽象层（NVIDIA / Ascend） | 两种硬件 |
| 服务面 | 无 | 多模态、EPD | 业务需求 |

**没有一项是"为了复杂而复杂"** —— 每一行都对应一个生产约束。反过来，nano-vllm 的 1450 行也每一项都有用，没有一行是"为了教学而省略的假代码"。

## 6. nano-vllm 回答不了、DeepInfer 才能回答的五个问题

1. **"GPU 为什么在等 CPU？"** —— nano 里 GPU 确实在等（同步），但它不测、不管、不优化。
2. **"长上下文怎么不炸显存？"** —— nano 只有 paged KV，没有稀疏注意力、没有滑窗、没有跨层共享。
3. **"跨实例的前缀怎么共享？"** —— nano 的缓存活在进程内存里，进程退出即消失。
4. **"专家负载不均怎么办？"** —— nano 没有专家。
5. **"同 batch 内结果为什么会变？"** —— nano 甚至没有 batch-invariant 的概念（它的采样只依赖温度，理论上与 batch 无关，但 `all_reduce` 的数值归约顺序会让 logits 有微小差异；这个坑 DeepInfer 用 batch-invariant kernel 显式解决了）。

## 7. 反过来：DeepInfer 教不了你的

- **最小的可运行骨架**：5.7 万行里没有一个文件能让你"跟着跑一遍"。nano-vllm 的 `example.py` 33 行就能看到全貌。
- **每个概念的"为什么"**：生产代码里到处是"因为历史原因/因为某次事故"的分支；nano-vllm 只有一个理由：这件事必须做，否则跑不起来。
- **CPU 侧的仿真验证**：nano-vllm 的调度器与块管理器**可以原样 import 跑**（`labs/`），DeepInfer 的 C++ 调度器做不到。

## 8. 推荐学习路径

```
第 1 步（20h）：本仓库 docs/00–15
   └─ 产出：能画出"prompt → token"全链路，能改调度参数预测行为

第 2 步（10h）：动手改 nano-vllm
   ├─ 加 top-k 采样（改 sampler.py + sampling_params.py，约 30 行）
   ├─ 加 swap（改 block_manager + scheduler，约 150 行，见 docs/06）
   └─ 加抢占计数埋点（3 行，见 docs/14 第 6 节）

第 3 步（40h）：deepinfer-study 的 15 节课
   └─ 每一节都回来对照本仓库的相应模块："最小版是什么样、生产版加了什么"

第 4 步：挑一个差异点做深
   推荐三选一：
   ├─ ZOS 异步调度 ↔ docs/04 的两队列（CPU/GPU 重叠）
   ├─ YoCo bounded replay ↔ docs/06 的全量重算（重算上界的价值）
   └─ batch-invariant sampler ↔ docs/12 的 Gumbel-max（数值一致性）
```

## 9. 一页纸对照（汇报用）

```
                     nano-vllm                    DeepInfer
定位          教学骨架（可跑通的最小闭环）      生产框架开源子集
规模          1,450 行                        ≈ 57,000 行（39×）
并行          只有 TP（切头）                  只有 EP/DP（切专家）
调度          两队列 / 同步 / 批级             ZOS 四段流水线 / 异步
前缀缓存      引擎内块哈希                    引擎外 3FS + 引擎内 COW
重算          抢占即全量                       bounded replay 128+6
注意力        FlashAttention（买）             CSA2 稀疏（自研 TileLang）
采样          12 行 Gumbel-max                 fused + batch-invariant
短板          decode 饥饿、无公平性、无 swap    复杂度高、无法在消费级卡跑
能学到的      "推理引擎最少需要什么"           "为单一模型做外科手术要付出什么"
```

## 10. 自测

1. 为什么 DeepInfer 不用 TP？如果它的模型换成稠密 7B，这个选择会变吗？
2. "重算无上界"在离线批处理里可以接受，在在线服务里为什么不行？
3. 前缀缓存放在引擎外（3FS）解决了什么问题、引入了什么新问题？
4. 用一句话回答：1450 行能打平 vLLM，靠的是什么？（提示：省掉了什么）
5. 如果你要给 nano-vllm 加一个"生产化"特性，按性价比排序，前三名是什么？

## 结束语

读完这 16 节，你应该能做到两件事：

- **看任何推理引擎的架构图，都能指出"调度、缓存、并行、kernel、采样"这五块在哪、边界在哪**；
- **遇到"为什么它这么快/这么慢"的问题时，知道该测什么**（吞吐 vs TTFT vs ITL，缓存命中率，抢占次数）。

回到 `docs/00` 的三遍读法，把第三遍（动手改）做完，这份材料才算真正吸收。
