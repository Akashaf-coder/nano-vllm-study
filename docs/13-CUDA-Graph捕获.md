# 13 CUDA Graph：为什么只给 decode 录

## 本课目标

- 说清 CUDA Graph 在这个项目里到底省了什么（不是省计算，是省 launch）；
- 看懂 `graph_bs` 的分桶规则与"用大桶凑小 batch"的代价；
- 掌握 `run_model` 里那五个静态 buffer 的填充规则（含 `-1` padding 的意义）；
- 知道为什么 prefill 永远走 eager。

## 1. 问题的量级

decode 一步，一条序列只算 1 个 token。此时 **GPU 计算时间 ≈ kernel launch 时间**：

```
28 层 × (norm + qkv + rope + attn + o + norm + gate_up + silu + down + ...) ≈ 数百个 kernel
每 kernel launch overhead ≈ 5–10 μs（Python + CUDA driver）
→ 纯 launch 开销可达 1–3 ms/步，而计算可能只要 0.3 ms
```

CUDA Graph 把"录好的一整串 kernel 调用 + 参数"一次 `graph.replay()` 提交，launch 开销从 O(kernel 数) 降到 O(1)。对 decode（形状固定、每步重复）是天作之合；对 prefill（`cu_seqlens` 形状随 batch 变化）则没法录。

## 2. 捕获代码逐段

```python
# model_runner.py:222-248
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)                                  # ① 桶上限
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)                           # ② 五个静态 buffer
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))               # ③ 分桶
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):                                            # ④ 从大到小录
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])                 # ⑤ 必须先热身一次
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])             # ⑥ 真正捕获
        if self.graph_pool is None:
            self.graph_pool = graph.pool()                                        # ⑦ 共享内存池
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    self.graph_vars = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
                           context_lens=context_lens, block_tables=block_tables, outputs=outputs)
```

### ① `min(max_num_seqs, 512)`

默认 `max_num_seqs=512`，所以桶上限 512。这个数同时决定了 `run_model` 的"超 512 走 eager"门限（`model_runner.py:197`）。

### ③ 分桶规则：`[1, 2, 4, 8] + 16, 32, 48, ..., 512`

十一个 + 三十一个 = **35 个 bucket**。小 batch 密（1/2/4/8），大 batch 每 16 一档。理由：在线服务的 batch 分布通常集中在小的那边，而且小 batch 时 launch 开销占比最高，值得多录几档。

### ④ `reversed(...)`：从大到小

`self.graph_pool` 只在第一次循环时设置（`if self.graph_pool is None`），也就是**用最大的桶（512）的 pool 作为共享池**。如果从小到大录，小桶的 pool 会先被记住，后续大桶的内存申请可能落在池外的碎片区，导致重放时地址不稳定。

这是 CUDA Graph 的经典写法（vLLM 里同样从大到小）。**顺序不能改。**

### ⑤ 先热身一次再捕获

`outputs[:bs] = self.model(...)` 在 `with torch.cuda.graph(...)` **之外**先跑一遍。原因：`torch.compile` 的编译、cuBLAS 的算法选择（autotuning）、分配器的首次扩张都必须在捕获前完成 —— 捕获过程中**不允许任何"第一次发生"**，否则图里会记录到不合法的地址/句柄。

### ⑦ 共享 pool

35 个图共用一块内存池。省显存（否则每桶一份中间激活），代价是**图之间必须"互斥"使用同一地址**，且捕获顺序决定了哪些地址被复用。

## 3. 重放：`run_model` 的五步填充

```python
# model_runner.py:196-212
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))     # ← eager 路径
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]         # ① 向上取桶
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids                               # ② 拷贝输入
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)                                   # ③ 清空 + padding
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()                                     # ④ 清零
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()                                                         # ⑤ 重放
        return self.model.compute_logits(graph_vars["outputs"][:bs])           # ⑥ 只取前 bs 行
```

六个点：

① **向上取桶**：`bs=17` → 用 32 的桶 → **浪费一半算力**。这是"图必须定长"的必然代价，也是分桶要密的理由。`next(x for x in graph_bs if x >= bs)` 在没有更大桶时会 `StopIteration` —— 但 `bs > 512` 已经在入口被拦到 eager，所以安全。

③ **`slot_mapping.fill_(-1)` 然后只写前 bs 个**：多出来的位置保持 -1，Triton kernel 遇到 `slot == -1` 直接 return（`attention.py:23`）。**这就是 `-1` 的全部意义 —— 让超出的行"什么都不写"，而不是写脏数据。**

④ **`context_lens.zero_()`**：FlashAttention 的 `cache_seqlens` 为 0 表示"没有 key"，配合 `slot_mapping=-1` 让多余行衰减成无害的 0。注意 `block_tables` **没有清零**，只覆盖前 `bs` 行 —— 因为多出的行读取由 `context_lens=0` 挡住。

⑥ **`outputs[:bs]`**：图里算的是整桶（比如 32 行），只取前 `bs` 行送进 LM head。所以"多算的行"不是完全白费 —— 它们的 hidden_states 被算出来又丢掉。

## 4. 输入拷贝是 H2D 的显性成本

②④ 五条赋值是**从 CPU 张量到静态 buffer 的拷贝**。`prepare_decode` 已经把张量建在 `pin_memory=True` 的锁页内存上（`model_runner.py:182-185`），配合 `.cuda(non_blocking=True)` 走异步传输 —— 这是为图重放做的铺垫：**锁页内存 + 静态 buffer = 只有 memcpy，没有分配。**

代价：五份 `[512]` / `[512, 16]` 的 buffer 常驻显存，很小（几 MB），可以忽略。

## 5. 为什么 prefill 不录

三个原因：

1. **形状不定**：`cu_seqlens` 的长度 = `num_seqs + 1`，且每步的 `seqlen_q` 不同（分块预填）；FlashAttention 的 varlen 接口会把 `cu_seqlens` 当设备指针传进去，图无法记录"指针指向的数据变化"。
2. **前缀缓存分支**：`block_tables` 有时为 `None`、有时是张量（`model_runner.py:162-163`），控制流不同就不能复用同一张图。
3. **不值得**：prefill 一步算几千 token，计算时间远超 launch 开销，图带来的收益 <5%。

`enforce_eager=True`（`example.py:9`）就是"全不录"，用于调试或显存紧张。

## 6. 三个坑

1. **`reversed` 顺序不能动**：改了会让 pool 变小，可能出现重放时地址错乱（症状是随机输出或非法访问，很难定位）。
2. **捕获期间的任何"首次"都会失败**：比如在 `with torch.cuda.graph` 里第一次触发 `torch.compile`、第一次 `cudaMalloc`，都会报 `operation not permitted` 类错误。这叫"捕获要热身"。
3. **随机数在图里**：`Sampler` 用了 `exponential_()`。图重放会复用捕获时记录的 RNG 指令序列，随机性来自 RNG 状态推进 —— 也就是说**重放的随机性取决于图的执行次数与顺序，不同桶之间不共享 RNG 流**。想做到严格可复现需要在图外注入种子，本项目没有这个能力（`docs/12` 第 6 节）。

## 7. 显存与桶数的权衡

| 桶数（TP=1，512 上限） | 捕获耗时 | 显存（共享 pool） | 覆盖粒度 |
| --- | --- | --- | --- |
| 35（本实现） | 数十秒 | 一份池 | bs≤8 精确，其余每 16 |
| 只用 [1,2,4,8,16] | 快 | 更小 | 大 batch 全走 eager |
| 128 档（每 4 一档） | 很慢 | 略大 | 更精确 |

`graph_bs` 是唯一旋钮（`model_runner.py:234`），改它不影响正确性，只影响"浪费率"与启动时间。

## 8. 自测

1. CUDA Graph 省的是计算还是 launch？为什么 decode 受益、prefill 不受益？
2. `slot_mapping.fill_(-1)` 的目的是什么？kernel 端怎么配合？
3. `bs=17` 时实际跑的是哪个桶？浪费多少算力？
4. 为什么必须 `reversed` 录制？从小到大会怎样？
5. 捕获前那次"热身前向"具体排除了哪些"首次"？
6. `context_lens` 加了 `zero_()`，`block_tables` 却没加，为什么不会出问题？

## 下一课

`docs/14-性能基准与实验方法.md`：README 那张表怎么读，以及自己复现时该控制哪些变量。
