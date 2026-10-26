# 09 Attention：分页写入与变长读取

## 本课目标

- 看懂 20 行 Triton kernel：为什么"写 KV"要自己写，而"算 attention"不用；
- 理解 prefill 与 decode 为什么调用两个完全不同的 FlashAttention 接口；
- 掌握 `cu_seqlens` / `slot_mapping` / `block_table` 三个索引张量的分工；
- 知道"前缀缓存时把 k/v 换成整块 cache"这个技巧解决了什么问题。

## 1. 三个索引张量，各管一段

| 张量 | 形状 | 谁用 | 作用 |
| --- | --- | --- | --- |
| `cu_seqlens_q` | `[bs+1]` int32 | `flash_attn_varlen_func` | 变长批里每条序列的 query 起点（累积和） |
| `cu_seqlens_k` | `[bs+1]` int32 | `flash_attn_varlen_func` | 每条序列的 key 起点（**含前缀缓存部分**） |
| `slot_mapping` | `[num_tokens]` int32 | Triton 写 KV | 每个 token 的 KV 写到哪个 slot |
| `block_table` | `[bs, max_blocks]` int32 | decode / 前缀缓存 prefill | 逻辑块 → 物理块，`-1` 表示填充 |

`prepare_prefill` 里三者同时生成（`model_runner.py:129-170`），`prepare_decode` 只生成后两个（`model_runner.py:172-188`）。

### 1.1 `cu_seqlens` 的构造

```python
# model_runner.py:138-148
for seq in seqs:
    start = seq.num_cached_tokens          # ← 前缀缓存命中的部分，不算 q
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end                          # ← k 从 0 数到 end（含缓存）
    input_ids.extend(seq[start:end])
    positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

关键：**q 只包含"要算的"（`start` 之后），k 包含"全部"（0 到 `end`）**。当前缀缓存命中 8 块（2048 token）时，`seqlen_q` 可能只有 100，而 `seqlen_k` 是 2148 —— 这两个数不相等，正是"partial prefill"的标志：

```python
# model_runner.py:162-163
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

**只有存在前缀缓存时才准备 `block_table`**，否则不传（省一次 H2D 拷贝）。这个判断用"k 比 q 长"来间接表达"有缓存"，很省事但也很隐晦。

### 1.2 `slot_mapping` 的构造

```python
# model_runner.py:151-161
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size     # 第一块从中间开始
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

即"把这段 token 逐个映射到物理 slot"。注意 `end` 用了**向下取整再加一**的写法（`(end + bs - 1) // bs`）来覆盖部分填充的尾块。

decode 时只有一行（`model_runner.py:181`）：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

最后一个 token 写在"最后一块的最后一个已用位置"。

## 2. Triton kernel：为什么写 KV 要自己写

```python
# attention.py:10-30
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                       # 一个 program 处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                        # ← CUDA Graph 的 padding
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**它只做一次 scatter：把 `[N, heads, dim]` 的连续 K/V 写到 `[slot]` 指定位置的页里。**

为什么不用 `k_cache[slots] = k`（PyTorch 高级索引）？三个原因：

1. **索引是"页号 × 256 + 偏移"算出来的**，是一条不连续的 scatter，PyTorch 高级索引会先构造 index tensor 再做一次 gather/scatter，多一轮显存访问；
2. **`slot == -1` 的 padding**（CUDA Graph 里 batch 不足一个桶时补 -1，`model_runner.py:206`）需要"跳过"，PyTorch 写不出来；
3. 每行 `D = num_heads * head_dim` 连续，**一个 program 一个 token** 的写法访存完全 coalesced。

`store_kvcache` 这层 wrapper 的四个 assert（`attention.py:36-39`）锁定了布局契约：

```python
assert key.stride(-1) == 1 and value.stride(-1) == 1      # 最后一维连续
assert key.stride(1) == head_dim and value.stride(1) == head_dim   # 头之间紧挨着
assert k_cache.stride(1) == D and v_cache.stride(1) == D  # 页内 token 之间紧挨着
assert slot_mapping.numel() == N                          # 一一对应
```

含义：**q/k/v 必须是 `[N, heads, dim]` 且内存连续**（`qkv.split` + `.view` 之后满足，`qwen3.py:78-81`），**cache 必须是页内连续**（`torch.empty` 一次性分配满足，`docs/08`）。这两条一旦破，kernel 读到的是错位数据，不会报错。

## 3. 两条完全不同的读路径

```python
# attention.py:59-75
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)   # ① 先写
    if context.is_prefill:
        if context.block_tables is not None:    # ② 有前缀缓存
            k, v = k_cache, v_cache             # ← 换成整块 cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=..., cu_seqlens_q=...,
                                   max_seqlen_k=..., cu_seqlens_k=...,
                                   softmax_scale=self.scale, causal=True,
                                   block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens,
                                    block_table=context.block_tables,
                                    softmax_scale=self.scale, causal=True)
    return o
```

### 3.1 prefill 的两个分支

| 情况 | k/v 用什么 | 为什么 |
| --- | --- | --- |
| 无前缀缓存 | 本步新算的 `k, v`（`[N, heads, dim]`） | 全部 token 都在本步里，无需读 cache |
| 有前缀缓存 | **整个 `k_cache, v_cache`**（`[blocks, 256, heads, dim]`） | 前 2048 token 的 KV 只在页里，不在本步的 k/v 里 |

第二个分支是本节最值得琢磨的地方：把 **4 维的整块 cache** 直接当 `k` 传给 `flash_attn_varlen_func`，靠 `block_table` 把"逻辑位置"翻译成"物理页"。FlashAttention 的 varlen + block_table 组合接口就是为这个场景设计的 —— 否则就得先把缓存 gather 成连续张量（多一次全量拷贝，几 MB 到几百 MB）。

### 3.2 decode：单 token 查询

```python
o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=context.context_lens, block_table=...)
```

- `q.unsqueeze(1)`：`[bs, heads, dim]` → `[bs, 1, heads, dim]`，因为接口要 `(batch, seqlen, heads, dim)`；
- `cache_seqlens`：每条序列当前的 KV 长度，决定"读到第几个槽"（不是读满 256）；
- `block_table`：`[bs, max_blocks]`，页号索引；
- **KV 已经在①写进 cache 了**，所以这里不再传 k/v，接口内部直接读页。

`causal=True` 在 decode 时（q_len=1）自然满足 —— 1 个 query 看全部 key 是合法的因果掩码。

## 4. 一个 batch 里的三个视图同时存在

```
prepare_decode 产出：
  input_ids    [bs]                     ← 每序列 1 个 token
  positions    [bs]                     ← len(seq)-1
  slot_mapping [bs]                     ← 写 KV 的位置
  context_lens [bs]                     ← 读 KV 的长度
  block_tables [bs, max_blocks]         ← 页映射（不足补 -1）

模型里：
  embed → 28 × (RMSNorm → QKV → RoPE → Attention → O → RMSNorm → MLP)
                                    │
                                    ├─ store_kvcache(k, v, slot_mapping)   ← 写：新 token 的 KV
                                    └─ flash_attn_with_kvcache(...)        ← 读：0..len-1 的 KV
```

**同一个 attention 调用里既写又读**，写的是"自己这一步的 KV"，读的是"包括自己刚才写进去的那一份"。顺序不能反（先写后读）。这也是为什么 `store_kvcache` 必须在 `if` 之前无条件执行。

## 5. 三个坑

1. **`k_cache.numel()` 为 0 时静默跳过写**：warmup 阶段就是这样（`docs/08`）。如果有人把 warmup 的 `Sequence` 加上了 `block_table`，就会写进未初始化的显存 —— 不报错，结果不可复现。
2. **`block_table` 的 `-1` 填充**：decode 时 `prepare_block_tables` 用 `-1` 补齐（`model_runner.py:125`）；FlashAttention 要求页号非负，靠 `cache_seqlens` 限制读取范围才不会越界。所以 `context_lens` 错一位就是随机读显存。
3. **`positions` 在前缀缓存时从 `start` 开始**（`model_runner.py:144`）：`range(start, end)` 而不是 `range(0, seqlen_q)`。写错的话 RoPE 旋转角全错 —— 输出不会崩，只是质量悄悄下降。这是**最难发现**的一类 bug。

## 6. 为什么不用自己写 attention kernel

对比 PagedAttention 论文的 vLLM 实现（自研 CUDA kernel），nano-vllm 的选择是：

- **算 attention** 交给 FlashAttention（工业级、支持 varlen/paged/GQA/causal）；
- **只自己写"写 KV"这一步**（FlashAttention 的接口不负责往 page 里写，因为写是调度侧的知识）。

这个分工很聪明：**attention 的难点在数值稳定性和 tiling，KV 写入的难点在索引映射** —— 前者买现成，后者 20 行 Triton 解决。代价是必须接受 FlashAttention 的接口约定（`stride`、页大小、`-1` 填充），也就是第 5 节那些坑。

## 7. 自测

1. `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 为真说明什么？为什么这时才需要 `block_table`？
2. `slot_mapping` 里的 `-1` 是谁写进去的？kernel 怎么处理？
3. 哪一行决定 RoPE 的位置？如果写成 `range(0, seqlen_q)`，会出什么问题、会不会报错？
4. 为什么 decode 的 `q` 要 `unsqueeze(1)`，而 prefill 的不用？
5. 如果 `kv_cache` 改成"每块单独 `torch.empty`"，哪条 assert 会挂？为什么？

## 下一课

`docs/10-并行线性层与权重加载.md`：QKV 怎么在两张卡上切，`weight_loader` 又怎么把 safetensors 的碎片拼回去。
