# 05 BlockManager 与前缀缓存

## 本课目标

- 看懂 120 行里的四个数据结构，以及它们为什么必须同时存在；
- 讲清"链式哈希"如何把"前缀相同"变成"哈希相同"；
- 解释 `range(seq.num_blocks - 1)` 这个上界为什么必须减 1；
- 用本机实测数字说明块大小对复用率的影响。

## 1. 四个数据结构

```python
# block_manager.py:28-33
self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   # 全部物理块
self.hash_to_block_id: dict[int, int] = dict()                     # 哈希 → 块号
self.free_block_ids: deque[int] = deque(range(num_blocks))          # 空闲块（FIFO）
self.used_block_ids: set[int] = set()                              # 在用的块
```

单个 `Block`（`block_manager.py:8-23`）只有四个字段：

| 字段 | 用途 |
| --- | --- |
| `block_id` | 物理块号（也等于 KV 大张量里的第几块） |
| `ref_count` | 被几条序列共享；0 表示可回收 |
| `hash` | 链式哈希值，-1 表示"没有缓存价值" |
| `token_ids` | 这个块里的 token（**用于哈希碰撞时的二次校验**） |

`token_ids` 存在块里不是为了算哈希（哈希时直接用序列的 token），而是为了在命中时比对：

```python
# block_manager.py:66
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

哈希命中但 token 不等 = 碰撞，直接放弃这条链后面的所有块。**这是 64 位 xxhash 的安全网**，也说明作者认真想过碰撞。

## 2. 链式哈希：为什么前缀相同才能同哈希

```python
# block_manager.py:35-41
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

关键在 `prefix` 参数：**第 i 块的哈希 = H(第 i-1 块的哈希 ‖ 第 i 块的 token)**。

```
块0: h0 = H(tokens[0:256])
块1: h1 = H(h0 ‖ tokens[256:512])
块2: h2 = H(h1 ‖ tokens[512:768])
```

这就把"前缀"编进了哈希：两条序列只要在块 2 处不同，块 2 及之后的所有哈希都不同 —— **不需要额外记录"这是第几块"，哈希本身就编码了位置**。反过来，"哈希相同"就意味着"从开头到最后一块的 token 完全一致"，这正是前缀缓存要的语义。

对比一个常见错误实现：只对块内 token 做哈希。那样"块 1 内容相同但块 0 不同"的两条序列会误命中 —— KV 是依赖前文的，错了就会算出乱码。链式哈希是唯一能只靠一个整数保证正确性的做法。

## 3. `can_allocate`：唯一的"试探"函数

```python
# block_manager.py:58-73
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):          # ① 注意是 -1
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                # ② 链断了，后面的都不用看
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                  # ③ 命中的块若还在用，不额外占新块
    if len(self.free_block_ids) < num_new_blocks:
        return -1                                # ④ 块不够
    return num_cached_blocks                     # ⑤ 返回"能复用几块"
```

**① `range(num_blocks - 1)` 是本课最重要的一行**。尾块必须排除，因为：

- 尾块可能**没写满**（比如 300 token 的最后一块只有 44/256），内容会继续增长；
- 一旦继续写，它就不再是"那个哈希对应的 KV" —— 缓存会被自己污染。

所以规则是：**只有写满的块才可共享，而"写满"要等到下一个 token 落进新块才确定**。代价是每次 prefill 至少要重算最后一个块。

**③ 为什么只在 `used_block_ids` 里才减 `num_new_blocks`**：命中块有两种状态 —— 还被别人用着（`ref_count > 0`，直接增量引用即可，不耗新块），或者已经空闲（`ref_count == 0`，从 free 里摘出来，但那块本来就在 free 里，所以也不该算"新块"）。这段逻辑的写法有点绕，但效果是：`num_new_blocks` 只统计"必须从未使用区域新分配"的块数。

## 4. `allocate` 与 `hash_blocks`

```python
# block_manager.py:75-92
def allocate(self, seq, num_cached_blocks):
    h = -1
    for i in range(num_cached_blocks):            # 复用段：ref_count++ 或从 free 摘出
        ...
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())   # 新段
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

注意最后一行：**复用块直接把 `num_cached_tokens` 抬起来**，于是 `prepare_prefill` 里的 `start = seq.num_cached_tokens`（`model_runner.py:139`）会跳过这些 token —— 前缀缓存"省算力"就是在这里落地的。

`hash_blocks` 是反向操作：把本步**写满**的块登记进索引。

```python
# block_manager.py:110-120
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return                       # ① 一个块都没写满 → 什么都不做
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)       # ② 从上一块的 hash 续上
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id  # ③ 登记
```

① 那个 `if start == end: return` 是本机实验踩过的坑：`hash_blocks` 读的是 `num_scheduled_tokens`，如果调用方没先设它（我在 `labs/sim_prefix_cache.py` 的第一次尝试就漏了），哈希区间退化为空，**静默不登记任何块** —— 于是"前缀缓存看起来完全不工作"，且不报错。

## 5. 本机实测：尾块到底浪费多少

`labs/sim_prefix_cache.py`（跑的是仓库原码）输出：

```
[case1] A blocks=[0, 1, 2] cached_tokens=12
[case1] B 命中块数=2 复用 block=[0, 1] cached_tokens=8 需重算=4
结论：num_blocks-1 → 尾块永不共享；12 token 只复用 2 块（8 token），最后 4 token 必须重算

[case2] A cached=10 / B 命中 2 块 = 8 token          ← 前缀相同、尾部不同：只复用整块部分
[case4] 单块序列的第二次请求命中 0 块 → 必须整段重算

[case5] block_size= 4 命中 2 块 =  8 token / 12，重算 4 token
[case5] block_size= 8 命中 1 块 =  8 token / 12，重算 4 token
[case5] block_size=16 命中 0 块 =  0 token / 12，重算 12 token
```

两个可以直接引用的结论：

1. **同样的 12 token，块越大复用越差**：`block_size=16` 时一个块都没写满，前缀缓存完全失效。默认 256（`config.py:17`）意味着"两个请求前 256 token 完全相同"才有收益 —— 对话场景里 system prompt 通常够长，短 prompt 场景基本无效。
2. **单块序列永远零命中**（case4）：prompt 短于 256 token 时，前缀缓存一次都不会命中。这是 `range(num_blocks-1)` 的直接代价。

## 6. 没有淘汰：一个被低估的简化

`free_block_ids` 是 FIFO deque（`_deallocate_block` 用 `append`），`_allocate_block` 用 `popleft` —— **最早释放的块最先被重用**。没有 LRU、没有 TTL、没有容量水位。

这意味着：

- 高频复用的长 system prompt 的块，可能被一次突发流量冲掉（因为它是"最早释放"的那批之一）；
- 索引 `hash_to_block_id` 在 `deallocate` 时**不清除**（`block_manager.py:94-101` 只减 `ref_count`），所以释放后仍可能命中 —— 物理 KV 内容也没被清（`_deallocate_block` 只把块放回 free）。

`labs` case3 实测：

```
[case3] 分配后 hash_entries=2
[case3] 释放后 hash_entries=2 (deallocate 只减 ref_count，不删索引)
[case3] 新序列仍然命中 1 块（索引还在、物理 KV 也还在）
[case3] 冲突分配后 hash_entries=2，used=[2, 3]
```

**这是设计选择而非 bug**：块被释放但内容没被覆盖前，命中就是合法的（KV 还在）。真正的危险在于"块被重新分配写入新内容"，此时 `_allocate_block` 会做一次清理：

```python
# block_manager.py:47-48
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

它只在"索引确实指向这块"时才删，避免把别人的索引误删。这段是 120 行里最需要仔细读的三行。

## 7. 引用计数与 COW：这里其实没有 COW

`ref_count` 的作用只有共享与回收：

```
两条序列共享块 7：
  blocks[7].ref_count == 2
  序列 A 结束 → deallocate → ref_count 1（块留着）
  序列 B 结束 → deallocate → ref_count 0 → 回 free
```

**没有写时复制**。因为共享块永远是"已经写满、不会再写"的块（`range(num_blocks-1)` 保证了这点），所以不存在"两个序列要往同一块写不同内容"的情形。vLLM 里的 COW 是为"共享了未写满的块"服务的，nano-vllm 直接不共享未写满的块，问题在源头上消失。

一句话对比：**vLLM 用 COW 允许更激进的共享，nano-vllm 用"只共享写满的块"换掉了 COW 的复杂度。** 代价见第 5 节（尾块、单块序列）。

## 8. 生命周期图

```
请求到达
   │
   ▼
can_allocate()  ── 沿哈希链走到第一个不匹配的块
   │              命中 k 块 → 返回 k；块不够 → -1（调度器 break）
   ▼
allocate()      ── 前 k 块：ref_count++ / 从 free 摘出
   │               其余：_allocate_block()（popleft free + 清陈旧哈希索引）
   ▼
[模型前向，KV 通过 slot_mapping 写入物理块]
   │
   ▼
hash_blocks()   ── 本步写满的块 → 链式哈希 → 登记 hash_to_block_id
   │
   ▼
deallocate()    ── 逆序 ref_count--，归零则回 free（索引与内容保留）
```

## 9. 陷阱清单

1. **以为尾块也能复用**：必须重算，且单块序列零命中。
2. **忘了 `num_scheduled_tokens`**：`hash_blocks` 静默不登记，前缀缓存"看起来没生效"。
3. **以为 deallocate 会清 KV 内容**：不清，只改 `ref_count` 与 free 队列；靠"重新分配时覆盖"保证正确性。
4. **以为哈希只对块内 token**：是链式的，`prefix` 参与（否则会误命中）。
5. **调大 `block_size` 想省内存**：块越大，内部碎片越多（一个 300 token 的请求占 2 块 = 512 token 的 KV），且复用率下降。

## 10. 自测

1. 两条序列共享块 0–1，A 结束后 `ref_count` 是多少？什么时候块真正回 free？
2. `can_allocate` 为什么不能简单地对全部块做哈希查找？
3. 为什么 `hash_to_block_id` 在 `deallocate` 时不删？什么情况下必须删？
4. `block_size=256` 对一个平均 120 token 的 prompt 意味着什么？（占 1 块、零前缀复用、内部碎片 136 token）
5. 如果想让"单块短 prompt 也能复用"，最少要改哪几处？（提示：块大小与尾块规则是两个独立的旋钮）

## 下一课

`docs/06-抢占与重算.md`：`ref_count` 归零之后会发生什么，以及为什么"抢占"在这个项目里等于"从头再来"。
