# 04 Scheduler：两队列调度

## 本课目标

- 把 92 行调度器拆成"三段结构"，每段只有一个职责；
- 理解 `max_num_seqs` / `max_num_batched_tokens` 两个上限分别在哪儿生效；
- 知道 `can_allocate == -1` 时为什么是 `break` 而不是 `continue`（队首阻塞）；
- 亲眼看一次 decode 饥饿、一次 LIFO 抢占、以及一个**会 assert 崩溃的边界**（本机已复现）。

## 1. 数据结构：只要两条 deque

```python
# scheduler.py:16-17
self.waiting: deque[Sequence] = deque()
self.running: deque[Sequence] = deque()
```

没有 swapped 队列（对应 vLLM 的三队列），因为**没有 swap**。`is_finished()` 就是"两个都空"。

还有两个上限，来自 Config：

| 参数 | 默认 | 作用点 |
| --- | ---: | --- |
| `max_num_seqs` | 512 | 一次 step 最多几条序列 |
| `max_num_batched_tokens` | 16384 | 一次 step 最多算多少 token（只对 prefill 有意义） |

## 2. 三段结构

```python
# scheduler.py:25-73（删掉注释后的骨架）
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # 段 1：prefill（把 waiting 里的往 running 搬）
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        ...
    if scheduled_seqs:
        return scheduled_seqs, True      # ← 段 2：只要有 prefill，直接返回

    # 段 3：decode（只有 waiting 空了才会到这里）
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        ...
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

**段 2 的那两行是整个调度哲学的核心**：一次 step 要么全是 prefill，要么全是 decode，**永不混合**。vLLM V1 允许同批混合（prefill 填满剩余配额的同时带上 decode），nano-vllm 不做。

## 3. 段 1：prefill 循环逐行

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]                       # ① 只看队首，不遍历
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:                     # ② 还没分配过 KV
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break                               # ③ KV 不够 → 整批停下（队首阻塞）
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                       # ④ 已分配但没预填完（分块预填/被抢占回来）
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs:
        break                                   # ⑤ 只有队首能分块
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)   # ⑥ 本步算多少
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING      # ⑦ 这一步能算完 → 转入 running
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
```

七个点各有一条结论：

① **严格 FIFO**：`self.waiting[0]` 只看队首，没有按长度排序、没有优先级。长 prompt 排在最前面的代价是"整批等它"。

③ **`break` 不是 `continue`**：一旦队首的 KV 分配失败，后面即使有短序列也不调度了。这叫**队首阻塞（head-of-line blocking）**，换来的是"严格保序"。`case3` 实测里 B 被抢占后一直卡在 waiting 正是这条：A 占着块，B 分不到，整批 decode 继续跑。

⑤ **`and scheduled_seqs` 是整个项目最值得记住的一个条件**：`remaining < num_tokens` 本来是"分块"的触发条件，但只有 `scheduled_seqs` 非空（即已经有别的序列占了配额）时才 `break`；**队首自己永远可以分块**。这就是"chunked prefill 只对队首生效"：

```
配额 20：  [A:10 能塞下] [B:100 塞不下]
           step:  A 全进（10）, 剩 10
                  B 需要 100 > 10 且 scheduled_seqs 非空 → break
           → B 整条等待，不被切成 10+10+…  （labs case2 实测）
```

⑥ **`num_scheduled_tokens` 是"本步意图"**：`min(要算的, 剩余配额)`。它会在 `postprocess` 里被累加进 `num_cached_tokens` 后清零 —— 一个字段复用"计划"和"已执行"两个语义，是这个项目最容易读错的地方。

## 4. 段 3：decode 循环

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()                       # ① 出队
    while not self.block_manager.can_append(seq):      # ② 需不需要新块
        if self.running:
            self.preempt(self.running.pop())           # ③ LIFO 抢队尾
        else:
            self.preempt(seq)                          # ④ 只剩自己 → 抢占自己
            break
    else:                                              # ⑤ 内层 while 正常结束才调度
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))      # ⑥ 保持原顺序
```

`can_append` 的判据只有一行（`block_manager.py:104`）：

```python
def can_append(self, seq): return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`(len(seq) % block_size == 1)` 是**布尔当天数用**：只有"刚跨进新块的第一位"才需要新块（`>= True` 等价于 `>= 1`）。比如 `block_size=4`、长度 8→9 时 `9%4==1`，就需要 1 个新块。

⑥ `extendleft(reversed(...))` 是为了**把本轮没轮到的顺序还原**：`popleft` 打乱了，`reversed` + `extendleft` 让它们在队首按原序回去。

## 5. 三个实测行为

### 5.1 prefill 优先 → decode 饥饿（`labs/sim_scheduler.py` case1）

`max_num_batched_tokens=8`，A、B 各 30 token：

```
step0: PREFILL(8tok)   waiting=2 running=0 | #0:waiting 8/30   #1:waiting 0/30
step1: PREFILL(8tok)   waiting=2 running=0 | #0:waiting 16/30  #1:waiting 0/30
step2: PREFILL(8tok)   waiting=2 running=0 | #0:waiting 24/30  #1:waiting 0/30
step3: PREFILL(6tok)   waiting=1 running=1 | #0:running 30/31  #1:waiting 0/30
step4: PREFILL(2tok)   waiting=0 running=2 | #0:running 30/31  #1:running 30/31
step5: DECODE(bs=2)    waiting=0 running=2 | #0:running 31/32  #1:running 31/32
```

三点观察：

1. A 从 8→16→24 token 的整整三步里，**B 一个 token 都没算**；A 转成 running 之后也不 decode，因为 waiting 里还有 B（段 2 提前返回）。
2. step3 那行 `#0:running 30/31` —— 预填完 30 token 的同时 append 了第一个输出 token（`docs/02` 阶段 6）。
3. 这不只是"有点慢"：**只要提交的 prompt 总长超过一个 step 的配额，已经在跑的序列就完全停住**。在线服务里这就是长尾延迟归零（好）与 tokens/s 归零（坏）同时发生。

### 5.2 chunked prefill 只给队首（case2）

```
第一次 schedule：[(0, 10)]  is_prefill=True
```

A（10 token）吃满配额，B（100 token）整条等待，**不会被切块**。所以"chunked prefill"在这个项目里是**为长 prompt 的队首服务的**，不是为吞吐服务的。

### 5.3 抢占是 LIFO，且可能救不了自己（case3 / case5）

case3（4 个块、每条序列 2 块）实测：

```
[prefill 完] waiting=0 running=2 | #0:running 8/9  #1:running 8/9
step0: DECODE(bs=1)  waiting=1 running=1 | #1:waiting 0/9  #0:running 9/10
```

第一步 decode 就需要新块（8→9，`9%4==1`），free=0 → `preempt(running.pop())` 摘掉**队尾**的 B → B 的 `num_cached_tokens` 归零、回到 waiting 队首。之后 B 一直没回来：A 还在吃块，free 不够让 B 重新 `allocate`，而项目没有 swap 兜底。

case5 是边界情况，**本机跑出崩溃**：

```
预填：PREFILL(8tok)，free_blocks=0，len(seq)=9
确认：AssertionError 从 scheduler.py:71 抛出。
```

原因：`num_kvcache_blocks=2`、一条 8 token 序列恰好占满；下一步要新块，running 里只剩它自己 → 走 `else: preempt(seq); break` → 跳出内层 while **但跳过了 `while-else` 的调度体** → `scheduled_seqs` 为空 → `assert scheduled_seqs` 崩。

结论有两层：

- 工程上：**没有"序列比整个 KV 池还长"的前置校验**，配置错了不是报清晰错误，而是 `AssertionError`（无消息）。真实场景由 `num_kvcache_blocks` 自动按显存算（`docs/08`）来避免，但小显存 + 长 prompt 仍然可能撞到。
- 阅读上：`while ... else` 这种写法让"抢占自己后不调度"这个分支极难看出来。想改的话，把 `else: preempt(seq); break` 改成 `raise RuntimeError(f"seq {seq.seq_id} 需要 {seq.num_blocks} 块，超过池容量")` 更有用。

## 6. 调度器整体流程图

```
             ┌───────────────── schedule() ─────────────────┐
             │                                              │
   waiting ──┼─▶ 段1: 队首能否 allocate？                     │
             │     ├─ 否 → break（队首阻塞）                  │
             │     └─ 是 → 分配 + 记 num_scheduled_tokens     │
             │              能算完？ ──否──▶ 留 waiting（分块）│
             │                 │是                          │
             │                 ▼                            │
             │              RUNNING（挪到 running 队尾）      │
             │                                              │
             │   有 scheduled？──是──▶ return (seqs, True)   │
             │        │否                                    │
             │        ▼                                     │
   running ──┼─▶ 段3: can_append？                          │
             │     ├─ 否 → 抢队尾 / 只剩自己则抢自己          │
             │     └─ 是 → num_scheduled_tokens=1, decode    │
             │                                              │
             │   return (seqs, False)                       │
             └──────────────────────────────────────────────┘
```

## 7. 与 vLLM 的差异（一句话版）

| 维度 | nano-vllm | vLLM V1 |
| --- | --- | --- |
| 混合批 | 不支持 | 支持（prefill 与 decode 同批） |
| chunked prefill | 只给队首 | 所有等待序列按配额切 |
| 抢占 | LIFO + 全量重算 | 可 swap 到 CPU、可按优先级 |
| 队列 | 2 条 | waiting/running/swapped + 优先级策略 |
| 准入 | 队首阻塞（break） | 可跳过队首（continue） |

## 8. 自测

1. 把 `max_num_batched_tokens` 从 16384 改成 512，吞吐和 TTFT 各会怎么变？为什么？
2. 为什么段 3 里 `popleft` 之后要 `extendleft(reversed(...))`？不这么做会怎样？
3. `can_append` 里的 `(len(seq) % block_size == 1)` 换成 `(len(seq) % block_size == 0)` 会错在哪？（提示：预填完会 append 一个 token）
4. case5 的崩溃在什么配置下会真实发生？怎么改代码能让它变成清晰报错？

## 下一课

`docs/05-BlockManager与前缀缓存.md`：块分配、哈希链、以及"为什么尾块永远不共享"。
