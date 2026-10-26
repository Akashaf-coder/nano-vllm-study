# 03 Sequence 与状态机

## 本课目标

- 说清 `Sequence` 的 11 个字段分别归谁管；
- 弄清"两套状态"（`status` 三态 + `is_prefill` 布尔）为什么不能合并；
- 看懂 `__getstate__` / `__setstate__` 这一步"跨进程瘦身"省了什么、有没有风险。

## 1. 字段分工表

`sequence.py:18-31` 一共 11 个字段：

| 字段 | 初值 | 谁写 | 含义 |
| --- | --- | --- | --- |
| `seq_id` | 全局自增 | 构造 | 稳定 id，用于回填输出顺序 |
| `status` | `WAITING` | Scheduler | 三态：WAITING / RUNNING / FINISHED |
| `token_ids` | prompt 拷贝 | `append_token` | 全部 token（prompt + 生成） |
| `last_token` | `token_ids[-1]` | `append_token` | decode 输入，避免每步切片 |
| `num_tokens` | `len(prompt)` | `append_token` +1 | 总长度 |
| `num_prompt_tokens` | `len(prompt)` | 构造后不变 | prompt 长度 |
| `num_cached_tokens` | 0 | BlockManager / postprocess | 已经有 KV 的 token 数 |
| `num_scheduled_tokens` | 0 | Scheduler | 本 step 要算多少 token（**最易错的字段**） |
| `is_prefill` | `True` | Scheduler | 本步是 prefill 还是 decode |
| `block_table` | `[]` | BlockManager | 逻辑块 → 物理块 |
| `temperature` / `max_tokens` / `ignore_eos` | 来自 SamplingParams | 构造 | 采样三参数（拍平存储） |

三个派生属性：`num_blocks`（向上取整的块数）、`last_block_num_tokens`（最后一块填了多少）、`num_completion_tokens`（生成长度）。全部是 `@property`，不占存储。

注意 `num_tokens` 在**分块预填**时已经等于整个 prompt 长度，而 `num_cached_tokens` 才是在 gradual 增长的 —— 这两个数的差就是"还剩多少要算"：

```python
# scheduler.py:41
num_tokens = seq.num_tokens - seq.num_cached_tokens
```

## 2. 两套状态：`status` 与 `is_prefill`

```
                  add_request
                       │
                       ▼
              ┌────────────────┐
              │    WAITING     │  is_prefill=True, block_table=[]
              └───┬────────┬───┘
       prefill 完成 │        │ 被抢占（preempt）
   （cached==total）│        │
                   ▼        │
              ┌────────────────┐
              │    RUNNING     │  is_prefill=False（下一步起 decode）
              └───┬────────┬───┘
   eos/max_tokens  │        │ 再次被抢占
                   ▼        │
              ┌────────────────┐
              │   FINISHED     │  已 deallocate，从 running 移除
              └────────────────┘
```

为什么需要**两套**：

- `status` 管**生命周期**（在哪条队列里）；
- `is_prefill` 管**这一步怎么算**（q 长度是整个 prompt 还是 1）。

两者不等价，而且组合里有一个容易被忽略的：**`(RUNNING, True)`**。分块预填的最后一块调度完时，`scheduler.py:48-51` 先把 `status` 改成 `RUNNING`、把它从 waiting 挪进 running，**但 `is_prefill` 仍是 `True`** —— 因为这一步的模型前向确实还是 prefill（算最后几个 prompt token）。直到下一次走 decode 分支才被置 `False`（`scheduler.py:68`）。

所以三种组合都真实存在：

| 组合 | 场景 |
| --- | --- |
| `(WAITING, True)` | 刚入队，或分块预填未完成，或被抢占后回来 |
| `(RUNNING, True)` | 分块预填的最后一块（本步仍是 prefill） |
| `(RUNNING, False)` | 常规 decode |
| `(FINISHED, *)` | 已 `deallocate` 并移出 running |

一句话：`is_prefill` 是"**最近一次调度意图**"的缓存，不是请求的长期状态。想验证的话，把 `(status, is_prefill)` 每步打印出来，`labs/sim_scheduler.py` case1 的 step3 就能看到 `#0:running` 且本步是 `PREFILL`。

## 3. 跨进程瘦身：`__getstate__`

TP > 1 时 `seqs` 列表要被 pickle 后塞进 SharedMemory（`docs/07`）。如果每条序列都把完整 `token_ids` 序列化，256 条 × 1024 token 的 int 列表会有几 MB —— 而 SharedMemory 只有 1MB（`model_runner.py:43`）。所以作者只传"另一个 rank 用得到的东西"：

```python
# sequence.py:72-83
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
     self.num_scheduled_tokens, self.block_table, last_state) = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```

三个要点：

1. **prefill 必须传全量 `token_ids`**：非 0 rank 要用它算 `input_ids`（`model_runner.py:143` 的 `seq[start:end]`）。
2. **decode 只传 `last_token`**：decode 一步只需要最后一个 token，于是 `token_ids = []`。这就是 decode 阶段序列化体积从 O(n) 降到 O(1) 的地方。
3. **`block_table` 必须传**：它决定 KV 写入位置（`slot_mapping`），每个 rank 都要。

代价：**rank>0 的 `Sequence.token_ids` 是空的**。谁能用、谁不能用就变成隐式约定 —— 任何在非 0 rank 上访问 `seq.token_ids` 的代码都会静默拿到 `[]`。这是本项目最危险的一个隐式契约，扩功能时第一个撞。

另外 `status`、`seq_id`、`temperature` 等字段**都没传**：非 0 rank 用不到状态机（`postprocess` 只在 rank0 跑），也就不用采样参数。这种"按需序列化"省得干净，但字段增减时极易漏改。

## 4. 一个隐含假设：`block_size` 是类变量

```python
# sequence.py:15-16
class Sequence:
    block_size = 256
    counter = count()
```

`Sequence.block_size` 是**类属性**，由 `LLMEngine.__init__` 在构造时改写：

```python
# llm_engine.py:21
Sequence.block_size = config.kvcache_block_size
```

也就是说：`Sequence` 不能在 Config 之前构造，且**同一进程里只能有一个 block_size**。两个不同配置的 `LLM` 实例共存会互相踩。这类"全局可变状态"在本项目里出现三次（`Sequence.block_size`、`Sequence.counter`、`utils/context.py` 的 `_CONTEXT`），都是"少传一个参数"的代价。

## 5. 陷阱清单

1. **`num_tokens` vs `num_cached_tokens` 混用**：前者是"序列有多长"，后者是"KV 里已经有几位"。分块预填期间两者不等。
2. **在 rank>0 上读 `token_ids`**：空列表，不报错，直接算错。
3. **以为 `preempt` 保留了进度**：`deallocate` 会把 `num_cached_tokens` 归零（`block_manager.py:100`），抢占 = 全部重算（`docs/06`）。
4. **`len(seq)` 返回 token 数而不是块数**：`__len__` 返回 `num_tokens`（`sequence.py:33-34`），块数要用 `seq.num_blocks`。`block_manager.py:104` 的 `len(seq) % block_size == 1` 正是靠这个语义。
5. **`last_token` 在 prefill 后被复用**：prefill 完成时会 `append_token`，所以 `last_token` 变成"刚生成的第一个 token" —— decode 第一步的输入就是它。

## 6. 自测

1. 一条 300 token 的 prompt，`block_size=256`，`num_blocks` 是多少？`last_block_num_tokens` 是多少？（2；44）
2. decode 阶段一条序列被序列化后，体积和序列长度有关吗？为什么？
3. `status` 和 `is_prefill` 各回答什么问题？举出一个"RUNNING 且 is_prefill=True"的场景。
4. 为什么 `Sequence` 把 `temperature` 拍平存储，而不是持有 `SamplingParams` 对象？（提示：看 `sampling_params.py` 的 `slots=True` 和序列化成本）

## 下一课

`docs/04-Scheduler两队列调度.md`：92 行的调度器，三个 while 循环决定了一切。
