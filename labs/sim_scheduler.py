"""实验 2：用仓库原码跑 Scheduler（纯 CPU），观察三件在文档里说不清的事：

  1. prefill 优先 → 只要 waiting 队列非空，decode 一个 token 都发不出去（decode 饥饿）；
  2. chunked prefill 只对"队首"生效：后面的长 prompt 会被整条卡住，而不是切块；
  3. KV 不够时抢占是 LIFO（pop 队尾），受害者被清空前缀、回到 waiting 队首，
     下一轮从 0 重算（不是从上次进度续算）。

Scheduler 本身不碰 torch，只有 Config 依赖 transformers —— 用一个鸭子类型的桩替掉。
运行：~/.venv-nanovllm-lab/bin/python labs/sim_scheduler.py
"""
import os
import sys
import types
from pathlib import Path

REPO = Path(os.environ.get("NANOVLLM_SRC", Path.home() / "nano-vllm")).expanduser().resolve()

pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(REPO / "nanovllm")]
sys.modules["nanovllm"] = pkg
cfg_stub = types.ModuleType("nanovllm.config")
cfg_stub.Config = object
sys.modules["nanovllm.config"] = cfg_stub

from nanovllm.engine.scheduler import Scheduler      # noqa: E402  仓库原码
from nanovllm.engine.sequence import Sequence        # noqa: E402  仓库原码


class StubConfig:
    """只实现 Scheduler.__init__ 真正读取的五个字段（scheduler.py:11-15）。"""

    def __init__(self, max_num_seqs=8, max_num_batched_tokens=16384,
                 eos=-1, kvcache_block_size=256, num_kvcache_blocks=64):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.kvcache_block_size = kvcache_block_size
        self.num_kvcache_blocks = num_kvcache_blocks


def build(**kw):
    cfg = StubConfig(**kw)
    Sequence.block_size = cfg.kvcache_block_size
    Sequence.counter = __import__("itertools").count()   # 每次实验重置 seq_id
    return Scheduler(cfg)


def step(sched, sched_seqs=None, log=None):
    seqs, is_prefill = sched.schedule()
    if is_prefill:
        # prefill 阶段每个 seq 可能只算了一部分；给它一个假 token 推进 postprocess
        n = sum(s.num_scheduled_tokens for s in seqs)
        sched.postprocess(seqs, [0] * len(seqs), True)
        kind = f"PREFILL({n}tok)"
    else:
        sched.postprocess(seqs, [0] * len(seqs), False)
        kind = f"DECODE(bs={len(seqs)})"
    return kind, seqs


def dump(sched, label):
    rows = []
    for s in list(sched.waiting) + list(sched.running):
        rows.append(f"#{s.seq_id}:{s.status.name.lower()} {s.num_cached_tokens}/{s.num_tokens}")
    print(f"  [{label}] waiting={len(sched.waiting)} running={len(sched.running)} | " + "  ".join(rows))


def case1_prefill_starvation():
    print("\n=== case1: prefill 优先 → decode 饥饿 ===")
    sched = build(max_num_batched_tokens=8, kvcache_block_size=4, num_kvcache_blocks=64)
    sched.add(Sequence(list(range(30))))      # A: 30 token
    sched.add(Sequence(list(range(30))))      # B: 30 token
    for i in range(10):
        kind, seqs = step(sched)
        print(f"step{i}: {kind:<14}", end="")
        dump(sched, "")
        if not sched.waiting and i > 8:
            break
    print("结论：A 还在做 prefill 时，B 一次也没 decode；只要 waiting 非空，schedule() 直接返回 prefill。")


def case2_chunked_only_head():
    print("\n=== case2: chunked prefill 只给队首 ===")
    sched = build(max_num_batched_tokens=20, kvcache_block_size=4, num_kvcache_blocks=64)
    sched.add(Sequence(list(range(10))))      # A: 10 token，能塞满配额
    sched.add(Sequence(list(range(100))))     # B: 100 token，塞不下剩余配额
    seqs, is_prefill = sched.schedule()
    print(f"  第一次 schedule：{[(s.seq_id, s.num_scheduled_tokens) for s in seqs]} "
          f"is_prefill={is_prefill}")
    print("结论：B 需要的 100 token 超过剩余配额且 scheduled_seqs 非空 → break（scheduler.py:42），"
          "B 整条等待，不会被切块。")


def case3_preemption():
    print("\n=== case3: KV 不足时的抢占是 LIFO + 全量重算 ===")
    sched = build(max_num_batched_tokens=256, kvcache_block_size=4, num_kvcache_blocks=4)
    sched.add(Sequence(list(range(8))))       # A: 8 token = 2 块
    sched.add(Sequence(list(range(8))))       # B: 8 token = 2 块
    kind, _ = step(sched)
    print(f"  预填后：{kind}，free_blocks={len(sched.block_manager.free_block_ids)}")
    dump(sched, "prefill 完")
    for i in range(4):
        kind, seqs = step(sched)
        print(f"  step{i}: {kind}", end="")
        dump(sched, "")
    print("结论：第一次 decode 就需要新块（8 token 预填后 append 到 9，9%block_size==1），free_blocks=0 → "
          "can_append 为假 → preempt(running.pop()) 摘掉队尾的 B，B 的 num_cached_tokens 归零、回到 waiting 队首。")
    print("      注意 B 之后一直卡在 waiting：A 继续吃块，free_blocks 不足以让 B 重新 allocate，"
          "nano-vllm 没有 swap/offload 兜底，被抢占的长序列可能长期饥饿。")


def case4_batch_limit():
    print("\n=== case4: max_num_seqs 截断 ===")
    sched = build(max_num_seqs=2, max_num_batched_tokens=256,
                  kvcache_block_size=4, num_kvcache_blocks=64)
    for _ in range(5):
        sched.add(Sequence(list(range(4))))
    seqs, _ = sched.schedule()
    print(f"  5 条入队，一次最多调度 {len(seqs)} 条（max_num_seqs=2）。")




def case5_assert_edge():
    """边界：单条序列占满全部 KV 块、下一步又要新块 → decode 分支的 assert 会崩。

    scheduler.py:58-71 —— running 只剩自己时走 `else: self.preempt(seq); break`，
    跳过了 while-else 的调度体，scheduled_seqs 为空，最后 assert scheduled_seqs 失败。
    """
    print("\n=== case5: 单序列占满 KV → assert scheduled_seqs 崩 ===")
    sched = build(max_num_batched_tokens=256, kvcache_block_size=4, num_kvcache_blocks=2)
    sched.add(Sequence(list(range(8))))       # 8 token = 恰好 2 块，free=0
    kind, _ = step(sched)
    print(f"  预填：{kind}，free_blocks={len(sched.block_manager.free_block_ids)}，"
          f"len(seq)={len(sched.running[0])}（预填已 append 1 个 token）")
    try:
        step(sched)
        print("  没有崩：说明这条路径被别的东西挡住了，需重新分析")
    except AssertionError:
        print("  确认：AssertionError 从 scheduler.py:71 抛出。")
        print("  含义：KV 耗尽且 running 只剩一条时，抢占救不了它自己 —— 全项目没有"
              "\"序列比整个 KV 池还长\"的兜底检查。")


if __name__ == "__main__":
    case1_prefill_starvation()
    case2_chunked_only_head()
    case3_preemption()
    case4_batch_limit()
    case5_assert_edge()
