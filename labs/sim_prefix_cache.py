"""实验 1：用仓库原码验证 BlockManager 的前缀缓存语义（纯 CPU，不需要 GPU）。

nanovllm/engine/block_manager.py 只依赖 xxhash + numpy，因此可以在没有 CUDA 的机器上
原封不动地跑。这里只做一件事：把 nanovllm.config 换成一个空壳，绕开它触发的
transformers 依赖（以及 nanovllm/__init__.py 里对 torch 的导入）。

运行：~/.venv-nanovllm-lab/bin/python labs/sim_prefix_cache.py
"""
import os
import sys
import types
from pathlib import Path

# 源码仓库位置（默认 ~/nano-vllm，可用环境变量覆盖）
REPO = Path(os.environ.get("NANOVLLM_SRC", Path.home() / "nano-vllm")).expanduser().resolve()
if not (REPO / "nanovllm").is_dir():
    sys.exit(f"找不到源码：{REPO}/nanovllm —— 先 git clone https://github.com/GeeeekExplorer/nano-vllm，"
             f"或设 NANOVLLM_SRC=<path>")

# --- 绕开 torch/transformers：手工构造最小的包结构 ---
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(REPO / "nanovllm")]
sys.modules["nanovllm"] = pkg
cfg_stub = types.ModuleType("nanovllm.config")
cfg_stub.Config = object
sys.modules["nanovllm.config"] = cfg_stub

from nanovllm.engine.block_manager import BlockManager   # noqa: E402  仓库原码
from nanovllm.engine.sequence import Sequence            # noqa: E402  仓库原码

BS = 4
Sequence.block_size = BS


def new_seq(token_ids):
    seq = Sequence(token_ids)
    return seq


def run_prefill(bm, seq):
    """复刻 Scheduler.postprocess 的 prefill 分支（scheduler.py:81-87）：
    allocate → 记 num_scheduled_tokens → hash_blocks → 累加 num_cached_tokens。
    hash_blocks 依赖 num_scheduled_tokens，漏掉它哈希区间会退化成空（block_manager.py:111-113）。
    """
    n = bm.can_allocate(seq)
    assert n != -1, "KV 块不足"
    bm.allocate(seq, n)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return n


def show(title, bm):
    print(f"\n--- {title} ---")
    print("free:", len(bm.free_block_ids), "used:", sorted(bm.used_block_ids),
          "hash_entries:", len(bm.hash_to_block_id))


def case1_same_prompt():
    """两条完全相同的 prompt：第二条能复用几个块？"""
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = new_seq(list(range(12)))              # 12 token = 3 块
    run_prefill(bm, a)
    print(f"[case1] A blocks={a.block_table} cached_tokens={a.num_cached_tokens}")

    b = new_seq(list(range(12)))              # 逐字节相同
    n2 = bm.can_allocate(b)
    bm.allocate(b, n2)
    print(f"[case1] B 命中块数={n2} 复用 block={b.block_table[:n2]} "
          f"cached_tokens={b.num_cached_tokens} 需重算="
          f"{b.num_tokens - b.num_cached_tokens}")
    assert a.block_table[:n2] == b.block_table[:n2], "命中块应指向同一物理块"
    show("case1", bm)
    print("结论：num_blocks-1 → 尾块永不共享；12 token 只复用 2 块（8 token），最后 4 token 必须重算")


def case2_prefix_only():
    """共享前缀但尾部不同：只复用前缀块。"""
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = new_seq(list(range(10)))              # 3 块（最后一块只填 2/4）
    run_prefill(bm, a)
    b = new_seq(list(range(8)) + [99, 98])    # 前 8 token 相同，尾不同
    n2 = bm.can_allocate(b)
    print(f"[case2] A cached={a.num_cached_tokens} / B 命中 {n2} 块 = {n2*BS} token")
    assert n2 * BS == 8, n2
    show("case2", bm)


def case3_release_invalidates():
    """释放后哈希索引还在，靠 _allocate_block 兜底清理。"""
    bm = BlockManager(num_blocks=4, block_size=BS)
    a = new_seq(list(range(8)))
    run_prefill(bm, a)
    print(f"[case3] 分配后 hash_entries={len(bm.hash_to_block_id)}")
    bm.deallocate(a)
    print(f"[case3] 释放后 hash_entries={len(bm.hash_to_block_id)} "
          f"(deallocate 只减 ref_count，不删索引)")
    # 新序列复用同一批物理块时，_allocate_block 会把陈旧索引清掉
    c = new_seq(list(range(8)))
    n2 = bm.can_allocate(c)
    print(f"[case3] 新序列仍然命中 {n2} 块（索引还在、物理 KV 也还在）")
    d = new_seq([7] * 8)                      # 不命中的新内容
    n3 = bm.can_allocate(d); bm.allocate(d, n3)
    print(f"[case3] 冲突分配后 hash_entries={len(bm.hash_to_block_id)}，"
          f"used={sorted(bm.used_block_ids)}")


def case4_can_allocate_tail():
    """can_allocate 的循环上界：range(num_blocks-1)。"""
    bm = BlockManager(num_blocks=8, block_size=BS)
    a = new_seq(list(range(4)))               # 正好 1 块
    run_prefill(bm, a)
    a2 = new_seq(list(range(4)))
    n2 = bm.can_allocate(a2)
    print(f"[case4] 单块序列的第二次请求命中 {n2} 块 → 必须整段重算")


def case5_block_size_matters():
    """块大小如何影响可复用比例：同样 12 token，块越大浪费越多。"""
    for bs in (4, 8, 16):
        Sequence.block_size = bs
        bm = BlockManager(num_blocks=64, block_size=bs)
        a = new_seq(list(range(12)))
        run_prefill(bm, a)
        b = new_seq(list(range(12)))
        n = bm.can_allocate(b)
        print(f"[case5] block_size={bs:2d} 命中 {n} 块 = {n*bs:2d} token / 12，"
              f"重算 {12 - n*bs} token")
    Sequence.block_size = BS


if __name__ == "__main__":
    case1_same_prompt()
    case2_prefix_only()
    case3_release_invalidates()
    case4_can_allocate_tail()
    case5_block_size_matters()
    print("\n全部断言通过：本机跑的是仓库原码，不是复述。")
