# 本机实验真实输出

环境：

```
Host: macOS 27.0.1 / Apple M5（无 CUDA）
Python: Python 3.14.7（仅装 numpy + xxhash，无 torch/transformers）
源码: ~/nano-vllm @ bb823b3
复现: bash labs/run_labs.sh
```

```text
== 源码：/Users/menglinghuan/nano-vllm（bb823b3）
== Python：/Users/menglinghuan/.venv-nanovllm-lab/bin/python

################ sim_prefix_cache ################
[case1] A blocks=[0, 1, 2] cached_tokens=12
[case1] B 命中块数=2 复用 block=[0, 1] cached_tokens=8 需重算=4

--- case1 ---
free: 12 used: [0, 1, 2, 3] hash_entries: 3
结论：num_blocks-1 → 尾块永不共享；12 token 只复用 2 块（8 token），最后 4 token 必须重算
[case2] A cached=10 / B 命中 2 块 = 8 token

--- case2 ---
free: 13 used: [0, 1, 2] hash_entries: 2
[case3] 分配后 hash_entries=2
[case3] 释放后 hash_entries=2 (deallocate 只减 ref_count，不删索引)
[case3] 新序列仍然命中 1 块（索引还在、物理 KV 也还在）
[case3] 冲突分配后 hash_entries=2，used=[2, 3]
[case4] 单块序列的第二次请求命中 0 块 → 必须整段重算
[case5] block_size= 4 命中 2 块 =  8 token / 12，重算 4 token
[case5] block_size= 8 命中 1 块 =  8 token / 12，重算 4 token
[case5] block_size=16 命中 0 块 =  0 token / 12，重算 12 token

全部断言通过：本机跑的是仓库原码，不是复述。

################ sim_scheduler ################

=== case1: prefill 优先 → decode 饥饿 ===
step0: PREFILL(8tok)   [] waiting=2 running=0 | #0:waiting 8/30  #1:waiting 0/30
step1: PREFILL(8tok)   [] waiting=2 running=0 | #0:waiting 16/30  #1:waiting 0/30
step2: PREFILL(8tok)   [] waiting=2 running=0 | #0:waiting 24/30  #1:waiting 0/30
step3: PREFILL(6tok)   [] waiting=1 running=1 | #1:waiting 0/30  #0:running 30/31
step4: PREFILL(2tok)   [] waiting=0 running=2 | #0:running 30/31  #1:running 30/31
step5: DECODE(bs=2)    [] waiting=0 running=2 | #0:running 31/32  #1:running 31/32
step6: DECODE(bs=2)    [] waiting=0 running=2 | #0:running 32/33  #1:running 32/33
step7: DECODE(bs=2)    [] waiting=0 running=2 | #0:running 33/34  #1:running 33/34
step8: DECODE(bs=2)    [] waiting=0 running=2 | #0:running 34/35  #1:running 34/35
step9: DECODE(bs=2)    [] waiting=0 running=2 | #0:running 35/36  #1:running 35/36
结论：A 还在做 prefill 时，B 一次也没 decode；只要 waiting 非空，schedule() 直接返回 prefill。

=== case2: chunked prefill 只给队首 ===
  第一次 schedule：[(0, 10)] is_prefill=True
结论：B 需要的 100 token 超过剩余配额且 scheduled_seqs 非空 → break（scheduler.py:42），B 整条等待，不会被切块。

=== case3: KV 不足时的抢占是 LIFO + 全量重算 ===
  预填后：PREFILL(16tok)，free_blocks=0
  [prefill 完] waiting=0 running=2 | #0:running 8/9  #1:running 8/9
  step0: DECODE(bs=1)  [] waiting=1 running=1 | #1:waiting 0/9  #0:running 9/10
  step1: DECODE(bs=1)  [] waiting=1 running=1 | #1:waiting 0/9  #0:running 10/11
  step2: DECODE(bs=1)  [] waiting=1 running=1 | #1:waiting 0/9  #0:running 11/12
  step3: DECODE(bs=1)  [] waiting=1 running=1 | #1:waiting 0/9  #0:running 12/13
结论：第一次 decode 就需要新块（8 token 预填后 append 到 9，9%block_size==1），free_blocks=0 → can_append 为假 → preempt(running.pop()) 摘掉队尾的 B，B 的 num_cached_tokens 归零、回到 waiting 队首。
      注意 B 之后一直卡在 waiting：A 继续吃块，free_blocks 不足以让 B 重新 allocate，nano-vllm 没有 swap/offload 兜底，被抢占的长序列可能长期饥饿。

=== case4: max_num_seqs 截断 ===
  5 条入队，一次最多调度 2 条（max_num_seqs=2）。

=== case5: 单序列占满 KV → assert scheduled_seqs 崩 ===
  预填：PREFILL(8tok)，free_blocks=0，len(seq)=9（预填已 append 1 个 token）
  确认：AssertionError 从 scheduler.py:71 抛出。
  含义：KV 耗尽且 running 只剩一条时，抢占救不了它自己 —— 全项目没有"序列比整个 KV 池还长"的兜底检查。

################ sim_sampler ################
=== 分布正确性（n=200k）===
5 类均匀        目标分布=[0.2 0.2 0.2 0.2 0.2]  经验分布=[0.201 0.199 0.198 0.201 0.201]  最大偏差=0.0018
5 类偏斜        目标分布=[0.5  0.25 0.15 0.08 0.02]  经验分布=[0.501 0.251 0.15  0.079 0.02 ]  最大偏差=0.0015
教科书式         目标分布=[0.5  0.25 0.15 0.08 0.02]  经验分布=[0.5   0.25  0.149 0.08  0.02 ]  最大偏差=0.0005

=== 温度的效果：logits 除以 T 后再 softmax（sampler.py:9）===
T=0.3   p=[0.958 0.034 0.006 0.001]  最可能 token 命中率=0.959
T=1.0   p=[0.579 0.213 0.129 0.078]  最可能 token 命中率=0.579
T=2.0   p=[0.409 0.248 0.193 0.15 ]  最可能 token 命中率=0.403
T=10.0  p=[0.279 0.252 0.24  0.228]  最可能 token 命中率=0.274

=== 为什么 assert temperature > 1e-10（sampling_params.py:11）===
T→0 时 softmax 退化为 one-hot：p=[1. 0. 0. 0.] → argmax 恒定，greedy 应由单独的采样路径实现，作者选择直接禁止。

全部实验通过。数字来源见 labs/RUN_OUTPUT.md。
```
