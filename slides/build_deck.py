#!/usr/bin/env python3
"""构建 nano-vllm 深度解析 slides：把 SLIDES 注入 deck_template.html，输出自包含 HTML。

用法：
    python3 slides/build_deck.py
    # 再导出 PDF（需要 Chrome）：
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
      --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
      --virtual-time-budget=4000 --print-to-pdf="slides/nano-vllm-深度解析.pdf" \
      "file://$PWD/slides/nano-vllm-深度解析.html"
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOOT = "nano-vllm @ bb823b3 · 1450 行 · 16 lessons"

# (页眉, 标题, 正文 HTML)
SLIDES = [
    ("", "nano-vllm 深度解析",
     '''<p class="sub" style="margin-top:8px">1,450 行 Python，一个能跑的推理引擎<br>
    调度 · 分页 KV · 前缀缓存 · 张量并行 · CUDA Graph</p>
     <p class="sub" style="margin-top:26px;color:#4a5766">
     docs/ 16 节课 + labs/ 可跑实验 + 本 slides</p>'''),

    ("论点", "它把推理引擎削到只剩必要部件",
     '''<div class="row">
       <div class="col">
         <div class="card"><h3>有的</h3><p>连续批处理 · 分页 KV · 前缀缓存 ·
         TP · FlashAttention · CUDA Graph · 一行采样</p></div>
         <div class="card"><h3>没有的（grep = 0）</h3><p>top-k/top-p · 日志概率 ·
         流式 · HTTP 服务 · swap/offload · 量化 · 滑窗 · 投机解码 · LoRA</p></div>
       </div>
       <div class="col">
         <div class="card"><h3>一句话</h3><p>不是"简化版 vLLM"，是
         <b>把 vLLM 散在几十个文件的路径压到 1450 行</b>，
         每读懂一个函数，vLLM 里都有一个包了十层皮的近亲。</p></div>
       </div>
     </div>
     <p class="sub">分析对象：<b>GeeeekExplorer/nano-vllm</b> commit bb823b3（2026-04-26）v0.2.0</p>'''),

    ("规模", "1450 行，对比生产框架 39 倍",
     '''<table>
       <tr><th>项目</th><th class="num">文件</th><th class="num">行数</th><th>构成</th></tr>
       <tr><td>nano-vllm</td><td class="num">21</td><td class="num hl">1,450</td><td>Python 单语言</td></tr>
       <tr><td>DeepInfer（生产）</td><td class="num">285</td><td class="num">≈ 57,000</td><td>Python 54,929 + C++ 2,425 + TileLang</td></tr>
     </table>
     <div class="row" style="margin-top:22px">
       <div class="col"><div class="card"><h3>39 倍换来什么</h3>
       <p>不是更快的 decode 循环，而是：CPU 与 GPU 重叠、跨实例缓存、
       重算上界、专家并行、双硬件、服务面</p></div></div>
       <div class="col"><div class="card"><h3>1450 行里有假代码吗</h3>
       <p>没有。每一行都在跑。省略的都是"被选择不做"，不是"没写完"</p></div></div>
     </div>'''),

    ("边界", "六个边界：别拿它当生产引擎",
     '''<table>
       <tr><th>边界</th><th>证据</th></tr>
       <tr><td>单模型（只有 Qwen3）</td><td>model_runner.py:9 硬编码，无注册表</td></tr>
       <tr><td>只有 TP，无 EP/DP/PP</td><td>tensor_parallel_size 1–8，一卡一份全量</td></tr>
       <tr><td>只能离线批处理</td><td>generate 同步阻塞，无流式/无服务端</td></tr>
       <tr><td>调度粒度是"批"</td><td>一次 step 全 prefill 或全 decode</td></tr>
       <tr><td>KV 不可换出</td><td>grep swap/offload = 0，抢占即重算</td></tr>
       <tr><td>采样只有温度</td><td>grep top_k/top_p = 0，且禁止 T=0</td></tr>
     </table>
     <p class="sub" style="margin-top:14px">grep 覆盖 nanovllm/ + bench.py + example.py</p>'''),

    ("血统", "vLLM 骨架的重写，不是 fork",
     '''<table class="small">
       <tr><th>vLLM</th><th>nano-vllm</th><th>差异</th></tr>
       <tr><td>EngineCore + 前端</td><td>llm_engine.py（90 行）</td><td>同步、单进程</td></tr>
       <tr><td>Scheduler</td><td>scheduler.py（92 行）</td><td>两队列，无 swapped</td></tr>
       <tr><td>KVCacheManager</td><td>block_manager.py（120 行）</td><td>无 LRU 淘汰</td></tr>
       <tr><td>PagedAttention kernel</td><td>20 行 Triton + FlashAttention</td><td>不自己写 attention</td></tr>
       <tr><td>LinearMethod</td><td>linear.py（156 行）</td><td>无量化分支</td></tr>
       <tr><td>CUDAGraphRunner</td><td>capture_cudagraph</td><td>按 bs 分桶 + 共享 pool</td></tr>
     </table>'''),

    ("系统地图", "一张图看完",
     '''<pre>prompts ─▶ LLMEngine ─▶ Scheduler ────────────┐
                          waiting ⇄ running      │
                          + BlockManager         │
                       (分页 KV + 前缀哈希)       │
                                │ seqs, is_prefill
                                ▼                │
                        ModelRunner ─────────────┘
                   prepare_*（造 input/slot/block）
                                │
                        Qwen3ForCausalLM ×28
                     embed → layer → norm → lm_head
                       │              │
              Triton 写 KV      FlashAttention 读
                                │ logits（只算末位）
                             Sampler ─▶ token</pre>'''),

    ("生命周期", "七个阶段，每跳都有行号",
     '''<table>
       <tr><th>#</th><th>阶段</th><th>位置</th></tr>
       <tr><td>1</td><td>参数归一</td><td>llm_engine.py:66-68</td></tr>
       <tr><td>2</td><td>分词（<b>不加 chat template</b>）</td><td>llm_engine.py:44-45</td></tr>
       <tr><td>3</td><td>建 Sequence 入 waiting</td><td>llm_engine.py:46</td></tr>
       <tr><td>4</td><td>调度</td><td>scheduler.py:25-73</td></tr>
       <tr><td>5</td><td>GPU 前向</td><td>model_runner.py:214-220</td></tr>
       <tr><td>6</td><td>后处理（哈希/推进/结束）</td><td>scheduler.py:81-92</td></tr>
       <tr><td>7</td><td>收集 + 解码</td><td>llm_engine.py:54-55,84-89</td></tr>
     </table>'''),

    ("生命周期", "step() 用符号编码两件事",
     '''<pre>num_tokens = sum(s.num_scheduled_tokens for s in seqs) if is_prefill else -len(seqs)
                 └──── prefill：这一步算了多少 token ────┘      └ decode：-N 条各出 1 ─┘

if num_tokens &gt; 0:  prefill_throughput =  num_tokens / dt
else:               decode_throughput  = -num_tokens / dt</pre>
     <ul>
       <li>负数 = "decode 的 N 条序列"，不是笔误</li>
       <li>统计的是<b>引擎侧吞吐</b>，不是 TTFT/ITL —— Sequence 里没有时间戳</li>
     </ul>'''),

    ("生命周期", "第一个 token 在 prefill 那一步就出来了",
     '''<pre class="small">step0: PREFILL(8tok)  waiting=2 running=0 | #0:waiting 8/30  #1:waiting 0/30
step1: PREFILL(8tok)  waiting=2 running=0 | #0:waiting 16/30
step2: PREFILL(8tok)  waiting=2 running=0 | #0:waiting 24/30
step3: PREFILL(6tok)  waiting=1 running=1 | #0:running 30/31  ◀ 预填完 + append 首发 token
step4: PREFILL(2tok)  waiting=0 running=2 | #1:running 30/31
step5: DECODE(bs=2)   waiting=0 running=2 | #0:running 31/32</pre>
     <ul>
       <li><code>postprocess</code> 里"分块没算完 → continue"，最后一块才 append</li>
       <li>max_tokens=256 ⇒ 1 次 prefill + 255 次 decode，不是 256 次 decode</li>
     </ul>'''),

    ("Sequence", "11 个字段，谁写什么",
     '''<table class="small">
       <tr><th>字段</th><th>谁写</th><th>含义</th></tr>
       <tr><td>status</td><td>Scheduler</td><td>WAITING / RUNNING / FINISHED</td></tr>
       <tr><td>num_tokens</td><td>append_token</td><td>序列总长</td></tr>
       <tr><td>num_cached_tokens</td><td>BlockManager</td><td>已有 KV 的 token 数</td></tr>
       <tr><td>num_scheduled_tokens</td><td>Scheduler</td><td><b>本步意图</b>（复用"计划+已执行"）</td></tr>
       <tr><td>is_prefill</td><td>Scheduler</td><td>本步怎么算（q 是全长还是 1）</td></tr>
       <tr><td>block_table</td><td>BlockManager</td><td>逻辑块 → 物理块</td></tr>
     </table>
     <p class="sub">num_tokens − num_cached_tokens = 还剩多少要算</p>'''),

    ("Sequence", "两套状态，四种组合都真实存在",
     '''<pre>(status, is_prefill)
  (WAITING,  True)   刚入队 / 分块未完成 / 被抢占回来
  (RUNNING,  True)   ◀ 分块预填的最后一块，本步仍是 prefill
  (RUNNING,  False)  常规 decode
  (FINISHED,  ·  )   已 deallocate，移出 running

is_prefill 是"最近一次调度意图"的缓存，不是请求状态</pre>
     <p class="sub">scheduler.py:48-51 先改 status=RUNNING，is_prefill 要到下一次 decode 分支才被置 False（:68）</p>'''),

    ("Sequence", "跨进程瘦身：decode 只传 1 个 token",
     '''<pre>def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (num_tokens, num_prompt_tokens, num_cached_tokens,
            num_scheduled_tokens, block_table, last_state)</pre>
     <ul>
       <li>prefill：必须传全量 token_ids（非 0 rank 要算 input_ids）</li>
       <li>decode：只传 last_token，<b>体积 O(n) → O(1)</b></li>
       <li class="warn">代价：rank&gt;0 的 <code>seq.token_ids == []</code>，访问不报错、静默算错</li>
     </ul>'''),

    ("调度", "92 行，三段结构",
     '''<pre class="small">def schedule(self):
    scheduled_seqs = []
    # 段1：prefill（waiting → running）
    while self.waiting and len(scheduled_seqs) &lt; self.max_num_seqs:
        ...
    if scheduled_seqs:
        return scheduled_seqs, True        # ◀ 段2：有 prefill 就直接返回

    # 段3：decode（只有 waiting 空了才到这）
    while self.running and len(scheduled_seqs) &lt; self.max_num_seqs:
        ...
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False</pre>'''),

    ("调度", "prefill 循环的七个决定",
     '''<table class="small">
       <tr><th>行</th><th>代码</th><th>含义</th></tr>
       <tr><td>①</td><td>self.waiting[0]</td><td>严格 FIFO，不排序</td></tr>
       <tr><td>③</td><td>can_allocate == -1 → break</td><td><b>队首阻塞</b>：后面的短序列也不调度</td></tr>
       <tr><td>⑤</td><td>remaining &lt; n and scheduled_seqs</td><td>只有队首能分块，<b>B 被整条卡住</b></td></tr>
       <tr><td>⑥</td><td>num_scheduled_tokens = min(...)</td><td>计划量，postprocess 后清零</td></tr>
       <tr><td>⑦</td><td>cached + scheduled == num_tokens</td><td>本步能算完 → 转 RUNNING</td></tr>
     </table>
     <p class="sub">case2 实测：配额 20，A(10) 全进，B(100) 一等到底 → 不是 10+10+…</p>'''),

    ("调度", "decode 饥饿：waiting 非空，decode 就停摆",
     '''<pre class="small">=== case1: prefill 优先 → decode 饥饿 ===
step0: PREFILL(8tok)  waiting=2 running=0
step1: PREFILL(8tok)  waiting=2 running=0     ◀ A 在 prefill，B 一个 token 都没算
step2: PREFILL(8tok)  waiting=2 running=0
step3: PREFILL(6tok)  waiting=1 running=1     ◀ A 转 running，但仍不 decode
step4: PREFILL(2tok)  waiting=0 running=2
step5: DECODE(bs=2)   waiting=0 running=2     ◀ 直到 waiting 清空</pre>
     <ul>
       <li>根因：<code>if scheduled_seqs: return ..., True</code> 两行</li>
       <li>vLLM V1 允许同批混合（prefill 填满配额 + 带 decode），这里永不混合</li>
     </ul>'''),

    ("调度", "can_append：一行布尔当天数",
     '''<pre>def can_append(self, seq):
    return len(self.free_block_ids) &gt;= (len(seq) % self.block_size == 1)
                                          └────── 布尔 → True=1 / False=0

block_size=4，长度 8 → 9 时 9%4 == 1 → 需要 1 个新块</pre>
     <ul>
       <li>"刚跨进新块的第一位"才需要新块 —— 因为预填完成时会 append 一个 token</li>
       <li>换成 <code>== 0</code> 会多分配一块（浪费一整块 256 token）</li>
     </ul>'''),

    ("抢占", "四行代码，三个副作用",
     '''<pre>def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)   # ← num_cached_tokens=0, block_table 清空
    self.waiting.appendleft(seq)         # ← 插队到最前</pre>
     <ul>
       <li>token_ids <b>保留</b> → 能重算整段（prompt + 已生成），文本不丢</li>
       <li>生成越久被抢占越亏：重算长度是 num_tokens，不是 prompt 长度</li>
       <li>没有 swap / 没有优先级 / 没有上界</li>
     </ul>'''),

    ("抢占", "LIFO + 无限期推迟（实测）",
     '''<pre class="small">=== case3: KV 不足时的抢占 ===
  [prefill 完] waiting=0 running=2 | #0:running 8/9  #1:running 8/9
step0: DECODE(bs=1)  waiting=1 running=1 | #1:waiting 0/9  #0:running 9/10
step1: DECODE(bs=1)  waiting=1 running=1 | #1:waiting 0/9
step2: DECODE(bs=1)  waiting=1 running=1 | #1:waiting 0/9
step3: DECODE(bs=1)  waiting=1 running=1 | #1:waiting 0/9   ◀ B 一直回不来</pre>
     <ul>
       <li><code>running.pop()</code> 抢<b>队尾</b>（LIFO）；A 继续吃块，B 分不到</li>
       <li>"抢占"在这里不是暂停，是<b>无限期推迟</b> —— 没有公平性</li>
     </ul>'''),

    ("抢占", "边界：抢占救不了自己时会崩",
     '''<pre class="small">while not can_append(seq):
    if self.running:  preempt(self.running.pop())
    else:
        preempt(seq); break        # ← break 跳过 while-else 的调度体
else:
    scheduled_seqs.append(seq)     # ← 抢自己这条路径永远进不来
assert scheduled_seqs              # ← 空 → 崩</pre>
     <pre class="small">实测：num_kvcache_blocks=2，单条 8 token 占满
      预填后 len(seq)=9，free=0
      → AssertionError @ scheduler.py:71（无消息）</pre>
     <p class="sub">没有"序列比整个 KV 池还长"的校验；改 <code>raise RuntimeError(...)</code> 更有用</p>'''),

    ("BlockManager", "四个数据结构同时存在",
     '''<pre>self.blocks            = [Block(i) for i in range(num_blocks)]  全部物理块
self.hash_to_block_id  = {}                    哈希 → 块号
self.free_block_ids    = deque(range(n))        空闲块（FIFO）
self.used_block_ids    = set()                  在用

class Block:   block_id · ref_count · hash · token_ids</pre>
     <ul>
       <li><code>token_ids</code> 不是用来算哈希的，是<b>碰撞二次校验</b>（block_manager.py:66）</li>
       <li>64 位 xxhash 也可能撞 → 命中后再比 token，不等就断链</li>
     </ul>'''),

    ("BlockManager", "链式哈希：位置被编进哈希",
     '''<pre>h0 = H(tokens[0:256])
h1 = H(h0 ‖ tokens[256:512])      ← 前一块的哈希参与本块
h2 = H(h1 ‖ tokens[512:768])

两条序列在块 2 处不同 → 块 2 及之后所有哈希都不同
"哈希相同" ⇒ "从头到此的 token 完全一致" = 前缀语义

（只对块内 token 哈希是错的：块 1 相同但块 0 不同会误命中）</pre>'''),

    ("BlockManager", "range(num_blocks − 1)：尾块永不共享",
     '''<pre class="small">for i in range(seq.num_blocks - 1):     # ← 少看一块
    ...
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        break</pre>
     <table>
       <tr><th>实测（labs/sim_prefix_cache.py）</th><th class="num">结果</th></tr>
       <tr><td>12 token、bs=4、完全相同的 prompt</td><td class="num">复用 2 块 = 8 token，重算 4</td></tr>
       <tr><td>单块序列（≤256 token）的第二次请求</td><td class="num bad">命中 0 块</td></tr>
     </table>
     <p class="sub">原因：尾块可能没写满、还会继续写 → 共享会被自己污染。代价：每次都至少重算一个块</p>'''),

    ("BlockManager", "块越大，前缀缓存越容易失效（实测）",
     '''<pre class="small">[case5] block_size= 4 命中 2 块 =  8 token / 12，重算 4 token
[case5] block_size= 8 命中 1 块 =  8 token / 12，重算 4 token
[case5] block_size=16 命中 0 块 =  0 token / 12，重算 12 token</pre>
     <ul>
       <li>默认 256 ⇒ 两条请求前 256 token 完全相同才有收益</li>
       <li>短 prompt（&lt;256）场景：前缀缓存<b>一次都不命中</b></li>
       <li>块越大：内部碎片越多（300 token 占 512 的 KV）+ 复用率越低</li>
     </ul>'''),

    ("BlockManager", "没有淘汰，也没有 COW",
     '''<pre>deallocate:  只 ref_count--，到 0 就回 free；索引和 KV 内容都保留
_allocate_block: popleft free（FIFO），顺手清掉指向自己的陈旧索引</pre>
     <ul>
       <li>释放后仍可命中（KV 还在）—— 设计选择，不是 bug</li>
       <li>没有 LRU / TTL / 水位：高频长 system prompt 可能被突发流量冲掉</li>
       <li><b>没有写时复制</b>，因为只共享"已写满、不再写"的块 —— 问题在源头消失</li>
     </ul>'''),

    ("进程模型", "rank0 在主进程，其余在子进程",
     '''<pre class="small">ctx = mp.get_context("spawn")
for i in range(1, tensor_parallel_size):
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
self.model_runner = ModelRunner(config, 0, self.events)   # rank0 就在本进程</pre>
     <ul>
       <li>spawn 而非 fork：不继承 CUDA context</li>
       <li>rank0 在主进程 ⇒ <code>generate</code> 同步阻塞，没有前端线程</li>
       <li>子进程构造完就永久阻塞在 <code>loop()</code> 等 Event</li>
     </ul>'''),

    ("进程模型", "IPC：4 字节长度 + pickle，1MB 上限",
     '''<pre class="small">write_shm:  buf[0:4] = 长度;  buf[4:n+4] = pickle([method, *args]);  event.set()
read_shm:   event.wait();  n = buf[0:4];  pickle.loads(buf[4:n+4])</pre>
     <table>
       <tr><th>场景</th><th>每条序列</th><th>512 条</th></tr>
       <tr><td>decode（只传 last_token）</td><td>~200 B</td><td>~100 KB ✓</td></tr>
       <tr><td>prefill 512 token（传全量）</td><td>~2.2 KB</td><td class="bad">~1.1 MB ✗</td></tr>
     </table>
     <p class="sub">推算（需 CUDA 才能实测）：TP&gt;1 + 大 batch prefill 会超 1MB；短 prompt 小 batch 才安全</p>'''),

    ("ModelRunner", "构造十步，顺序不能换",
     '''<pre class="small">1  init_process_group("nccl", "tcp://localhost:2333")   ← 端口写死
2  torch.cuda.set_device(rank)                            ← 设备号==rank
3  set_default_dtype(bf16)
4  set_default_device("cuda")                             ← 之后建参数直接上卡
5  Qwen3ForCausalLM(hf_config)
6  load_model(model, path)
7  Sampler()
8  warmup_model()          ← 先测峰值
9  allocate_kv_cache()     ← 再按剩余显存切块
10 capture_cudagraph()     ← 最后录图</pre>
     <p class="sub">8 必须在 9 之前：块数公式要用 warmup 测出的 peak</p>'''),

    ("KV 显存", "一个块 28 MiB 是算出来的",
     '''<pre>block_bytes = 2 × num_layers × block_size × (kv_heads/tp) × head_dim × dtype_size
            = 2 × 28 × 256 × 8 × 128 × 2 B
            = 29,360,128 B = 28.0 MiB / 块</pre>
     <table>
       <tr><th>量</th><th class="num">值</th></tr>
       <tr><td>一块 → 多少 token 的 KV</td><td class="num">256</td></tr>
       <tr><td>4096 token 的序列</td><td class="num">16 块 = 448 MiB</td></tr>
       <tr><td>8GB 卡（可用 ~6.5G）KV 池上限</td><td class="num">≈ 230 块 ≈ 5.9 万 token</td></tr>
       <tr><td>TP=2 时每块</td><td class="num">14 MiB（容量不变）</td></tr>
     </table>'''),

    ("KV 显存", "块数公式：预留的是峰值，不是当前",
     '''<pre class="small">free, total = torch.cuda.mem_get_info()
used    = total - free                      # 整张卡已占用
peak    = allocated_bytes.all.peak          # 本进程历史最高
current = allocated_bytes.all.current       # 当前持有（已 empty_cache）

blocks = int(total*0.9 - used - peak + current) // block_bytes
                      └目标┘   └别人┘  └我即将要用的净增量┘</pre>
     <ul>
       <li><code>+current</code>：因为 used 已含 current，要用 peak 替换它</li>
       <li>隐含假设：真实峰值 ≈ warmup 峰值（4×4096 = 16384 token 的假 prefill）</li>
       <li class="warn"><code>assert &gt; 0</code> 是唯一提示 —— 不说差多少</li>
     </ul>'''),

    ("Attention", "三个索引张量，各管一段",
     '''<table class="small">
       <tr><th>张量</th><th>形状</th><th>作用</th></tr>
       <tr><td>cu_seqlens_q</td><td>[bs+1]</td><td>每条序列 query 起点（只含要算的）</td></tr>
       <tr><td>cu_seqlens_k</td><td>[bs+1]</td><td>每条序列 key 起点（含前缀缓存）</td></tr>
       <tr><td>slot_mapping</td><td>[N]</td><td>新 KV 写到哪个 slot</td></tr>
       <tr><td>block_table</td><td>[bs, max_blk]</td><td>逻辑块 → 物理块（-1 填充）</td></tr>
     </table>
     <pre class="small">if cu_seqlens_k[-1] &gt; cu_seqlens_q[-1]:   # 有前缀缓存
    block_tables = self.prepare_block_tables(seqs)</pre>'''),

    ("Attention", "Triton 写 20 行，attention 买现成",
     '''<pre class="small">@triton.jit
def store_kvcache_kernel(...):
    idx  = tl.program_id(0)                    # 一个 program = 一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                     # ← CUDA Graph 的 padding
    key   = tl.load(key_ptr + idx*key_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot*D + tl.arange(0, D), key)   # 同理 V</pre>
     <ul>
       <li>为什么不用 <code>k_cache[slots] = k</code>：不连续 scatter、要跳过 -1、要多一次索引构造</li>
       <li>分工：<b>写 KV 的难点在索引映射</b>（自己写），<b>attention 的难点在 tiling</b>（买）</li>
     </ul>'''),

    ("Attention", "前缀缓存时的 prefill 技巧",
     '''<pre class="small">if context.is_prefill:
    if context.block_tables is not None:   # 有缓存
        k, v = k_cache, v_cache            # ← 4 维整块 cache 当 k/v 传进去
    o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)
else:
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens, block_table=...)</pre>
     <ul>
       <li>已缓存的 2048 token 的 KV 只在页里，不在本步的 k/v 里</li>
       <li>靠 block_table 翻译"逻辑位置 → 物理页"，省掉一次全量 gather</li>
     </ul>'''),

    ("并行", "列切输出、行切输入、末尾 all_reduce",
     '''<table>
       <tr><th>层</th><th>切什么</th><th>通信</th><th>用在</th></tr>
       <tr><td>ColumnParallel</td><td>输出维</td><td>无</td><td>—</td></tr>
       <tr><td>QKVParallel</td><td>输出维，q/k/v 各段偏移</td><td>无</td><td>qkv_proj</td></tr>
       <tr><td>MergedColumn</td><td>输出维，等宽分段</td><td>无</td><td>gate_up_proj</td></tr>
       <tr><td>RowParallel</td><td>输入维</td><td>all_reduce</td><td>o_proj/down_proj</td></tr>
     </table>
     <pre class="small">y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size &gt; 1: dist.all_reduce(y)    # bias 只在 rank0 加，否则被放大 tp 倍</pre>'''),

    ("并行", "QKV 为什么不能 chunk",
     '''<pre class="small">num_attention_heads=16, num_key_value_heads=8, head_dim=128
合并宽度 = (16 + 2×8) × 128 = 4096        # 比例不是 1:1:1

q: shard_offset = 0
k: shard_offset = num_heads*head_dim          = 2048
v: shard_offset = (num_heads+num_kv_heads)*head_dim = 3072

param_data.narrow(tp_dim, shard_offset, shard_size)      # 按段偏移
loaded_weight.chunk(tp_size, tp_dim)[tp_rank]            # 按头均分</pre>
     <p class="sub">两次切的对象不同（一个按段、一个按均分）—— 这段最容易看错</p>'''),

    ("Qwen3", "三个融合点",
     '''<div class="row">
       <div class="col">
         <div class="card"><h3>① 残差并入 RMSNorm</h3><p>attn 输出不自己加残差，推到下一次
         <code>add_rms(h, r)</code>：每层从 4 次显存往返降到 2 次</p></div>
         <div class="card"><h3>② qk-norm</h3><p>Qwen3 全系 attention_bias=False ⇒
         <code>q_norm/k_norm = RMSNorm(head_dim)</code>，必须<b>在 RoPE 之前</b></p></div>
       </div>
       <div class="col">
         <div class="card"><h3>③ SwiGLU 合并</h3><p>gate_up_proj 一次 GEMM 出
         [gate|up]，<code>silu(gate)*up</code> 再合成一个 kernel</p></div>
         <div class="card"><h3>@torch.compile 用在哪</h3><p>5 处全是逐元素小算子
         （norm/rope/silu/sampler）；GEMM 不编译，交给 cuBLAS</p></div>
       </div>
     </div>'''),

    ("Qwen3", "一个静默失效：rope_scaling 只取 theta",
     '''<pre>if isinstance(rope_scaling, dict):
    rope_theta = rope_scaling.get("rope_theta", rope_theta)
# type / factor 全被忽略</pre>
     <ul>
       <li>Qwen3-0.6B 的 config 里 <code>rope_scaling=None</code>，现在没问题</li>
       <li>换成需要 YaRN 扩展的模型 → <span class="warn">位置编码静默用错</span>：不报错、不崩，只是质量下降</li>
       <li>同类坑：<code>assert rotary_dim == head_size</code>（不支持部分维度 RoPE）</li>
     </ul>'''),

    ("采样", "12 行，一行是核心",
     '''<pre>logits = logits.float().div_(temperatures.unsqueeze(1))
probs  = torch.softmax(logits, dim=-1)
sample = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)</pre>
     <pre class="small">证明（三行）：
  T_k = E_k / p_k ~ Exp(rate = p_k)
  argmax(p_k/E_k) = argmin(T_k)                 ← 指数竞速
  P(k 先到期) = p_k / Σp = p_k                  ∎</pre>
     <p class="sub">= Gumbel-max 的另一种写法：argmax(log p + G)。少两次 transcendental</p>'''),

    ("采样", "实测：分布对得上，温度只影响锐度",
     '''<pre class="small">5 类偏斜  目标=[0.5 0.25 0.15 0.08 0.02]  经验=[0.501 0.251 0.15 0.079 0.02]  偏差 0.0015
教科书式  目标=[0.5 0.25 0.15 0.08 0.02]  经验=[0.5   0.25  0.149 0.08 0.02]  偏差 0.0005

T=0.3   p=[0.958 0.034 0.006 0.001]  命中率 0.959
T=1.0   p=[0.579 0.213 0.129 0.078]  命中率 0.579
T=10.0  p=[0.279 0.252 0.24  0.228]  命中率 0.274   ← 接近随机按键</pre>
     <ul>
       <li>没有 top-k/top-p 兜底 ⇒ 输出质量对温度远更敏感</li>
       <li>无 seed 字段 ⇒ 同 prompt 两次结果不同，不可复现</li>
     </ul>'''),

    ("CUDA Graph", "为什么只给 decode 录",
     '''<div class="row">
       <div class="col">
         <div class="card"><h3>decode 的问题</h3><p>一条序列只算 1 token，
         数百个 kernel 的 launch 开销（1–3 ms）远大于计算（~0.3 ms）</p></div>
         <div class="card"><h3>图的收益</h3><p>launch 从 O(kernel 数) 降到 O(1)；
         一份 pool 复用，35 个桶共享</p></div>
       </div>
       <div class="col">
         <div class="card"><h3>prefill 为什么不录</h3><p>cu_seqlens 形状随 batch 变；
         前缀缓存时 block_tables 时有时无；计算时间长，收益 &lt;5%</p></div>
         <div class="card"><h3>分桶</h3><p>[1,2,4,8] + 16..512 步长 16；
         bs=17 用 32 的桶，浪费一半算力</p></div>
       </div>
     </div>'''),

    ("CUDA Graph", "重放时的五个静态 buffer",
     '''<pre class="small">bs    = input_ids.size(0)
graph = self.graphs[next(x for x in self.graph_bs if x &gt;= bs)]   # 向上取桶
graph_vars["input_ids"][:bs]     = input_ids
graph_vars["slot_mapping"].fill_(-1)          # ← 多余行置 -1，kernel 直接 return
graph_vars["slot_mapping"][:bs]  = context.slot_mapping
graph_vars["context_lens"].zero_()            # ← 多余行长度 0
graph_vars["context_lens"][:bs]  = context.context_lens
graph_vars["block_tables"][:bs, :w] = context.block_tables
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])</pre>
     <p class="sub">捕获必须从大到小（pool 由最大的桶创建）；捕获前要先跑一次热身，排除所有"首次"</p>'''),

    ("基准", "README 那张表能得出什么",
     '''<pre class="small">Hardware: RTX 4070 Laptop 8GB | Model: Qwen3-0.6B | 256 seqs | in/out 100–1024

| Engine     | Output Tokens | Time (s) | Throughput (tok/s) |
| vLLM       | 133,966       | 98.37    | 1361.84            |
| Nano-vLLM  | 133,966       | 93.41    | 1434.13            |

校验：133,966/93.41 = 1434.2 ✓   133,966/98.37 = 1361.8 ✓   快 5.3%</pre>
     <table class="small">
       <tr><th>能说</th><th>不能说</th></tr>
       <tr><td>这台机、这个模型、这批请求上打平 vLLM</td><td>nano 比 vLLM 快（5.3% 可能是噪声，只跑 1 次）</td></tr>
       <tr><td>1450 行能顶住单卡小模型离线批处理</td><td>长上下文/有前缀缓存/多卡/在线服务也一样</td></tr>
     </table>'''),

    ("基准", "随机 prompt 让前缀缓存完全失效",
     '''<pre>prompt_token_ids = [[randint(0, 10000) ...] for _ in range(256)]
                   └─ 每条都不同 → 块哈希全不命中 → 纯测计算路径</pre>
     <ul>
       <li>这份基准测的是<b>冷启动吞吐</b>，含全部 prefill</li>
       <li><code>ignore_eos=True</code> 是公平对比的前提：总 token 数可预测</li>
       <li>分母含 prefill + 排队 ⇒ 不是用户感知速度；TTFT/ITL/p99 一个都没测</li>
     </ul>'''),

    ("本机", "哪些结论是跑出来的，哪些是读出来的",
     '''<div class="row">
       <div class="col">
         <div class="card"><h3>跑出来的（labs/，纯 CPU 跑原码）</h3>
         <p>块分配 / 前缀缓存复用率 / 尾块不共享 · prefill 饥饿 ·
         chunk 只给队首 · LIFO 抢占 · 抢占边界 assert 崩 · Gumbel-max 分布</p></div>
       </div>
       <div class="col">
         <div class="card"><h3>读出来的（需要 CUDA）</h3>
         <p>Triton kernel 效率 · CUDA Graph 正确性与收益 ·
         FlashAttention paged 行为 · 端到端 tok/s · 1MB shm 上限</p></div>
       </div>
     </div>
     <p class="sub">Apple M5 / 无 CUDA / Python 3.14 超出 requires-python(&lt;3.13) ⇒ 引擎跑不起来；
     block_manager 与 scheduler 只依赖 xxhash+numpy，可以原样 import</p>'''),

    ("对照", "八维对照：两种极端",
     '''<table class="small">
       <tr><th>维度</th><th>nano-vllm</th><th>DeepInfer</th></tr>
       <tr><td>调度</td><td>两队列 / 同步 / 批级</td><td>ZOS 四段流水线 / 异步</td></tr>
       <tr><td>前缀缓存</td><td>引擎内块哈希</td><td>引擎外 3FS + 引擎内 COW</td></tr>
       <tr><td>重算</td><td>抢占即全量，无上界</td><td>bounded replay 128 + 6</td></tr>
       <tr><td>并行</td><td>只有 TP（切头）</td><td>只有 EP/DP（切专家）</td></tr>
       <tr><td>精度</td><td>bf16</td><td>FP4 + state slot</td></tr>
       <tr><td>注意力</td><td>FlashAttention（买）</td><td>CSA2 稀疏（自研 TileLang）</td></tr>
       <tr><td>采样</td><td>12 行 Gumbel-max</td><td>fused + batch-invariant</td></tr>
       <tr><td>服务面</td><td>无</td><td>多模态 / EPD / 双硬件</td></tr>
     </table>'''),

    ("对照", "并行轴由模型结构决定",
     '''<div class="row">
       <div class="col">
         <div class="card"><h3>nano-vllm 只有 TP</h3>
         <p>稠密小模型 · 单卡装得下 · 切头实现成本低（切输出 + all_reduce）<br>
         <span class="bad">缺点：每层 2 次 all_reduce，小 batch 时通信占比高</span></p></div>
       </div>
       <div class="col">
         <div class="card"><h3>DeepInfer 只有 EP/DP</h3>
         <p>MoE 模型 · 专家数远超卡数 · 必须按专家切<br>
         <span class="bad">代价：路由不均 → 需要 EPLB 重排 + symm_mem shuffle</span></p></div>
       </div>
     </div>
     <p class="sub">稠密切头，稀疏切专家 —— 不是框架偏好，是模型结构</p>'''),

    ("收束", "三个核心结论",
     '''<ul>
       <li><b>它就是 vLLM 的骨架</b>：分页 KV + 前缀缓存 + 连续批处理 + TP，
       没有一处重新发明；价值在于把散在几十个文件的路径压到 1450 行</li>
       <li><b>调度是"批级"而非"token 级"</b>：一次 step 全 prefill 或全 decode，
       chunk 只给队首 —— decode 饥饿的根因（已实测）</li>
       <li><b>没有 swap / 淘汰 / 上界</b>：KV 不够就清空重算（LIFO 抢占），
       代价是被抢占的长序列可能长期饥饿，边界上还会 assert 崩</li>
     </ul>
     <p class="sub" style="margin-top:16px">它赢的不是"更聪明的算法"，是"敢省"</p>'''),

    ("路径", "下一步：读 → 改 → 对照",
     '''<pre class="small">第 1 步（20h）本仓库 docs/00–15
            产出：画出 prompt→token 全链路，能预测参数改动的影响
第 2 步（10h）动手改
            ├ 加 top-k 采样（~30 行）
            ├ 加 swap（~150 行，见 docs/06 改法 A）
            └ 加抢占计数埋点（3 行，见 docs/14 第 6 节）
第 3 步（40h）deepinfer-study 15 节课，逐节回来做对照
第 4 步 挑一个差异点做深
            ├ ZOS 异步调度 ↔ 两队列
            ├ YoCo 128+6 ↔ 全量重算
            └ batch-invariant ↔ Gumbel-max</pre>'''),

    ("自测", "十个问题，答不上就回去重读",
     '''<ol class="q">
       <li>第一步 decode 为什么就需要新块？（case3）</li>
       <li>哪两行导致 decode 饥饿？</li>
       <li>为什么尾块永不共享、代价是什么？</li>
       <li>抢占后 token_ids 还在吗？为什么？</li>
       <li>为什么没有 COW？</li>
       <li>一个块 28 MiB 怎么算？</li>
       <li>bias 为什么只在 rank0 加？</li>
       <li>slot_mapping 里的 -1 是谁写的？</li>
       <li>argmax(p/E) 为什么等价于按 p 采样？</li>
       <li>为什么 prefill 不录图？</li>
     </ol>'''),

    ("", "自己动手，才算读过",
     '''<p class="sub" style="font-size:24px;line-height:1.7">
     <code>docs/</code> 16 节课 · <code>labs/</code> 3 个可跑实验 · <code>slides/</code> 本 deck<br><br>
     仓库：nano-vllm-study<br>
     源码：GeeeekExplorer/nano-vllm @ bb823b3（MIT）<br>
     对照：deepseek-ai/DeepInfer @ 9bd9cf2</p>'''),
]


def render() -> str:
    parts = []
    total = len(SLIDES)
    for k, (kick, title, body) in enumerate(SLIDES, 1):
        cover = k in (1, total)
        num = "" if cover else f'<span class="n">{k:02d}</span>'
        head = f'<h2>{num}{title}</h2>' if title else ""
        kicker = f'<div class="kicker">{kick}</div>' if kick else ""
        parts.append(f'<section class="slide">\n{kicker}\n{head}\n{body}\n'
                     f'<footer><span>{FOOT}</span><span>{k} / {total}</span></footer>\n</section>')
    return "\n".join(parts)


def main():
    tpl = (HERE / "deck_template.html").read_text(encoding="utf-8")
    assert "__SLIDES__" not in tpl and "<!--SLIDES-->" in tpl, "模板占位符缺失"
    html = tpl.replace("<!--SLIDES-->", render())
    assert html.count('<section class="slide">') == len(SLIDES)
    out = HERE / "nano-vllm-深度解析.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  slides={len(SLIDES)}  bytes={len(html):,}")


if __name__ == "__main__":
    main()
