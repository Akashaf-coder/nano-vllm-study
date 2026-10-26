# 11 Qwen3 模型图与三个融合点

## 本课目标

- 把 216 行模型代码压成一张图：embed → 28 × (attn + mlp) → norm → lm_head；
- 看懂三个"省一次显存往返"的融合：残差融合 RMSNorm、qk-norm、SwiGLU；
- 知道哪些 `torch.compile` 装饰器是真优化、哪些只是"省一次 Python 开销"；
- 认清 `rope_scaling` 的处理缺口（长上下文场景会踩）。

## 1. 模型尺寸（Qwen3-0.6B 实测 config）

```
hidden_size=1024   intermediate_size=3072   num_hidden_layers=28
num_attention_heads=16   num_key_value_heads=8   head_dim=128
vocab_size=151936   max_position_embeddings=40960   rope_theta=1000000
tie_word_embeddings=True   attention_bias=False   torch_dtype=bfloat16
```

两个立刻要用的推论：

- `attention_bias=False` → **qk-norm 生效**（`qwen3.py:68-70` 的条件分支）；
- `tie_word_embeddings=True` → `lm_head` 与 `embed_tokens` 共享权重（`qwen3.py:202-203`）。

## 2. 一张图

```
input_ids [N]
   │
   ▼
VocabParallelEmbedding ──▶ hidden [N, 1024]
   │
   ▼  residual = None
┌──────────────── Qwen3DecoderLayer × 28 ────────────────┐
│                                                        │
│  hidden, residual = input_layernorm(hidden, residual)  │  ← 融合点①
│        │                                               │
│        ▼                                               │
│   Qwen3Attention                                       │
│     qkv_proj  ─▶ q,k,v 按头切                           │
│     q_norm / k_norm（per-head RMSNorm）                 │  ← 融合点②
│     rotary_emb(positions, q, k)                        │
│     Attention ─▶ Triton 写 KV + FlashAttention 读 KV    │
│     o_proj（行并行 + all_reduce）                       │
│        │                                               │
│        ▼                                               │
│  hidden, residual = post_attention_layernorm(hidden, residual)  ← 融合点①
│        │                                               │
│        ▼                                               │
│   Qwen3MLP: gate_up_proj → SiluAndMul → down_proj      │  ← 融合点③
│                                                        │
└────────────────────────────────────────────────────────┘
   │
   ▼
RMSNorm（最后一次，带 residual）
   │
   ▼
ParallelLMHead（prefill 只取每条序列最后一位；TP 时 gather 到 rank0）
   │
   ▼
logits [bs, 151936]
```

## 3. 融合点①：残差并入 RMSNorm

```python
# qwen3.py:146-159
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

注意 `self_attn` **不接受 residual** —— 也就是说 attention 的输出**不在这里加残差**，而是被推到下一次 `post_attention_layernorm` 里一起做。这就是"残差融合"：

```
朴素写法（每层 4 次显存往返）：
  h = h + attn(norm(h))          # 读 h、写 h
  h = h + mlp(norm(h))           # 再读写一次

融合写法（每层 2 次）：
  x, r = add_rms(h, r)   # r 是上一步的残差，一次算完 add + norm
  x    = attn(x)
  x, r = add_rms(x, r)   # attention 输出在这里才和残差相加
  x    = mlp(x)
```

实现（`layernorm.py:28-40`）：

```python
@torch.compile
def add_rms_forward(self, x, residual):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())        # ① 累加（fp32）
    residual = x.to(orig_dtype)                 # ② 新的残差 = 累加结果
    var = x.pow(2).mean(dim=-1, keepdim=True)   # ③ 在 fp32 上算方差
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)      # ④ 归一化后再降回 bf16
    return x, residual
```

三个细节：**fp32 累加**（bf16 直接累加会丢精度）、**`mul_` 原地**（省一次分配）、**`@torch.compile` 把四步合成一个 kernel**（否则就是 4 次 kernel launch + 4 次显存往返，对 decode 这种"每步只算 1 token"的场景是纯开销）。

`model.forward` 最后一行 `hidden_states, _ = self.norm(hidden_states, residual)`（`qwen3.py:182`）—— 最后一层的残差在这里才消费掉。

## 4. 融合点②：qk-norm（Qwen3 特有）

```python
# qwen3.py:68-70
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

```python
# qwen3.py:82-84
if not self.qkv_bias:
    q = self.q_norm(q)      # q 形状 [N, 16, 128] → 对最后一维（head_dim）归一化
    k = self.k_norm(k)
```

两点值得注意：

1. **条件是 `not qkv_bias`**，而不是显式的 `use_qk_norm` 配置项。Qwen3 全系 `attention_bias=False`，所以这个分支在 Qwen3 上恒为真 —— 但语义上是"没 bias 就补 norm"，这是个**脆弱的隐含约定**：换一个"既没有 qkv_bias 又不需要 qk-norm"的模型（比如某些 Llama 变体）就会静默多算一次 norm。
2. **`RMSNorm(head_dim)` 作用在每个头上**：形状 `[N, heads, head_dim]`，`mean(dim=-1)` 正好是对每个头归一化。如果写成对 `hidden_size` 归一化，形状都不对，会直接报错 —— 这个设计"错不了"。

qk-norm 的作用是抑制 attention logits 的幅度（Qwen 系列为了长上下文稳定）。它必须在 RoPE **之前**（`qwen3.py:82-85` 的顺序），因为 RoPE 是旋转、norm 是缩放，交换顺序结果不同。

## 5. 融合点③：SwiGLU

```python
# activation.py:6-11
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

`gate_up_proj` 一次算出 `[gate | up]`（宽度 `2 × 3072`），然后 `chunk` 切开、`silu(gate) * up`。融合的意义：**一次 GEMM 代替两次**（`gate_up_proj` 是 `MergedColumnParallelLinear`，一张权重矩阵、一次 kernel），并且 `silu * mul` 被 `torch.compile` 合成一个 kernel。

## 6. RoPE：`lru_cache(1)` 与一个缺口

```python
# rotary_embedding.py:51-59
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
```

- **`lru_cache(1)`**：全部 28 层共享同一个 RoPE 对象（参数完全相同），省 27 份 `cos_sin_cache`。`max_position=40960`、`head_dim=128` 时这张表是 `40960 × 128 × 2 × 2 B ≈ 20 MiB`，共享是必要的。
- **`assert rotary_dim == head_size`**（`rotary_embedding.py:28`）：只支持"全维度 RoPE"，不支持 GPT-NeoX 那种只转一部分的配置（部分模型 `partial_rotary_factor=0.25`）。
- **`rope_scaling` 只取 `rope_theta`**（`qwen3.py:54-55`）：

```python
if isinstance(rope_scaling, dict):
    rope_theta = rope_scaling.get("rope_theta", rope_theta)
```

也就是说 `rope_scaling={"type": "yarn", "factor": 4.0, ...}` 里的 **type/factor 全被忽略**，只拿 `rope_theta`。Qwen3-0.6B 的 config 里 `rope_scaling=None`，所以现在没问题；**一旦换成需要 YaRN 扩展上下文的模型（比如 Qwen3 长上下文版本），这里会静默用错位置编码** —— 不报错、不崩，只是输出质量下降。这是本项目最危险的一处静默失效，比 `docs/09` 第 5 节的 `positions` 问题更隐蔽，因为它发生在构造期而不是每步。

RoPE 计算本身（`rotary_embedding.py:6-14`）是标准的"前后半交替"实现：

```python
x1, x2 = torch.chunk(x.float(), 2, dim=-1)   # 前半 / 后半
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

注意它在 fp32 里算再降回 bf16 —— 与 RMSNorm 同样的精度考虑。

## 7. `@torch.compile` 用在哪、为什么

全项目 5 处（`grep -c "torch.compile"` = 5）：

| 位置 | 函数 | 为什么值得编译 |
| --- | --- | --- |
| `layernorm.py:16` | `rms_forward` | decode 时 28 层各调一次，小算子 launch 开销大 |
| `layernorm.py:28` | `add_rms_forward` | 同上 + 融合 add |
| `rotary_embedding.py:37` | `forward` | chunk/cat/mul 一串小算子 |
| `activation.py:8` | `SiluAndMul.forward` | 两个算子合一 |
| `sampler.py:7` | `Sampler.forward` | 温度除法 + softmax + 除法 + argmax 四合一 |

规律：**全是"逐元素/小形状"的算子**。矩阵乘（`F.linear`）**没有**编译 —— 那已经交给 cuBLAS。这个选择很专业：`torch.compile` 的收益来自"减少 kernel launch 与中间张量"，而不是"加速 GEMM"。

代价：首次调用要编译（`docs/08` 说 warmup 把这段吃掉了），且编译缓存不持久（进程重启重编）。

## 8. 前向全程的通信次数（TP=2）

| 位置 | 通信 | 次数/层 |
| --- | --- | --- |
| `o_proj` | `all_reduce` | 1 |
| `down_proj` | `all_reduce` | 1 |
| `embed_tokens` | `all_reduce` | 1（仅一次） |
| `lm_head` | `gather` | 1（仅一次） |

28 层 × 2 = 56 次 `all_reduce`，每步 decode 都做。batch=1 时这是纯开销 —— 也是"小模型不值得 TP"的原因（通信量固定，计算量随 batch 增长）。

## 9. 陷阱清单

1. **`rope_scaling` 的 type/factor 被忽略**：换长上下文模型会静默错。
2. **`assert rotary_dim == head_size`**：不支持部分维度 RoPE。
3. **qk-norm 的门是 `not qkv_bias`**：不是显式配置项。
4. **`_ = self.norm(...)` 的最后一个残差**：漏了 `residual` 会少加一次残差，输出质量崩但不报错。
5. **`tie_word_embeddings` 直接赋值 `.data`**：共享存储，加载时若两个权重都出现在 safetensors 里，后加载的会覆盖前者。
6. **`lru_cache(1)` 的 key 是参数元组**：如果同一进程里跑两个不同 `rope_theta` 的模型，第二个会拿到第一个的缓存（`lru_cache(1)` 只留最近一个，所以最多"乒乓"失效，性能退化为每次重建 20 MiB 表）。

## 10. 自测

1. 为什么 `self_attn` 的输出不在 attention 内部加残差？这样省了什么？
2. `add_rms_forward` 为什么要先 `.float()` 再算方差？直接 bf16 会怎样？
3. qk-norm 为什么作用在 `head_dim` 上，而不是 `hidden_size` 上？
4. 28 层共享一个 RoPE 对象省了多少显存？为什么能用 `lru_cache(1)`？
5. `rope_scaling` 支持不全，会在什么场景下被触发？为什么会"静默"？

## 下一课

`docs/12-Sampler与采样.md`：一行除法 + argmax 怎么就等价于按概率采样了。
