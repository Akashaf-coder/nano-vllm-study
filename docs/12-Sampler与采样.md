# 12 Sampler：除以一个指数分布的 argmax

## 本课目标

- 看懂"`probs / Exp(1)` 再 `argmax`"为什么等价于按概率采样（含证明思路）；
- 知道温度在这里是怎么实现的（除 logits，不是除概率）；
- 理解为什么 `temperature=0` 被 assert 禁掉；
- 用本机实测数字确认"批次内不同温度"和"无种子不可复现"这两件事。

## 1. 全部代码：12 行

```python
# sampler.py:5-12
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

逐行：

| 行 | 做了什么 | 注意 |
| --- | --- | --- |
| `logits.float()` | bf16 → fp32 | softmax 在 bf16 上容易丢精度 |
| `.div_(temperatures.unsqueeze(1))` | 每行除以该序列的温度 | 原地操作，温度是 `[bs]` 广播到 `[bs, vocab]` |
| `softmax(dim=-1)` | 归一化成概率 | |
| `empty_like(probs).exponential_(1)` | 抽 `Exp(1)` 噪声 | 与 `probs` 同形状同设备，无额外分配 |
| `.clamp_min_(1e-10)` | 防除零/防 inf | `Exp(1)` 可能小到接近 0，除以它会把 `probs` 放成 inf |
| `probs.div_(...)` | `p_k / E_k` | 原地 |
| `.argmax(-1)` | 取最大者的下标 | |

一行做三件事（除温度、softmax、采样），并且**全部原地**：这个函数在每步 decode 都跑，`[bs, 151936]` 的张量哪怕多一次分配也是几十 MB 的浪费。

## 2. 为什么 `argmax(p/E)` 是正确采样

**定理**：若 `E_k ~ Exp(1)` 独立，则 `argmax_k (p_k / E_k)` 的分布恰好是 `p`。

证明思路（三行）：

1. 令 `T_k = E_k / p_k`。指数分布的性质：`E_k ~ Exp(1)` 除以常数 `p_k` 得到 `T_k ~ Exp(rate = p_k)`，即"速率参数为 `p_k` 的指数分布"。
2. `argmax(p_k/E_k) = argmax(1/T_k) = argmin(T_k)` —— **比赛谁先"到期"**（这就是"指数竞速 / exponential race"的经典构造）。
3. 在独立指数竞速里，第 `k` 个先到期的概率 = `p_k / Σ_j p_j = p_k`（因为 `p` 是 softmax 输出，和为 1）。

这正是 **Gumbel-max 技巧**的另一种写法。教科书写法是 `argmax(log p_k + G_k)`，`G_k ~ Gumbel(0,1)`；两种写法等价，因为 `-log E` 服从标准 Gumbel 分布。

**工程上为什么选"除以指数"而不是"Gumbel 采样"**：Gumbel 需要 `-log(-log(u))` 两次超越函数，指数分布只要一次 `-log(u)`（`torch.exponential_` 内部就是这么做的），而且可以直接原地相除、不需要构造 `log(p)`。**少两次 transcendental，多一步除法 —— 在 vocab=15 万时是实打实的差别。**

## 3. 本机实测

`labs/sim_sampler.py` 用 numpy 等价复现（本机无 torch），20 万次采样：

```
=== 分布正确性（n=200k）===
5 类均匀        目标分布=[0.2 0.2 0.2 0.2 0.2]  经验分布=[0.201 0.199 0.198 0.201 0.201]  最大偏差=0.0018
5 类偏斜        目标分布=[0.5  0.25  0.15  0.08  0.02 ] 经验分布=[0.501 0.251 0.15  0.079 0.02 ]  最大偏差=0.0015
教科书式        目标分布=[0.5  0.25  0.15  0.08  0.02 ] 经验分布=[0.5   0.25  0.149 0.08  0.02 ]  最大偏差=0.0005
```

三种写法（本项目式、教科书 Gumbel、偏斜分布）的最大偏差都在 ±0.002 内，符合 20 万次采样的统计误差（`≈ 1.7σ ≈ 0.0018`）。**这验证的是算法，不是那行 torch 代码** —— 跑 torch 版本需要 GPU。

## 4. 温度：除 logits，不是除概率

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

标准定义：`p = softmax(logits / T)`。`T < 1` 让分布更尖（更容易选高频 token），`T > 1` 更平。实测（同一组 logits `[2.0, 1.0, 0.5, 0.0]`）：

```
T=0.3   p=[0.958 0.034 0.006 0.001]  最可能 token 命中率=0.959
T=1.0   p=[0.579 0.213 0.129 0.078]  最可能 token 命中率=0.579
T=2.0   p=[0.409 0.248 0.193 0.15 ]  最可能 token 命中率=0.403
T=10.0  p=[0.279 0.252 0.24  0.228]  最可能 token 命中率=0.274
```

注意 `T=10` 时分布接近均匀（0.25），此时模型几乎在"随机按键" —— **没有任何截断（top-k/top-p）保护**，长尾里那些本不该出现的 token 会被真实采到。这就是"只有温度的采样"的实际风险：**输出质量对温度的敏感度远高于带截断采样的实现**。

批次内温度不同完全没问题：`temperatures` 是 `[bs]`，由 `prepare_sample` 逐序列收集（`model_runner.py:190-193`），广播到 `[bs, vocab]` 后每行各除各的。**一个 batch 里可以有人要 `T=0.6`、有人要 `T=1.2`，一次 kernel 搞定** —— 这是"fused sampler"相对"逐请求采样"的优势。

## 5. 为什么禁止 `temperature=0`

```python
# sampling_params.py:10-11
def __post_init__(self):
    assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

`T=0` 时 `logits/0` → inf → softmax 变成 one-hot → `argmax` 恒为最大值下标。数学上"能跑"，但：

1. **`0/0` 的情况会出 NaN**（全零 logits 除以 0）；
2. **命中率恒为 1**，性能表现与采样路径不同（没有随机性 → 相同 prompt 完全可复现），会让基准测试"看起来更快/更稳"；
3. 作者选择**显式禁掉**而不是静默退化成 `argmax`，这样调用方必须自己明确意图。

要想贪心解码，改法不是传 `T=0`，而是改这里：

```python
sample_tokens = probs.div_(noise).argmax(-1)   # 改成 logits.argmax(-1)，跳过 softmax 与噪声
```

`bench.py` 用 `temperature=0.6` + `ignore_eos=True` 来保证"生成满 max_tokens"，**没有走贪心路径**。

## 6. 其他影响输出的因素（都不可控）

| 因素 | 现状 | 后果 |
| --- | --- | --- |
| 随机种子 | 无字段（`sampling_params.py` 只 3 个参数） | 同 prompt 两次结果不同，**不可复现** |
| `top_k` / `top_p` | 无（grep 命中 0） | 长尾 token 可能被采到 |
| 重复惩罚 | 无 | 长文本容易复读 |
| `logprobs` | 无 | 无法做置信度筛选 |
| 停止串 | 无 | 只能在客户端后处理 |

这五条合起来是一句话：**这个采样器只能"跑通"，不能"调优"**。教学实现里没问题，但它也解释了为什么 vLLM 的 sampler 有几百行 —— 采样质量是用户第一眼能感知的东西。

## 7. 与 `@torch.compile` 的配合

`Sampler.forward` 带 `@torch.compile`（`sampler.py:7`），把"除温度 → softmax → 抽噪声 → 除 → argmax"合成尽量少的 kernel。注意：

- **`torch.empty_like(probs).exponential_(1)` 在编译图里是随机算子**，`torch.compile` 能保留其随机性（不会因为重排而被消除），但**图捕获（CUDA Graph）对随机数有特殊处理**：图重放会用同一份 RNG 状态推进，所以同一 graph 多次重放的随机序列是确定的（取决于图的捕获顺序）。这会带来"同一 batch size 下随机性相关性"的微妙问题 —— 属于 CUDA Graph + 随机数的经典坑，见 `docs/13`。

## 8. 自测

1. 用三行话证明 `argmax(p/E)` 等价于按 `p` 采样。
2. 为什么工程上更喜欢"除以指数"而不是"Gumbel 采样"？
3. `clamp_min_(1e-10)` 挡住的是什么情况？不挡会怎样？
4. 为什么禁掉 `T=0` 而不是让它退化成 `argmax`？
5. 一个 batch 里不同序列温度不同，代码里哪一行保证它们互不干扰？

## 下一课

`docs/13-CUDA-Graph捕获.md`：为什么 decode 要"录一遍再重放"，而 prefill 不录。
