"""实验 3：验证 Sampler 的"除以指数分布"为什么等价于按概率采样。

nanovllm/layers/sampler.py 只有一行核心：
    probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
即 argmax_k  p_k / E_k ，E_k ~ Exp(1)。这在数学上就是 Gumbel-max 技巧的另一种写法
（指数分布竞速：每个候选抽一个"到期时间"，-log(E)/p 最小者胜出，等价于 p/E 最大）。

本机没有 torch，所以这份脚本是 numpy 的等价复现，用来验证分布是否正确 —— 它验证的是
**算法**，不是那行 torch 代码本身（那需要 GPU 环境，见 docs/12）。
运行：~/.venv-nanovllm-lab/bin/python labs/sim_sampler.py
"""
import numpy as np

rng = np.random.default_rng(0)


def sample_gumbel_ratio(probs: np.ndarray) -> np.ndarray:
    """与 sampler.py 完全同构的 numpy 版（probs 已是 softmax 输出）。"""
    e = rng.exponential(1.0, size=probs.shape).clip(1e-10)
    return (probs / e).argmax(axis=-1)


def sample_gumbel_max(probs: np.ndarray) -> np.ndarray:
    """教科书写法：argmax(log p + Gumbel(0,1))，用于交叉验证。"""
    g = -np.log(-np.log(rng.random(probs.shape) + 1e-12) + 1e-12)
    return (np.log(probs) + g).argmax(axis=-1)


def report(name, p, fn, n=200_000):
    x = np.repeat(p[None, :], n, axis=0)
    idx = fn(x)
    emp = np.bincount(idx, minlength=len(p)) / n
    dev = np.abs(emp - p).max()
    print(f"{name:<12} 目标分布={np.round(p, 3)}  经验分布={np.round(emp, 3)}  最大偏差={dev:.4f}")


def main():
    print("=== 分布正确性（n=200k）===")
    report("5 类均匀", np.array([0.2] * 5), sample_gumbel_ratio)
    report("5 类偏斜", np.array([0.5, 0.25, 0.15, 0.08, 0.02]), sample_gumbel_ratio)
    report("教科书式", np.array([0.5, 0.25, 0.15, 0.08, 0.02]), sample_gumbel_max)

    print("\n=== 温度的效果：logits 除以 T 后再 softmax（sampler.py:9）===")
    logits = np.array([2.0, 1.0, 0.5, 0.0])
    for T in (0.3, 1.0, 2.0, 10.0):
        z = logits / T
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        emp = np.bincount(sample_gumbel_ratio(np.repeat(p[None, :], 20000, 0)),
                          minlength=len(p)) / 20000
        print(f"T={T:<5} p={np.round(p, 3)}  最可能 token 命中率={emp.max():.3f}")

    print("\n=== 为什么 assert temperature > 1e-10（sampling_params.py:11）===")
    z = logits / 1e-12
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    print(f"T→0 时 softmax 退化为 one-hot：p={np.round(p, 6)} → argmax 恒定，"
          f"greedy 应由单独的采样路径实现，作者选择直接禁止。")


if __name__ == "__main__":
    main()
