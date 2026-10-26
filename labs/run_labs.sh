#!/usr/bin/env bash
# 一键跑全部 CPU 实验（不需要 GPU / torch / transformers）
#
# 前置：
#   python3 -m venv ~/.venv-nanovllm-lab
#   ~/.venv-nanovllm-lab/bin/pip install numpy xxhash
#   git clone https://github.com/GeeeekExplorer/nano-vllm ~/nano-vllm
#
# 用法：
#   bash labs/run_labs.sh                 # 用默认路径
#   NANOVLLM_SRC=/path/to/nano-vllm bash labs/run_labs.sh
set -euo pipefail

PY="${PY:-$HOME/.venv-nanovllm-lab/bin/python}"
SRC="${NANOVLLM_SRC:-$HOME/nano-vllm}"

if [ ! -x "$PY" ]; then
  echo "找不到 python：$PY"
  echo "先建环境：python3 -m venv ~/.venv-nanovllm-lab && ~/.venv-nanovllm-lab/bin/pip install numpy xxhash"
  exit 1
fi
if [ ! -d "$SRC/nanovllm" ]; then
  echo "找不到源码：$SRC/nanovllm（用 NANOVLLM_SRC 指定，或 git clone 到 ~/nano-vllm）"
  exit 1
fi

cd "$(dirname "$0")/.."
export NANOVLLM_SRC="$SRC"

echo "== 源码：${SRC}（$(git -C "${SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)）"
echo "== Python：${PY}"

for f in sim_prefix_cache sim_scheduler sim_sampler; do
  echo
  echo "################ $f ################"
  "$PY" "labs/$f.py"
done

echo
echo "全部实验通过。数字来源见 labs/RUN_OUTPUT.md。"
