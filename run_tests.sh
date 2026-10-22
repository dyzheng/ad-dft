#!/usr/bin/env bash
# 逐文件运行：XLA 编译缓存在单进程内累积到 GB 级，会让长会话 OOM。
# CI 里也是分文件并行跑（这也顺便给出每个门禁的独立时间预算）。
set -u
export PYTHONPATH="$(pwd)/src:$(pwd)/tests"
PY=${PYTHON:-python}
fail=0
for f in tests/test_*.py; do
  echo "── $f"
  "$PY" -m pytest "$f" 2>&1 | tail -1
  [ "${PIPESTATUS[0]}" -ne 0 ] && fail=1
done

if [ "${ADDFT_RUN_ABACUS:-0}" = "1" ]; then
  PYTHONPATH=src:tests:. "$PY" -m pytest tests/test_abacus_regression.py -q -m abacus
  [ $? -ne 0 ] && fail=1
fi

exit $fail
