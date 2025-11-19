#!/usr/bin/env bash
set -euo pipefail
source /app/.venv/bin/activate

echo "[ensure-uv] Checking numpy/pymc/pytensor imports…"
python - <<'PY'
import sys
def v(m):
    try:
        mod = __import__(m)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:
        print(f"[ensure-uv] import {m} failed: {e}")
        sys.exit(42)
print("[ensure-uv] numpy=", v("numpy"), "pymc=", v("pymc"), "pytensor=", v("pytensor"))
PY
rc=$?

if [ $rc -ne 0 ]; then
  echo "[ensure-uv] Detected broken imports; reconciling with uv sync…"
  (cd /workspace && uv sync --frozen -E torchcu124)
  echo "[ensure-uv] Recheck:"
  python - <<'PY'
import numpy, pymc, pytensor
print("[ensure-uv] ✅ fixed:",
      "numpy", numpy.__version__,
      "pymc", getattr(pymc, "__version__", "?"),
      "pytensor", getattr(pytensor, "__version__", "?"))
PY
else
  echo "[ensure-uv] Environment looks consistent."
fi

python -m pip check || true
