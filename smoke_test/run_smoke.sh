#!/usr/bin/env bash
set -euo pipefail

echo "Python:"
python -V
echo "pip:"
python -m pip -V
echo

python -m pip install -U pip

PKG="owrf1d"
if [ -n "${OWRF1D_VERSION:-}" ]; then
  PKG="owrf1d==${OWRF1D_VERSION}"
fi

echo "Package spec: ${PKG}"
echo

echo "=== Phase A: wheel-only probe (optional) ==="
set +e
python -m pip install --no-cache-dir --only-binary=:all: "${PKG}"
WHEEL_RC=$?
set -e

if [ "${WHEEL_RC}" -ne 0 ]; then
  echo "Wheel-only install failed for this python/platform."
  if [ "${STRICT_WHEEL:-0}" = "1" ]; then
    echo "STRICT_WHEEL=1 -> failing."
    exit 2
  fi
  echo "Continuing with normal install (may build from sdist)."
else
  echo "Wheel-only install OK."
fi
echo

echo "=== Phase B: normal install from PyPI ==="
python -m pip install --no-cache-dir -U "${PKG}"
python -m pip check
echo

echo "=== Phase C: runtime smoke checks ==="
python /work/smoke_test/smoke.py
echo

echo "=== Phase D: pure-python fallback sanity ==="
OWRF1D_FORCE_PY=1 python /work/smoke_test/smoke_force_py.py
echo

echo "DONE."
