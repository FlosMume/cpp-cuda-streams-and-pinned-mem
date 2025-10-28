#!/usr/bin/env bash
set -euo pipefail

echo "== GPU Info =="
nvidia-smi || echo "nvidia-smi not found"

echo
echo "== Build =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo
echo "== Micro-benchmark =="
N=${N:-16777216} N_STREAMS=${N_STREAMS:-4} FLOP_ITERS=${FLOP_ITERS:-256} ./build/overlap_streams || true
