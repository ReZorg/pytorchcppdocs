#!/usr/bin/env bash
# CI-style validation: configure and build every generated target against a
# LibTorch install. Pass the LibTorch prefix as $1 or set Torch_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/pytorchcppdocs-build}"
CMAKE_ARGS=(-S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${1:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$1")
fi
cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
echo "All generated examples and stubs built successfully."
