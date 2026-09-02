#!/usr/bin/env bash
TORCH=/home/runner/.local/lib/python3.12/site-packages/torch
g++ -std=c++20 -D_GLIBCXX_USE_CXX11_ABI=1 -fsyntax-only "$1" \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include 2>&1
