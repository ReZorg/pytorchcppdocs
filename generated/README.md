# Generated C++ code from the PyTorch C++ docs

This tree is produced by `codegen/generate.py` from the documentation in this
repository (`_sources/**`, `llms-full.txt`, `_coverage/**`). It mirrors the
documented PyTorch C++ API as a buildable codebase.

> Provenance: this repository is auto-generated from
> `pytorch/pytorch` `docs/cpp/source`. Doc content changes belong upstream;
> only the generator and generated layout are maintained here.

## Layout

- `include/` — API-surface stub headers mirroring the real LibTorch header
  paths recorded under `_coverage/` (63 stubs generated).
- `include/torch/**/*.hpp` — header-only template stubs (`nn::Module`,
  `data::Dataset`, `OrderedDict`).
- `src/` — non-template reference implementations compiled into the
  `generated_stubs` static library.
- `examples/<module>/` — one runnable example per documentation page
  (48 examples). Each example has a single `main()` and
  doubles as a compile-test / smoke test.
- `coverage_report.md` — diff of the generated `include/` tree against the
  166 documented headers in `_coverage/`.

## File-type conventions

| Extension | Role |
|-----------|------|
| `.h`      | Public API surface stubs, umbrella headers, stable-ABI / C-compatible interface |
| `.hpp`    | Template-heavy, header-only implementations (nn/data templates, options) |
| `.cpp`    | Non-template implementations and every compilable example program |
| `.cu`     | CUDA-only kernels/examples, built only when a CUDA compiler is found |

`.c` and `.hh` files are deliberately **not** generated: neither extension is
used anywhere in the documented PyTorch C++ API.

## Build

```bash
./validate.sh /path/to/libtorch        # or: cmake -S . -B build -DCMAKE_PREFIX_PATH=...
cmake --build build -j$(nproc)
```

The build follows the documented LibTorch CMake pattern
(`find_package(Torch REQUIRED)`, `${TORCH_CXX_FLAGS}`, C++20, MSVC DLL copy).
CUDA `.cu` targets are only configured when `check_language(CUDA)` finds NVCC.

## Regenerating

```bash
python3 ../codegen/generate.py
```

Curated examples (hand-written, doc-derived) are preserved on re-run; the
generator only fills gaps for pages without a curated example.
