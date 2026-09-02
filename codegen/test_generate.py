#!/usr/bin/env python3
"""Exhaustive unit tests for codegen/generate.py.

Run with:  python3 -m unittest codegen.test_generate -v   (from the repo root)
     or:   python3 codegen/test_generate.py -v

All tests are hermetic: no network, no LibTorch install, and no writes outside
temporary directories (tests that exercise main() sandbox the generator module
constants so the checked-in generated/ tree is never touched).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "generated"
MANIFEST = GENERATED_DIR / "codegen_manifest.json"


def write_sandbox_docs(root: Path) -> dict:
    """Create a minimal fake doc corpus: one _sources page + _coverage headers."""
    sources = root / "_sources"
    (sources / "notes").mkdir(parents=True)
    (sources / "notes" / "widget.md.txt").write_text(
        "# Widget Notes\n\n"
        "{cpp:class}`torch::Widget`\n\n"
        "{cpp:enum-class}`torch::WidgetKind`\n\n"
        "Header file: `#include <torch/widget.h>`\n\n"
        "- `torch/widget.h`\n\n"
        "```cpp\n"
        "#include <torch/torch.h>\n"
        "int main() {\n"
        "  auto t = torch::ones({2, 2});\n"
        "  return 0;\n"
        "}\n"
        "```\n",
        encoding="utf-8",
    )
    coverage = root / "_coverage"
    (coverage / "torch").mkdir(parents=True)
    (coverage / "torch" / "widget.h.gcov.html").write_text(
        "<html>coverage</html>", encoding="utf-8"
    )
    aten = coverage / "__w" / "pytorch" / "pytorch" / "aten" / "src" / "ATen"
    aten.mkdir(parents=True)
    (aten / "Widget.h.gcov.html").write_text("<html>coverage</html>", encoding="utf-8")
    return {
        "sources": sources,
        "coverage": coverage,
        "llms": root / "llms-full.txt",
        "generated": root / "generated",
    }


@contextlib.contextmanager
def sandboxed_generator(root: Path, dirs: dict):
    """Point the generator's module-level paths at a temporary sandbox."""
    saved = {
        name: getattr(generate, name)
        for name in (
            "REPO_ROOT",
            "SOURCES_DIR",
            "COVERAGE_DIR",
            "LLMS_FULL",
            "GENERATED_DIR",
            "INCLUDE_DIR",
            "SRC_DIR",
            "EXAMPLES_DIR",
        )
    }
    generate.REPO_ROOT = root
    generate.SOURCES_DIR = dirs["sources"]
    generate.COVERAGE_DIR = dirs["coverage"]
    generate.LLMS_FULL = dirs["llms"]
    generate.GENERATED_DIR = dirs["generated"]
    generate.INCLUDE_DIR = dirs["generated"] / "include"
    generate.SRC_DIR = dirs["generated"] / "src"
    generate.EXAMPLES_DIR = dirs["generated"] / "examples"
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(generate, name, value)


class TestSanitizeIdentifier(unittest.TestCase):
    def test_plain_name_is_unchanged(self):
        self.assertEqual(generate.sanitize_identifier("tensor"), "tensor")

    def test_scopes_and_dots_become_underscores(self):
        self.assertEqual(generate.sanitize_identifier("torch::nn.Linear"), "torch_nn_Linear")

    def test_template_args_are_stripped(self):
        self.assertEqual(
            generate.sanitize_identifier("Dataset<CustomDataset>"), "Dataset"
        )

    def test_leading_digit_is_prefixed(self):
        self.assertEqual(generate.sanitize_identifier("3dconv"), "_3dconv")

    def test_empty_string_falls_back_to_unnamed(self):
        self.assertEqual(generate.sanitize_identifier("!!!"), "unnamed")

    def test_repeated_separators_collapse(self):
        self.assertEqual(generate.sanitize_identifier("a--b::c"), "a_b_c")


class TestNameHelpers(unittest.TestCase):
    def test_unqualified(self):
        self.assertEqual(generate.unqualified("torch::nn::Module"), "Module")
        self.assertEqual(generate.unqualified("Module"), "Module")

    def test_leaf_namespace(self):
        self.assertEqual(generate.leaf_namespace("torch::nn::Module"), "torch::nn")
        self.assertEqual(generate.leaf_namespace("Module"), "")

    def test_camel_from_stem(self):
        self.assertEqual(generate.camel_from_stem("ordered_dict"), "OrderedDict")
        self.assertEqual(generate.camel_from_stem("custom-class"), "CustomClass")
        self.assertEqual(generate.camel_from_stem("tensor"), "Tensor")


class TestParsePage(unittest.TestCase):
    def test_directives_blocks_and_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.md.txt"
            page.write_text(
                "# Demo\n\n"
                "```{doxygenclass} torch::data::datasets::Dataset\n"
                "```\n\n"
                "```{doxygenstruct} torch::data::Example\n"
                "```\n\n"
                "```{doxygenfunction} torch::ones(IntArrayRef)\n"
                "```\n\n"
                "```{doxygendefine} TORCH_MODULE\n"
                "```\n\n"
                "```{cpp:class} torch::nn::Module\n"
                "```\n\n"
                "```{cpp:enum-class} torch::DeviceType\n"
                "```\n\n"
                "```{cpp:function} at::Tensor at::rand(IntArrayRef)\n"
                "```\n\n"
                "Headers:\n"
                "- `torch/all.h`\n",
                encoding="utf-8",
            )
            parsed = generate.parse_page(page)

        self.assertIn("torch::data::datasets::Dataset", parsed["classes"])
        self.assertIn("torch::nn::Module", parsed["classes"])
        self.assertIn("torch::data::Example", parsed["structs"])
        self.assertTrue(any(f.startswith("torch::ones") for f in parsed["functions"]))
        self.assertIn("TORCH_MODULE", parsed["defines"])
        self.assertIn("torch::DeviceType", parsed["enums"])
        self.assertTrue(any("at::rand" in f for f in parsed["cpp_functions"]))
        self.assertIn("torch/all.h", parsed["headers"])
        # MyST directive fences pair up under the plain-fence rule, so this
        # page yields only empty blocks (matching real corpus behavior); code
        # block extraction is covered by test_language_tags_and_plain_fences.
        self.assertEqual(parsed["blocks"], [""] * 6)

    def test_language_tags_and_plain_fences(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.md.txt"
            page.write_text(
                "```cpp\nint a = 1;\n```\n\n"
                "```c++\nint b = 2;\n```\n\n"
                "```\nint c = 3;\n```\n\n"
                "```python\nint d = 4;\n```\n",
                encoding="utf-8",
            )
            parsed = generate.parse_page(page)
        self.assertEqual(len(parsed["blocks"]), 3)
        self.assertFalse(any("d = 4" in b for b in parsed["blocks"]))


class TestRealRepoInvariants(unittest.TestCase):
    """Invariants over the real doc corpus (no sandboxing required)."""

    def test_collect_pages_is_sorted_and_nonempty(self):
        pages = generate.collect_pages()
        self.assertGreater(len(pages), 50)
        self.assertEqual(pages, sorted(pages))
        self.assertTrue(all(str(p).endswith(".md.txt") for p in pages))

    def test_manifest_matches_live_counts(self):
        stats = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(stats["pages_parsed"], len(generate.collect_pages()))
        self.assertEqual(stats["coverage_total"], len(generate.coverage_headers()))

    def test_coverage_headers_use_public_include_layout(self):
        headers = generate.coverage_headers()
        self.assertGreater(len(headers), 100)
        self.assertEqual(headers, sorted(set(headers)))
        for header in headers:
            self.assertFalse(header.startswith("__w/"), header)
            self.assertFalse(header.startswith("aten/src/"), header)
        self.assertIn("torch/library.h", headers)
        self.assertIn("torch/csrc/autograd/autograd.h", headers)

    def test_every_llms_page_title_is_found(self):
        titles = []
        for page_path in generate.collect_pages():
            text = page_path.read_text(encoding="utf-8", errors="replace")
            match = generate.re.search(r"^# (.+)$", text, generate.re.MULTILINE)
            if match:
                titles.append(match.group(1).strip())
        corpus = generate.LLMS_FULL.read_text(encoding="utf-8", errors="replace")
        missing = [t for t in titles if t not in corpus]
        self.assertEqual(missing, [])


class TestStubGeneration(unittest.TestCase):
    def test_class_stub_wraps_namespace(self):
        stub = generate.class_stub("torch::nn::Module")
        self.assertIn("namespace torch::nn {", stub)
        self.assertIn("class Module {", stub)
        self.assertIn("}  // namespace torch::nn", stub)

    def test_struct_and_enum_stubs_without_namespace(self):
        struct = generate.struct_stub("Example")
        self.assertNotIn("namespace", struct)
        self.assertIn("struct Example {", struct)
        enum = generate.enum_stub("DeviceType")
        self.assertIn("enum class DeviceType", enum)

    def test_function_stub_parses_signature(self):
        # The parser requires an unqualified function name to split return type.
        stub = generate.function_stub("page", "at::Tensor rand(IntArrayRef size)")
        self.assertIn("inline at::Tensor rand()", stub)
        self.assertIn("return at::Tensor{};", stub)

    def test_function_stub_tolerates_qualified_name(self):
        stub = generate.function_stub("page", "at::Tensor at::rand(IntArrayRef size)")
        self.assertIn("inline void at_Tensor_at_rand()", stub)

    def test_function_stub_defaults_to_void(self):
        stub = generate.function_stub("page", "train()")
        self.assertIn("inline void train()", stub)
        self.assertNotIn("return void{}", stub)

    def test_stub_header_mentions_header_and_symbols(self):
        page = {
            "headers": ["torch/nn/module.h"],
            "classes": ["torch::nn::Module"],
            "structs": [],
            "enums": [],
        }
        header = generate.generate_stub_header("torch/nn/module.h", [page])
        self.assertTrue(header.startswith(generate.HEADER_BANNER))
        self.assertIn("#ifndef GENERATED_TORCH_NN_MODULE_H_", header)
        self.assertIn("#define GENERATED_TORCH_NN_MODULE_H_", header)
        self.assertIn("class Module {", header)
        self.assertTrue(header.endswith("#endif  // GENERATED_TORCH_NN_MODULE_H_\n"))

    def test_stub_header_fallback_when_unreferenced(self):
        header = generate.generate_stub_header("c10/core/Device.h", [])
        self.assertIn("class Device {", header)  # CamelCase fallback from the stem
        self.assertIn("}  // namespace generated", header)


class TestExampleGeneration(unittest.TestCase):
    def test_no_blocks_returns_none(self):
        self.assertIsNone(
            generate.generate_example_from_page(
                {"page": Path("x.md.txt"), "blocks": []}
            )
        )

    def test_example_inlines_main_and_collects_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            page_file = Path(tmp) / "api" / "demo.md.txt"
            page_file.parent.mkdir(parents=True)
            page_file.touch()
            with sandboxed_generator(Path(tmp), {
                "sources": Path(tmp),
                "coverage": Path(tmp),
                "llms": Path(tmp) / "llms-full.txt",
                "generated": Path(tmp) / "gen",
            }):
                example = generate.generate_example_from_page(
                    {
                        "page": page_file,
                        "blocks": [
                            "#include <torch/torch.h>\n"
                            "int main() {\n"
                            "  auto t = torch::zeros({1});\n"
                            "  return 0;\n"
                            "}"
                        ],
                    }
                )
        self.assertIsNotNone(example)
        self.assertEqual(example.count("int main()"), 1)
        self.assertIn("#include <iostream>", example)
        self.assertIn("#include <torch/torch.h>", example)
        self.assertIn("auto t = torch::zeros({1});", example)
        self.assertIn("example completed", example)
        self.assertIn("Doc page: api/demo", example)


class TestHppTemplatesAndSrc(unittest.TestCase):
    def test_hpp_templates_are_guarded(self):
        files = generate.generate_hpp_templates()
        self.assertEqual(
            sorted(files),
            ["torch/data/dataset.hpp", "torch/nn/module.hpp", "torch/ordered_dict.hpp"],
        )
        for rel, content in files.items():
            self.assertTrue(content.startswith(generate.HEADER_BANNER), rel)
            self.assertIn("#ifndef GENERATED_", content, rel)
            self.assertIn("#endif  // GENERATED_", content, rel)
        self.assertIn("class OrderedDict", files["torch/ordered_dict.hpp"])
        self.assertIn("class Cloneable", files["torch/nn/module.hpp"])
        self.assertIn("class Dataset", files["torch/data/dataset.hpp"])

    def test_src_files_match_checked_in_copies(self):
        files = generate.generate_src_files()
        self.assertEqual(sorted(files), ["dataset.cpp", "net.cpp"])
        for name, content in files.items():
            checked_in = (GENERATED_DIR / "src" / name).read_text(encoding="utf-8")
            self.assertEqual(content, checked_in, f"generated/src/{name} is stale")


class TestCmakeLists(unittest.TestCase):
    def test_registers_every_example_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            examples = Path(tmp) / "examples"
            (examples / "aten").mkdir(parents=True)
            (examples / "aten" / "tensor.cpp").touch()
            (examples / "aten" / "creation.cpp").touch()
            cu = examples / "cuda"
            cu.mkdir()
            (cu / "kernel.cu").touch()
            with sandboxed_generator(Path(tmp), {
                "sources": Path(tmp),
                "coverage": Path(tmp),
                "llms": Path(tmp) / "llms-full.txt",
                "generated": Path(tmp),
            }):
                cmake = generate.cmake_lists(
                    sorted(examples.rglob("*.cpp")), sorted(examples.rglob("*.cu"))
                )
        self.assertIn("enable_testing()", cmake)
        self.assertIn("add_executable(example_aten_tensor examples/aten/tensor.cpp)", cmake)
        self.assertIn("add_executable(example_aten_creation examples/aten/creation.cpp)", cmake)
        self.assertIn("add_test(NAME aten_tensor COMMAND example_aten_tensor)", cmake)
        self.assertIn("add_test(NAME aten_creation COMMAND example_aten_creation)", cmake)
        # CUDA targets stay gated behind compiler detection and LibTorch.
        self.assertIn("if(CMAKE_CUDA_COMPILER AND Torch_FOUND)", cmake)
        self.assertIn("add_executable(example_cuda_kernel examples/cuda/kernel.cu)", cmake)
        self.assertIn("add_test(NAME cuda_kernel COMMAND example_cuda_kernel)", cmake)
        self.assertIn("if(MSVC AND Torch_FOUND)", cmake)
        # Each example appears as: executable, link, std, test command, DLL list.
        self.assertEqual(cmake.count("example_aten_tensor"), 5)

    def test_checked_in_cmake_lists_every_example(self):
        cmake = (GENERATED_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        examples = sorted((GENERATED_DIR / "examples").rglob("*.cpp"))
        self.assertGreaterEqual(len(examples), 50)
        for example in examples:
            rel = example.relative_to(GENERATED_DIR).as_posix()
            stem = generate.re.sub(
                r"[^A-Za-z0-9]", "_",
                example.relative_to(GENERATED_DIR / "examples").with_suffix("").as_posix(),
            )
            self.assertIn(f"add_executable(example_{stem} {rel})", cmake)
            self.assertIn(f"add_test(NAME {stem} COMMAND example_{stem})", cmake)
        self.assertIn("enable_testing()", cmake)

    def test_cmake_output_matches_checked_in_file(self):
        example_files = sorted(generate.EXAMPLES_DIR.rglob("*.cpp"))
        cu_files = sorted(generate.EXAMPLES_DIR.rglob("*.cu"))
        expected = generate.cmake_lists(example_files, cu_files)
        checked_in = (GENERATED_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertEqual(expected, checked_in)


class TestCheckExtensions(unittest.TestCase):
    def test_rejects_disallowed_extensions(self):
        paths = [Path("a.h"), Path("b.c"), Path("c.hh"), Path("d.py")]
        with self.assertRaises(SystemExit):
            generate.check_extensions(paths)

    def test_accepts_documented_extensions_and_build_files(self):
        paths = [
            Path("a.h"), Path("b.hpp"), Path("c.cpp"), Path("d.cu"),
            Path("e.txt"), Path("f.md"), Path("g.json"),
            Path("CMakeLists.txt"), Path("validate.sh"),
        ]
        generate.check_extensions(paths)  # must not raise

    def test_generated_tree_uses_only_allowed_extensions(self):
        paths = [p for p in GENERATED_DIR.rglob("*") if p.is_file()]
        self.assertGreater(len(paths), 200)
        generate.check_extensions(paths)  # must not raise


class TestMainSandboxed(unittest.TestCase):
    def test_main_generates_full_tree_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = write_sandbox_docs(root)
            dirs["llms"].write_text("# Widget Notes\n", encoding="utf-8")
            with sandboxed_generator(root, dirs):
                generate.main()
                generated = dirs["generated"]
                manifest = json.loads(
                    (generated / "codegen_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["pages_parsed"], 1)
                self.assertEqual(manifest["coverage_total"], 2)
                self.assertEqual(manifest["stub_headers"], 5)  # 2 mirrored + 3 .hpp
                self.assertEqual(manifest["examples"], 1)
                self.assertEqual(manifest["gap_examples"], 1)

                # Mirrored stub headers exist at the public include layout.
                self.assertTrue((generated / "include" / "torch" / "widget.h").exists())
                self.assertTrue((generated / "include" / "ATen" / "Widget.h").exists())
                # Template stubs, src implementations, and the gap example.
                self.assertTrue((generated / "include" / "torch" / "nn" / "module.hpp").exists())
                self.assertTrue((generated / "src" / "dataset.cpp").exists())
                example = generated / "examples" / "notes" / "widget.cpp"
                self.assertTrue(example.exists())
                self.assertIn("int main()", example.read_text(encoding="utf-8"))
                # Build scaffolding.
                cmake = (generated / "CMakeLists.txt").read_text(encoding="utf-8")
                self.assertIn("add_executable(example_notes_widget", cmake)
                self.assertIn("add_test(NAME notes_widget", cmake)
                validate = generated / "validate.sh"
                self.assertTrue(os.access(validate, os.X_OK))
                self.assertTrue((generated / "coverage_report.md").exists())
                self.assertTrue((generated / "README.md").exists())

                # Second run is a no-op (idempotent regeneration). The manifest
                # gap_examples counter is run metadata (0 on a no-op rerun), so
                # compare the emitted trees rather than raw bytes.
                snapshot = {
                    p.relative_to(generated): p.read_bytes()
                    for p in generated.rglob("*")
                    if p.is_file() and p.name != "codegen_manifest.json"
                }
                generate.main()
                snapshot2 = {
                    p.relative_to(generated): p.read_bytes()
                    for p in generated.rglob("*")
                    if p.is_file() and p.name != "codegen_manifest.json"
                }
                self.assertEqual(snapshot, snapshot2)
                manifest2 = json.loads(
                    (generated / "codegen_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest2["gap_examples"], 0)
                self.assertEqual(manifest2["examples"], manifest["examples"])

                # A curated example is never overwritten by gap-fill.
                curated = generated / "examples" / "notes" / "widget.cpp"
                curated.write_text("// curated\nint main() { return 0; }\n", encoding="utf-8")
                generate.main()
                self.assertEqual(
                    curated.read_text(encoding="utf-8"),
                    "// curated\nint main() { return 0; }\n",
                )

    def test_main_warns_for_missing_llms_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = write_sandbox_docs(root)
            dirs["llms"].write_text("# Unrelated\n", encoding="utf-8")
            stderr = io.StringIO()
            with sandboxed_generator(root, dirs), contextlib.redirect_stderr(stderr):
                generate.main()
            self.assertIn("Widget Notes", stderr.getvalue())


class TestGeneratedTreeConsistency(unittest.TestCase):
    """The checked-in generated/ tree must stay in sync with the generator."""

    def test_validate_script_matches_generator(self):
        self.assertEqual(
            generate.validate_script(),
            (GENERATED_DIR / "validate.sh").read_text(encoding="utf-8"),
        )

    def test_validate_script_is_executable(self):
        mode = (GENERATED_DIR / "validate.sh").stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR)

    def test_readme_matches_generator(self):
        stats = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            generate.readme(stats),
            (GENERATED_DIR / "README.md").read_text(encoding="utf-8"),
        )

    def test_manifest_counts_match_tree(self):
        stats = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(stats["examples"], 52)
        self.assertEqual(
            stats["examples"],
            len(list((GENERATED_DIR / "examples").rglob("*.cpp")))
            + len(list((GENERATED_DIR / "examples").rglob("*.cu"))),
        )
        self.assertEqual(
            stats["stub_headers"],
            len(list((GENERATED_DIR / "include").rglob("*.h")))
            + len(list((GENERATED_DIR / "include").rglob("*.hpp"))),
        )

    def test_coverage_report_has_no_gaps(self):
        report = (GENERATED_DIR / "coverage_report.md").read_text(encoding="utf-8")
        self.assertIn("Documented headers without a stub: **0**", report)


class TestGeneratedStubsCompile(unittest.TestCase):
    """Header stubs must be self-contained and compile without LibTorch."""

    def test_hpp_templates_compile_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "smoke.cpp"
            source.write_text(
                "#include <torch/nn/module.hpp>\n"
                "#include <torch/data/dataset.hpp>\n"
                "#include <torch/ordered_dict.hpp>\n"
                "struct Net : generated::nn::Cloneable<Net> {};\n"
                "struct Row : generated::data::Dataset<Row, int> {\n"
                "  int get(size_t) override { return 0; }\n"
                "  size_t size() const override { return 1; }\n"
                "};\n"
                "int main() {\n"
                "  Net net;\n"
                "  net.train(false);\n"
                "  Row row;\n"
                "  generated::OrderedDict<int, int> dict;\n"
                "  dict.insert(1, 2);\n"
                "  return dict.at(1) == 2 && !net.is_training() && row.size() == 1 ? 0 : 1;\n"
                "}\n",
                encoding="utf-8",
            )
            binary = Path(tmp) / "smoke"
            compile_cmd = [
                "c++", "-std=c++20",
                "-I", str(GENERATED_DIR / "include"),
                str(source), "-o", str(binary),
            ]
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_representative_stub_headers_compile_standalone(self):
        samples = [
            "torch/library.h",
            "torch/nn/module.h",
            "ATen/core/Tensor.h",
            "c10/core/Device.h",
        ]
        for header in samples:
            self.assertTrue(
                (GENERATED_DIR / "include" / header).exists(),
                f"missing stub: {header}",
            )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "headers.cpp"
            source.write_text(
                "".join(f'#include "{h}"\n' for h in samples) + "int main() { return 0; }\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "c++", "-std=c++20", "-fsyntax-only",
                    "-I", str(GENERATED_DIR / "include"),
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
