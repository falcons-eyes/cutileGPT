"""Fail if the shipped package imports a third-party module it does not declare.

cutile-gpt 0.2.0 was published with only cupy-cuda12x and numpy in
requires-dist even though every kernel does `import cuda.tile as ct`. Installing
it from PyPI therefore succeeded and then raised ModuleNotFoundError on import.
This check walks the package's own source, collects every top-level module it
imports, and asserts each one is either stdlib, first-party, declared in
[project.dependencies], or explicitly listed below as lazily imported.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

from packaging.requirements import Requirement

PACKAGE = "cutile_gpt"
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Distribution name -> the module name it actually installs.
DIST_TO_MODULE = {
    "cuda-tile": "cuda",
    "cupy-cuda13x": "cupy",
    "cupy-cuda12x": "cupy",
}

# Imported inside a function or try block, guarded by a clear error message,
# and provided by an optional extra rather than the core install.
LAZY_OPTIONAL = {"transformers", "torch", "datasets", "tiktoken",
                 "safetensors", "huggingface_hub"}


def declared_modules(pyproject: dict) -> set[str]:
    mods = set()
    for spec in pyproject["project"].get("dependencies", []):
        name = Requirement(spec).name
        mods.add(DIST_TO_MODULE.get(name, name.replace("-", "_")))
    return mods


def imported_modules(pkg_dir: pathlib.Path) -> dict[str, set[pathlib.Path]]:
    found: dict[str, set[pathlib.Path]] = {}
    for path in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative (first-party) import.
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], set()).add(path.relative_to(ROOT))
    return found


def main() -> int:
    pyproject = tomllib.load((ROOT / "pyproject.toml").open("rb"))
    declared = declared_modules(pyproject)
    stdlib = sys.stdlib_module_names

    undeclared: dict[str, set[pathlib.Path]] = {}
    for module, files in imported_modules(ROOT / PACKAGE).items():
        if module in stdlib or module == PACKAGE:
            continue
        if module in declared or module in LAZY_OPTIONAL:
            continue
        undeclared[module] = files

    if undeclared:
        print("Undeclared third-party imports:\n", file=sys.stderr)
        for module, files in sorted(undeclared.items()):
            where = ", ".join(str(f) for f in sorted(files)[:3])
            print(f"  {module}  (imported by {where})", file=sys.stderr)
        print(
            "\nAdd the distribution to [project.dependencies], or to LAZY_OPTIONAL "
            "in this script if it is an optional extra imported lazily.",
            file=sys.stderr,
        )
        return 1

    print(f"all third-party imports declared ({', '.join(sorted(declared))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
