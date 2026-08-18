"""Import every module in the package with the optional extras blocked out.

The runner has no Blackwell GPU, so kernels cannot execute here. Importing them
still proves the package parses and its relative imports resolve.

The second pass is the interesting one. README promises `pip install cutile-gpt`
works with "minimal dependencies", which means no module may *require* an extra
at import time. Rather than inspect what happens to be installed, this hides
transformers/torch/datasets/tiktoken behind an import blocker and re-imports the
package from scratch: if any module needs one of them eagerly, it fails here.
That works the same whether or not the extras are present, so it behaves
identically on a developer machine and on a bare CI runner.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback

PACKAGE = "cutile_gpt"
OPTIONAL = ("transformers", "torch", "datasets", "tiktoken")


class Blocker:
    """Meta path finder that makes the optional extras look uninstalled."""

    def __init__(self, names):
        self.names = tuple(names)

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError(f"{fullname} is blocked by check_imports.py")
        return None


def module_names():
    pkg = importlib.import_module(PACKAGE)
    return [PACKAGE] + [
        m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{PACKAGE}.")
    ]


def import_all(names, label):
    failures = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc()))
    if failures:
        print(f"\n{label}: {len(failures)} module(s) failed", file=sys.stderr)
        for name, tb in failures:
            print(f"--- {name} ---\n{tb}", file=sys.stderr)
    else:
        print(f"{label}: all {len(names)} modules imported")
    return failures


def main() -> int:
    names = module_names()
    print(f"{PACKAGE} {sys.modules[PACKAGE].__version__}\n")

    failures = import_all(names, "with whatever is installed")

    # Drop the package and the extras, then re-import with the extras blocked.
    for name in list(sys.modules):
        if name.split(".")[0] in (PACKAGE,) + OPTIONAL:
            del sys.modules[name]
    sys.meta_path.insert(0, Blocker(OPTIONAL))
    failures += import_all(names, f"with {', '.join(OPTIONAL)} blocked")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
