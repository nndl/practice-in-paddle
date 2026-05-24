"""Shared helpers for per-chapter virtual tests."""
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def push_chapter_pkg(pkg_dirname: str):
    """Add a chapter's directory to sys.path so `from nndl import ...` works.

    Returns the inserted path. Pair with pop_chapter_pkg() in a finally.
    """
    p = str(ROOT / pkg_dirname)
    if p not in sys.path:
        sys.path.insert(0, p)
    # purge any cached nndl modules from a previous chapter
    for name in list(sys.modules):
        if name == "nndl" or name.startswith("nndl."):
            del sys.modules[name]
    return p


def pop_chapter_pkg(p: str):
    if p in sys.path:
        sys.path.remove(p)
    for name in list(sys.modules):
        if name == "nndl" or name.startswith("nndl."):
            del sys.modules[name]


def load_file_module(name: str, path: Path):
    """Load a .py file as a standalone module under a custom name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestReporter:
    def __init__(self, chapter):
        self.chapter = chapter
        self.passed = []
        self.failed = []

    def run(self, label, fn):
        try:
            fn()
            self.passed.append(label)
            print(f"  [PASS] {label}")
        except Exception as e:
            msg = str(e) or repr(e)
            short = (msg.splitlines()[0] if msg.splitlines() else msg)[:200]
            self.failed.append((label, type(e).__name__, short))
            print(f"  [FAIL] {label}: {type(e).__name__}: {short}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n  {self.chapter}: {len(self.passed)}/{total} passed")
        return len(self.failed) == 0
