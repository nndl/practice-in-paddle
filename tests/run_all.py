"""Run all per-chapter virtual tests and print an aggregate summary."""
import importlib
import sys
import time
from pathlib import Path

TESTS = [f"test_chap{n}" for n in range(1, 9)]

if __name__ == "__main__":
    # ensure tests/ is on sys.path so each test_chapN.py can import _common
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    overall_pass = 0
    overall_fail = 0
    chapter_results = []

    for name in TESTS:
        print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
        t0 = time.time()
        mod = importlib.import_module(name)
        ok = mod.main()
        elapsed = time.time() - t0
        chapter_results.append((name, ok, elapsed))
        if ok:
            overall_pass += 1
        else:
            overall_fail += 1

    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    for name, ok, elapsed in chapter_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}  ({elapsed:.1f}s)")
    print(f"\nChapters passed: {overall_pass}/{len(TESTS)}")
    sys.exit(0 if overall_fail == 0 else 1)
