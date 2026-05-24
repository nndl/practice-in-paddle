"""下载本书各章节所需的数据集到本地。

每次运行会跳过已存在的文件（按目标路径判断），可以反复运行。

数据放置位置（与 notebook 中的相对路径一致）：
  dataset/boston_house_prices.csv                  — chap2
  chap5卷积神经网络/datasets/cifar-10-batches-py/  — chap5
  chap6循环神经网络/dataset/{train,dev,test,vocab}.txt — chap6 IMDB
  chap8注意力机制/dataset/{train,dev,test,vocab}.txt    — chap8 IMDB
  chap8注意力机制/lcqmc/{train,dev,test}.txt        — chap8 LCQMC

MNIST 不在此处下载——`paddle.vision.datasets.MNIST(download=True)` 会自动下载。
Iris 通过 `sklearn.datasets.load_iris()` 直接取，无需下载。

用法（在仓库根目录）：
  python dataset/download.py
  python dataset/download.py --only=boston,lcqmc   # 只下载指定数据集
"""
import argparse
import os
import random
import re
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset"
CHAP5_DIR = ROOT / "chap5卷积神经网络" / "datasets"
CHAP6_DIR = ROOT / "chap6循环神经网络" / "dataset"
CHAP8_IMDB_DIR = ROOT / "chap8注意力机制" / "dataset"
CHAP8_LCQMC_DIR = ROOT / "chap8注意力机制" / "lcqmc"

SOURCES = {
    "boston": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "lcqmc": "https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz",
}


def log(msg):
    print(f"[download] {msg}", flush=True)


def _download(url: str, dest: Path):
    if dest.exists():
        log(f"skip (exists): {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"GET {url}")
    log(f"  -> {dest.relative_to(ROOT)}")

    def hook(blocks, block_size, total_size):
        if total_size > 0:
            done = blocks * block_size
            pct = min(100, done * 100 // total_size)
            mb_done = done / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)


# ----------------- Boston housing -----------------
def fetch_boston():
    dest = DATASET_ROOT / "boston_house_prices.csv"
    _download(SOURCES["boston"], dest)
    # The selva86 CSV uses lowercase column names; chap2 expects boston_house_prices.csv format.
    # Both work with pandas; no schema rewrite needed.


# ----------------- CIFAR-10 -----------------
def fetch_cifar10():
    target_dir = CHAP5_DIR / "cifar-10-batches-py"
    if target_dir.exists() and any(target_dir.iterdir()):
        log(f"skip (exists): {target_dir.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "cifar-10-python.tar.gz"
    _download(SOURCES["cifar10"], tar_path)
    log(f"extracting CIFAR-10 -> {CHAP5_DIR.relative_to(ROOT)}")
    CHAP5_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(CHAP5_DIR)
    log("CIFAR-10 ready")


# ----------------- IMDB -----------------
def fetch_imdb():
    # Already processed?
    chap6_train = CHAP6_DIR / "train.txt"
    if chap6_train.exists():
        log(f"skip (exists): {chap6_train.relative_to(ROOT)}")
        return

    tar_path = DATASET_ROOT / "aclImdb_v1.tar.gz"
    extract_dir = DATASET_ROOT / "_aclImdb_extract"
    _download(SOURCES["imdb"], tar_path)

    if not extract_dir.exists():
        log("extracting aclImdb (this takes a minute, ~50k small files)...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        log("aclImdb extracted")

    log("processing IMDB into train/dev/test.txt + vocab.txt")
    aclImdb = extract_dir / "aclImdb"

    def collect(split):
        items = []
        for label, label_dir in [("1", "pos"), ("0", "neg")]:
            d = aclImdb / split / label_dir
            for fp in sorted(d.glob("*.txt")):
                text = fp.read_text(encoding="utf-8", errors="ignore")
                # crude cleanup: <br /> -> space, drop control chars
                text = text.replace("<br />", " ")
                text = re.sub(r"\s+", " ", text).strip().lower()
                items.append((label, text))
        return items

    train_all = collect("train")  # 25k
    test_items = collect("test")  # 25k

    rng = random.Random(42)
    rng.shuffle(train_all)
    dev_size = 5000
    dev_items = train_all[:dev_size]
    train_items = train_all[dev_size:]

    def write_split(items, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fw:
            for label, text in items:
                fw.write(f"{label}\t{text}\n")

    # Build vocab from train (top 50k words by frequency, plus special tokens)
    counter = Counter()
    for _, text in train_items:
        counter.update(text.split(" "))
    vocab_tokens = ["[PAD]", "[UNK]"] + [w for w, _ in counter.most_common(50000) if w]

    for target_dir in [CHAP6_DIR, CHAP8_IMDB_DIR]:
        write_split(train_items, target_dir / "train.txt")
        write_split(dev_items, target_dir / "dev.txt")
        write_split(test_items, target_dir / "test.txt")
        vocab_path = target_dir / "vocab.txt"
        with vocab_path.open("w", encoding="utf-8") as fw:
            for tok in vocab_tokens:
                fw.write(tok + "\n")
        log(f"  written: {target_dir.relative_to(ROOT)} (train/dev/test/vocab)")

    # Clean up extracted folder to save disk; keep tar in case user wants to re-process
    log("cleaning up extracted aclImdb (keeping the tar.gz)...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    log("IMDB ready")


# ----------------- LCQMC -----------------
def fetch_lcqmc():
    target_dir = CHAP8_LCQMC_DIR
    train_path = target_dir / "train.txt"
    if train_path.exists():
        log(f"skip (exists): {train_path.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "lcqmc.tar.gz"
    extract_dir = DATASET_ROOT / "_lcqmc_extract"
    _download(SOURCES["lcqmc"], tar_path)

    log("extracting LCQMC")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    # paddle hub's tar typically contains lcqmc/{train,dev,test}.tsv with header
    # Find the actual layout and normalize to chap8 expected format.
    candidates = list(extract_dir.rglob("train.*"))
    log(f"  LCQMC extracted entries: {[p.relative_to(extract_dir) for p in candidates]}")
    src_dir = candidates[0].parent if candidates else None
    if src_dir is None:
        raise RuntimeError("LCQMC tar didn't produce a train.* file")

    target_dir.mkdir(parents=True, exist_ok=True)

    def copy_normalize(src: Path, dst: Path):
        with src.open("r", encoding="utf-8") as fr, dst.open("w", encoding="utf-8") as fw:
            for i, line in enumerate(fr):
                # paddle hub format: text_a\ttext_b\tlabel  (with possible header on line 0)
                parts = line.rstrip("\n").split("\t")
                if i == 0 and not (parts and parts[-1].isdigit()):
                    continue  # header line
                fw.write(line if line.endswith("\n") else line + "\n")

    for split in ("train", "dev", "test"):
        src = src_dir / f"{split}.tsv"
        if not src.exists():
            src = src_dir / f"{split}.txt"
        if not src.exists():
            log(f"  warning: {split} file not found in {src_dir}")
            continue
        copy_normalize(src, target_dir / f"{split}.txt")
        log(f"  written: {(target_dir / f'{split}.txt').relative_to(ROOT)}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    log("LCQMC ready")


# ----------------- main -----------------
FETCHERS = {
    "boston": fetch_boston,
    "cifar10": fetch_cifar10,
    "imdb": fetch_imdb,
    "lcqmc": fetch_lcqmc,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated subset, e.g. boston,lcqmc",
    )
    args = parser.parse_args()
    DATASET_ROOT.mkdir(exist_ok=True)
    selected = [k.strip() for k in args.only.split(",") if k.strip()] or list(FETCHERS.keys())
    for name in selected:
        if name not in FETCHERS:
            log(f"unknown dataset '{name}', skipping (valid: {list(FETCHERS.keys())})")
            continue
        log(f"=== {name} ===")
        try:
            FETCHERS[name]()
        except Exception as e:
            log(f"  ERROR for {name}: {type(e).__name__}: {e}")
            raise
    log("all done")


if __name__ == "__main__":
    main()
