"""验证已下载的数据集能被 notebook 期望的代码路径读取。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TestReporter


def main():
    r = TestReporter("dataset integrity")

    def t_boston():
        import pandas as pd
        path = ROOT / "dataset" / "boston_house_prices.csv"
        df = pd.read_csv(path)
        assert len(df) == 506, f"got {len(df)} rows"
        assert df.shape[1] == 14, f"got {df.shape[1]} cols"

    def t_cifar10():
        import pickle
        cifar_dir = ROOT / "dataset" / "cifar-10-batches-py"
        with (cifar_dir / "data_batch_1").open("rb") as fr:
            d = pickle.load(fr, encoding="latin1")
        assert "data" in d and "labels" in d
        assert d["data"].shape == (10000, 3072)
        assert len(d["labels"]) == 10000

    def t_imdb():
        import gzip
        ds_dir = ROOT / "dataset" / "imdb"
        def _read(name):
            with gzip.open(ds_dir / name, "rt", encoding="utf-8") as fr:
                return fr.read().splitlines()
        train = _read("train.txt.gz")
        dev = _read("dev.txt.gz")
        test = _read("test.txt.gz")
        vocab = _read("vocab.txt.gz")
        assert len(train) == 20000, f"train: {len(train)}"
        assert len(dev) == 5000, f"dev: {len(dev)}"
        assert len(test) == 25000, f"test: {len(test)}"
        assert vocab[0] == "[PAD]"
        assert vocab[1] == "[UNK]"
        label, text = train[0].split("\t", maxsplit=1)
        assert label in ("0", "1")
        assert len(text) > 0

    def t_imdb_loader():
        # Mimic chap6/8 notebook's load_imdb_data (gzip-aware)
        import gzip
        def load_imdb_data(path):
            sets = {}
            for split in ("train", "dev", "test"):
                examples = []
                with gzip.open(os.path.join(path, f"{split}.txt.gz"), "rt", encoding="utf-8") as fr:
                    for line in fr:
                        label, text = line.strip().split("\t", maxsplit=1)
                        examples.append((text, label))
                sets[split] = examples
            return sets["train"], sets["dev"], sets["test"]
        train_data, dev_data, test_data = load_imdb_data(
            str(ROOT / "dataset" / "imdb"))
        assert len(train_data) == 20000
        text, label = train_data[0]
        assert label in ("0", "1")

    def t_load_vocab():
        # Mimic chap6/8 notebook's load_vocab → word2id dict (gzip-aware)
        import gzip
        def load_vocab(path):
            opener = gzip.open if str(path).endswith(".gz") else open
            d = {}
            with opener(path, "rt", encoding="utf-8") as fr:
                for i, line in enumerate(fr):
                    d[line.strip()] = i
            return d
        word2id = load_vocab(str(ROOT / "dataset" / "imdb" / "vocab.txt.gz"))
        assert "[PAD]" in word2id and word2id["[PAD]"] == 0
        assert "[UNK]" in word2id and word2id["[UNK]"] == 1
        for w in ("the", "movie", "is"):
            assert w in word2id, f"common word '{w}' missing from vocab"

    def t_lcqmc():
        import gzip
        ld = ROOT / "dataset" / "lcqmc"
        with gzip.open(ld / "train.txt.gz", "rt", encoding="utf-8") as fr:
            train = fr.read().splitlines()
        assert len(train) > 200000, f"train: {len(train)}"
        parts = train[0].split("\t")
        assert len(parts) == 3
        assert parts[2] in ("0", "1")

    def t_imdb_dataset_paddle():
        # Build a paddle.io.Dataset from the IMDB data and check forward pass
        import gzip
        import paddle
        from paddle.io import Dataset, DataLoader

        def load_vocab(path):
            d = {}
            with gzip.open(path, "rt", encoding="utf-8") as fr:
                for i, line in enumerate(fr):
                    d[line.strip()] = i
            return d

        ds_dir = ROOT / "dataset" / "imdb"
        word2id = load_vocab(str(ds_dir / "vocab.txt.gz"))

        class IMDBDataset(Dataset):
            def __init__(self, path, word2id, n=64):
                super().__init__()
                self.examples = []
                with gzip.open(path, "rt", encoding="utf-8") as fr:
                    for i, line in enumerate(fr):
                        if i >= n:
                            break
                        label, text = line.strip().split("\t", maxsplit=1)
                        ids = [word2id.get(w, word2id["[UNK]"]) for w in text.split(" ")[:64]]
                        ids += [word2id["[PAD]"]] * (64 - len(ids))
                        self.examples.append((ids, int(label)))
            def __len__(self): return len(self.examples)
            def __getitem__(self, i):
                ids, label = self.examples[i]
                return paddle.to_tensor(ids, dtype='int64'), label

        ds = IMDBDataset(str(ds_dir / "train.txt.gz"), word2id, n=64)
        loader = DataLoader(ds, batch_size=8)
        for batch_x, batch_y in loader:
            assert batch_x.shape == [8, 64]
            assert batch_y.shape == [8]
            break

    r.run("boston_house_prices.csv (506 rows × 14 cols)", t_boston)
    r.run("CIFAR-10 batch unpickle (10000 × 3072)", t_cifar10)
    r.run("IMDB files + format", t_imdb)
    r.run("IMDB load_imdb_data() shape", t_imdb_loader)
    r.run("IMDB vocab.txt loadable as word2id", t_load_vocab)
    r.run("LCQMC train.txt format (text_a\\ttext_b\\tlabel)", t_lcqmc)
    r.run("IMDB → paddle.io.DataLoader full path", t_imdb_dataset_paddle)
    return r.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
