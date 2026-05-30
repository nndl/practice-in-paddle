# 数据集

## 自动下载

在仓库根目录执行：

```bash
python dataset/download.py             # 下载全部
python dataset/download.py --only=boston,lcqmc  # 只下载指定子集
```

脚本可重复执行，已下载的文件会自动跳过。数据来源全部为公开镜像，无需 AI Studio 账号：

| 数据集 | 来源 | 大小 | 用于 |
|---|---|---|---|
| Boston Housing | github.com/selva86/datasets | 30 KB | chap2 |
| CIFAR-10 | cs.toronto.edu | 163 MB tar | chap5 |
| IMDB (aclImdb_v1) | ai.stanford.edu | 80 MB tar，处理后 ~80MB | chap6, chap8 |
| LCQMC | bj.bcebos.com/paddlehub-dataset | 6 MB tar | chap8 |
| BERT 中文 vocab.txt | ModelScope（国内可达）| 110 KB | chap8 LCQMC Transformer 精确复刻（可选）|
| MNIST | `paddle.vision.datasets.MNIST(download=True)` 自动下载 | 11 MB | chap5, chap7 |
| Iris | `sklearn.datasets.load_iris()` 内置 | — | chap3, chap4 |

## 落地路径（共享一个 dataset/ 目录）

```
dataset/
  boston_house_prices.csv        # chap2
  cifar-10-batches-py/           # chap5（解压后的 pickled batches）
  imdb/                          # chap6 / chap8 共用（gzip 压缩）
    train.txt.gz                 # IMDB 训练集 20000 条
    dev.txt.gz                   # 开发集 5000 条
    test.txt.gz                  # 测试集 25000 条
    vocab.txt.gz                 # 词表（[PAD], [UNK] + top 50000 词）
  lcqmc/                         # chap8（gzip 压缩）
    train.txt.gz                 # 238766 条
    dev.txt.gz                   # 8802 条
    test.txt.gz                  # 12500 条
  bert-base-chinese/             # chap8 LCQMC Transformer 精确复刻（可选）
    vocab.txt                    # bert-base-chinese 字表（21128）
  download.py
  README.md
```

文本数据用 gzip 压缩存放，体积约为原文本的 1/3。notebook 里用 `gzip.open(path, "rt", encoding="utf-8")` 读取。

各章 notebook 从对应目录启动 jupyter，用相对路径 `../dataset/...` 访问。下载脚本会在解压完成后自动清理 tarball。

## 与原 AI Studio 版本的差异

notebook 里原本写的是 AI Studio 绝对路径（`/home/aistudio/work/...`、`/home/aistudio/datasets/...`），本地运行需要改成 `../dataset/...`。我们已在仓库内统一替换。

| 原 AI Studio 路径 | 新路径（本地） |
|---|---|
| `/home/aistudio/work/boston_house_prices.csv` | `../dataset/boston_house_prices.csv` |
| `/home/aistudio/datasets/cifar-10-batches-py/` | `../dataset/cifar-10-batches-py/` |
| `./dataset/train.txt` (chap6/8 IMDB) | `../dataset/imdb/train.txt` |
| `./dataset/vocab.txt` (chap6/8) | `../dataset/imdb/vocab.txt` |
| `'lcqmc'` (chap8) | `'../dataset/lcqmc'` |

## 其他备注

- **IMDB split**：Stanford 原始数据没有自带 dev split。下载脚本从 train 25000 条中随机抽 5000 作为 dev（固定种子 42），剩余 20000 作为 train；test 保持原 25000 条。词表从 train 词频生成（top 50000 + 2 个特殊 token）。
- **LCQMC split**：本镜像 dev 8802 / test 12500，与第 1 章描述一致；第 8 章正文里的"4401/4401"是早期划分，代码不依赖具体条数。
- **磁盘**：`dataset/` 总占用约 208MB —— CIFAR-10 占 178MB（已是二进制 pickle，几乎不可压），IMDB+LCQMC gzip 后仅 30MB，Boston 30KB。下载脚本会自动清理中间 tarballs。
- **AI Studio**：如果你有 AI Studio 课程账号，也可以从 [AI Studio 课程](https://aistudio.baidu.com/aistudio/education/group/info/25793) 下载（fork 项目即可访问）。
