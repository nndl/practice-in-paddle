# 数据集

## 自动下载

在仓库根目录执行：

```bash
python dataset/download.py             # 下载全部
python dataset/download.py --only=boston,lcqmc  # 只下载指定数据集
```

脚本可重复执行，已下载的文件会自动跳过。

数据来源全部为公开镜像，无需 AI Studio 账号：

| 数据集 | 来源 | 大小 | 用于 |
|---|---|---|---|
| Boston Housing | github.com/selva86/datasets | 30 KB | chap2 |
| CIFAR-10 | cs.toronto.edu | 163 MB | chap5 |
| IMDB (aclImdb_v1) | ai.stanford.edu | 80 MB（处理后 ~80MB） | chap6, chap8 |
| LCQMC | bj.bcebos.com/paddlehub-dataset | 6 MB | chap8 |
| MNIST | `paddle.vision.datasets.MNIST(download=True)` 自动下载 | 11 MB | chap5, chap7 |
| Iris | `sklearn.datasets.load_iris()` 内置 | — | chap3, chap4 |

## 落地路径

下载脚本会把数据放到 notebook 期望的相对路径上：

```
dataset/
  boston_house_prices.csv          # chap2 用
  download.py                       # 下载脚本本身
  README.md

chap5卷积神经网络/
  datasets/cifar-10-batches-py/    # CIFAR-10 原始格式

chap6循环神经网络/
  dataset/
    train.txt                       # IMDB 训练集 20000 条
    dev.txt                         # IMDB 开发集 5000 条
    test.txt                        # IMDB 测试集 25000 条
    vocab.txt                       # 词表（[PAD], [UNK] + top 50000 词）

chap8注意力机制/
  dataset/                          # IMDB（与 chap6 同源同处理）
    train.txt
    dev.txt
    test.txt
    vocab.txt
  lcqmc/                            # LCQMC 中文问题匹配
    train.txt                       # 238766 条
    dev.txt                         # 8802 条
    test.txt                        # 12500 条
```

## 与 AI Studio 版本的差异

- **chap2 boston_house_prices.csv**：notebook 里的路径是 `/home/aistudio/work/boston_house_prices.csv`（AI Studio 绝对路径）。本地运行需要把 cell 里的路径改成 `../dataset/boston_house_prices.csv` 或 `dataset/boston_house_prices.csv`（取决于 jupyter 启动目录）。
- **chap5 CIFAR-10**：notebook 路径写的是 `/home/aistudio/datasets/cifar-10-batches-py/`，本地为 `chap5卷积神经网络/datasets/cifar-10-batches-py/`。如果 jupyter 从 chap5 目录启动，把绝对路径改成 `datasets/cifar-10-batches-py/` 即可。
- **IMDB**：Stanford 原始数据没有自带 dev split，下载脚本从 train 25000 条中随机抽 5000 条作为 dev（固定种子 42），剩余 20000 条作为 train。test 保持原 25000 条。词表从 train 词频生成（top 50000）。
- **LCQMC**：与 AI Studio 版本基本一致；本镜像的 dev 是 8802 条、test 是 12500 条（与第 1 章描述吻合，与第 8 章正文里"4401/4401"的描述略有差异，但代码不依赖具体条数）。

## 备注

如果你已经有 AI Studio 课程账号，也可以从 [AI Studio 课程](https://aistudio.baidu.com/aistudio/education/group/info/25793) 下载（fork 项目即可访问），格式应该兼容。
