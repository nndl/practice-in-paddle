import math
import copy
import paddle
import numpy as np
from sklearn.datasets import load_iris

#新增make_moons函数
def make_moons(n_samples=1000, shuffle=True, noise=None):
    """
    生成带噪音的弯月形状数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples]
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    #采集第1类数据，特征为(x,y)
    #使用'paddle.linspace'在0到pi上均匀取n_samples_out个值
    #使用'paddle.cos'计算上述取值的余弦值作为特征1，使用'paddle.sin'计算上述取值的正弦值作为特征2
    outer_circ_x = paddle.cos(paddle.linspace(0, math.pi, n_samples_out))
    outer_circ_y = paddle.sin(paddle.linspace(0, math.pi, n_samples_out))

    inner_circ_x = 1 - paddle.cos(paddle.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 0.5 - paddle.sin(paddle.linspace(0, math.pi, n_samples_in))
    
    print('outer_circ_x.shape:', outer_circ_x.shape, 'outer_circ_y.shape:', outer_circ_y.shape)
    print('inner_circ_x.shape:', inner_circ_x.shape, 'inner_circ_y.shape:', inner_circ_y.shape)
    
    #使用'paddle.concat'将两类数据的特征1和特征2分别延维度0拼接在一起，得到全部特征1和特征2
    #使用'paddle.stack'将两类特征延维度1堆叠在一起
    X = paddle.stack(
        [paddle.concat([outer_circ_x, inner_circ_x]),
        paddle.concat([outer_circ_y, inner_circ_y])],
        axis=1
    )

    print('after concat shape:', paddle.concat([outer_circ_x, inner_circ_x]).shape)
    print('X shape:', X.shape)

    #使用'paddle. zeros'将第一类数据的标签全部设置为0
    #使用'paddle. ones'将第一类数据的标签全部设置为1
    y = paddle.concat(
        [paddle.zeros(shape=[n_samples_out]), paddle.ones(shape=[n_samples_in])]
    )

    print('y shape:', y.shape)

    #如果shuffle为True，将所有数据打乱
    if shuffle:
        #使用'paddle.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor做索引值，用于打乱数据
        idx = paddle.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    #如果noise不为None，则给特征值加入噪声
    if noise is not None:
        #使用'paddle.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
        X += paddle.normal(mean=0.0, std=noise, shape=X.shape)

    return X, y


#加载数据集
def load_data(shuffle=True):
    """
    加载鸢尾花数据
    输入：
        - shuffle：是否打乱数据，数据类型为bool
    输出：
        - X：特征数据，shape=[150,4]
        - y：标签数据, shape=[150,3]
    """
    #加载原始数据
    X = np.array(load_iris().data, dtype=np.float32)
    y = np.array(load_iris().target, dtype=np.int64)

    X = paddle.to_tensor(X)
    y = paddle.to_tensor(y)

    #数据归一化
    X_min = paddle.min(X, axis=0)
    X_max = paddle.max(X, axis=0)
    X = (X-X_min) / (X_max-X_min)

    #如果shuffle为True，随机打乱数据
    if shuffle:
        idx = paddle.randperm(X.shape[0])
        X_new = copy.deepcopy(X)
        y_new = copy.deepcopy(y)
        for i in range(X.shape[0]):
            X_new[i] = X[idx[i]]
            y_new[i] = y[idx[i]]
        X = X_new
        y = y_new

    return X, y

