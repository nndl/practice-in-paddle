import paddle

#x为tensor
def softmax(X):
    """
    输入：
        - X：shape=[N, C]，N为向量数量，C为向量维度
    """
    x_max = paddle.max(X, axis=1, keepdim=True)#N,1
    x_exp = paddle.exp(X - x_max)
    partition = paddle.sum(x_exp, axis=1, keepdim=True)#N,1
    return x_exp / partition