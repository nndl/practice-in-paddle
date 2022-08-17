import paddle

def optimizer_lsm(model, X, y, reg_lambda=0):
  """
    输入：
       - model: 模型
       - X: tensor, 特征数据，shape=[N,D]
       - y: tensor,标签数据，shape=[N]
       - reg_lambda: float, 正则化系数，默认为0
    输出：
       - model: 优化好的模型
    """

  N, D = X.shape

  # 对输入特征数据所有特征向量求平均
  x_bar_tran = paddle.mean(X,axis=0).T 
  
  # 求标签的均值,shape=[1]
  y_bar = paddle.mean(y)
  
  # paddle.subtract通过广播的方式实现矩阵减向量
  x_sub = paddle.subtract(X,x_bar_tran)

  # 使用paddle.all判断输入tensor是否全0
  if paddle.all(x_sub==0):
    model.params['b'] = y_bar
    model.params['w'] = paddle.zeros(shape=[D])
    return model
  
  # paddle.inverse求方阵的逆
  tmp = paddle.inverse(paddle.matmul(x_sub.T,x_sub)+
          reg_lambda*paddle.eye(num_rows = (D)))

  w = paddle.matmul(paddle.matmul(tmp,x_sub.T),(y-y_bar))
  
  b = y_bar-paddle.matmul(x_bar_tran,w)
  
  model.params['b'] = b
  model.params['w'] = paddle.squeeze(w,axis=-1)

  return model

from abc import abstractmethod

#新增优化器基类
class Optimizer(object):
    def __init__(self, init_lr, model):
        """
        优化器类初始化
        """
        #初始化学习率，用于参数更新的计算
        self.init_lr = init_lr
        #指定优化器需要优化的模型
        self.model = model

    @abstractmethod
    def step(self):
        """
        定义每次迭代如何更新参数
        """
        pass

#新增梯度下降法优化器
class SimpleBatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(SimpleBatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        #参数更新
        #遍历所有参数，按照公式(3.8)和(3.9)更新参数
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] = self.model.params[key] - self.init_lr * self.model.grads[key]