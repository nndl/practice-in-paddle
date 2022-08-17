import paddle
paddle.seed(10) #设置随机种子

class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError

# 线性算子
class Linear(Op):
    def __init__(self,input_size):
        """
        输入：
           - input_size:模型要处理的数据特征向量长度
        """

        self.input_size = input_size

        # 模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.input_size,1],dtype='float32') 
        self.params['b'] = paddle.zeros(shape=[1],dtype='float32')

    def __call__(self, X):
        return self.forward(X)

    # 前向函数
    def forward(self, X):
        """
        输入：
           - X: tensor, shape=[N,D]
           注意这里的X矩阵是由N个x向量的转置拼接成的，与原教材行向量表示方式不一致
        输出：
           - y_pred： tensor, shape=[N]
        """

        N,D = X.shape

        if self.input_size==0:
            return paddle.full(shape=[N,1], fill_value=self.params['b'])
        
        assert D==self.input_size # 输入数据维度合法性验证

        # 使用paddle.matmul计算两个tensor的乘积
        y_pred = paddle.matmul(X,self.params['w'])+self.params['b']
        
        return y_pred