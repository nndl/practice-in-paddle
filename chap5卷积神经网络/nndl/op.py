import paddle
from nndl.activation import softmax
paddle.seed(10) #设置随机种子
from paddle import nn
import paddle.nn.functional as F

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
    def __init__(self,dimension):
        """
        输入：
           - dimension:模型要处理的数据特征向量长度
        """

        self.dim = dimension

        # 模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.dim,1],dtype='float32') 
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

        if self.dim==0:
            return paddle.full(shape=[N,1], fill_value=self.params['b'])
        
        assert D==self.dim # 输入数据维度合法性验证

        # 使用paddle.matmul计算两个tensor的乘积
        y_pred = paddle.matmul(X,self.params['w'])+self.params['b']
        
        return y_pred

#新增Softmax算子
class model_SR(Op):
    def __init__(self, input_dim, output_dim):
        super(model_SR, self).__init__()
        self.params = {}
        #将线性层的权重参数全部初始化为0
        self.params['W'] = paddle.zeros(shape=[input_dim, output_dim])
        #self.params['W'] = paddle.normal(mean=0, std=0.01, shape=[input_dim, output_dim])
        #将线性层的偏置参数初始化为0
        self.params['b'] = paddle.zeros(shape=[output_dim])
        #存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.X = inputs
        #线性计算
        score = paddle.matmul(self.X, self.params['W']) + self.params['b']
        #Softmax 函数
        self.outputs = softmax(score)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]，其中N为样本数量
        """
        #计算偏导数
        N =labels.shape[0]
        labels = paddle.nn.functional.one_hot(labels, self.output_dim)
        self.grads['W'] = -1 / N * paddle.matmul(self.X.t(), (labels-self.outputs))
        self.grads['b'] = -1 / N * paddle.matmul(paddle.ones(shape=[N]), (labels-self.outputs))

#新增多类别交叉熵损失
class MultiCrossEntropyLoss(Op):
    def __init__(self):
        self.predicts = None
        self.labels = None
        self.num = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, 1]，N为样本数量
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = 0
        for i in range(0, self.num):
            index = self.labels[i]
            loss -= paddle.log(self.predicts[i][index])
        return loss / self.num

# 卷积算子
class Conv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)),
                    bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0))):
        super(Conv2D, self).__init__()
        # 创建卷积核
        self.weight = paddle.create_parameter(shape=[out_channels, in_channels, kernel_size,kernel_size],
                                                dtype='float32',
                                                attr=weight_attr)
        # 创建偏置
        self.bias = paddle.create_parameter(shape=[out_channels, 1],
                                                dtype='float32',
                                                attr=bias_attr)
        self.stride = stride
        self.padding = padding
        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.out_channels = out_channels

    # 基础卷积运算
    def single_forward(self, X, weight):
        # 零填充
        new_X = paddle.zeros([X.shape[0], X.shape[1]+2*self.padding, X.shape[2]+2*self.padding])
        new_X[:, self.padding:X.shape[1]+self.padding, self.padding:X.shape[2]+self.padding] = X
        u, v = weight.shape
        output_w = (new_X.shape[1] - u) // self.stride + 1
        output_h = (new_X.shape[2] - v) // self.stride + 1
        output = paddle.zeros([X.shape[0], output_w, output_h])
        for i in range(0, output.shape[1]):
            for j in range(0, output.shape[2]):
                output[:, i, j] = paddle.sum(
                    new_X[:, self.stride*i:self.stride*i+u, self.stride*j:self.stride*j+v]*weight, 
                    axis=[1,2])
        return output

    def forward(self, inputs):
        """
        输入：
            - inputs：输入矩阵，shape=[B, D, M, N]
            - weights：P组二维卷积核，shape=[P, D, U, V]
            - bias：P个偏置，shape=[P, 1]
        """
        feature_maps = []
        # 进行多次多输入通道卷积运算
        p=0
        for w, b in zip(self.weight, self.bias): # P个(w,b),每次计算一个特征图Zp
            multi_outs = []
            # 循环计算每个输入特征图对应的卷积结果
            for i in range(self.in_channels):
                single = self.single_forward(inputs[:,i,:,:], w[i])
                multi_outs.append(single)
                # print("Conv2D in_channels:",self.in_channels,"i:",i,"single:",single.shape)
            # 将所有卷积结果相加
            feature_map = paddle.sum(paddle.stack(multi_outs), axis=0) + b #Zp
            feature_maps.append(feature_map)
            # print("Conv2D out_channels:",self.out_channels, "p:",p,"feature_map:",feature_map.shape)
            p+=1
        # 将所有Zp进行堆叠
        out = paddle.stack(feature_maps, 1) 
        return out

# 汇聚层算子
class Pool2D(nn.Layer):
    def __init__(self, size=(2,2), mode='max', stride=1):
        super(Pool2D, self).__init__()
        # 汇聚方式
        self.mode = mode
        self.h, self.w = size
        self.stride = stride

    def forward(self, x):
        output_w = (x.shape[2] - self.w) // self.stride + 1
        output_h = (x.shape[3] - self.h) // self.stride + 1
        output = paddle.zeros([x.shape[0], x.shape[1], output_w, output_h])
        # 汇聚
        for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # 最大汇聚
                if self.mode == 'max':
                    output[:, :, i, j] = paddle.max(
                        x[:, :, self.stride*i:self.stride*i+self.w, self.stride*j:self.stride*j+self.h], 
                        axis=[2,3])
                # 平均汇聚
                elif self.mode == 'avg':
                    output[:, :, i, j] = paddle.mean(
                        x[:, :, self.stride*i:self.stride*i+self.w, self.stride*j:self.stride*j+self.h], 
                        axis=[2,3])
        
        return output

# 基于自定义算子的LeNet-5
class Model_LeNet(nn.Layer):
    def __init__(self, in_channels, num_classes=10):
        super(Model_LeNet, self).__init__()
        # 卷积层：输出通道数为6，卷积核大小为5×5
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5, weight_attr=paddle.ParamAttr())
        # 汇聚层：汇聚窗口为2×2，步长为2
        self.pool2 = Pool2D(size=(2,2), mode='max', stride=2)
        # 卷积层：输入通道数为6，输出通道数为16，卷积核大小为5×5，步长为1
        self.conv3 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, weight_attr=paddle.ParamAttr())
        # 汇聚层：汇聚窗口为2×2，步长为2
        self.pool4 = Pool2D(size=(2,2), mode='avg', stride=2)
        # 卷积层：输入通道数为16，输出通道数为120，卷积核大小为5×5
        self.conv5 = Conv2D(in_channels=16, out_channels=120, kernel_size=5, stride=1, weight_attr=paddle.ParamAttr())
        # 全连接层：输入神经元为120，输出神经元为84
        self.linear6 = nn.Linear(120, 84)
        # 全连接层：输入神经元为84，输出神经元为类别数
        self.linear7 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1：卷积层+激活函数
        output = F.relu(self.conv1(x))
        # S2：汇聚层
        output = self.pool2(output)
        # C3：卷积层+激活函数
        output = F.relu(self.conv3(output))
        # S4：汇聚层
        output = self.pool4(output)
        # C5：卷积层+激活函数
        output = F.relu(self.conv5(output))
        # 输入层将数据拉平[B,C,H,W] -> [B,CxHxW]
        output = paddle.squeeze(output, axis=[2,3])
        # F6：全连接层
        output = F.relu(self.linear6(output))
        # F7：全连接层
        output = self.linear7(output)
        return output

# 基于PaddlePaddle API的LeNet-5
class Paddle_LeNet(nn.Layer):
    def __init__(self, in_channels, num_classes=10):
        super(Paddle_LeNet, self).__init__()
        # 卷积层：输出通道数为6，卷积核大小为5*5
        self.conv1 = nn.Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5)
        # 汇聚层：汇聚窗口为2*2，步长为2
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 卷积层：输入通道数为6，输出通道数为16，卷积核大小为5*5
        self.conv3 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        # 汇聚层：汇聚窗口为2*2，步长为2
        self.pool4 = nn.AvgPool2D(kernel_size=2, stride=2)
        # 卷积层：输入通道数为16，输出通道数为120，卷积核大小为5*5
        self.conv5 = nn.Conv2D(in_channels=16, out_channels=120, kernel_size=5)
        # 全连接层：输入神经元为120，输出神经元为84
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        # 全连接层：输入神经元为84，输出神经元为类别数
        self.linear7 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # C1：卷积层+激活函数
        output = F.relu(self.conv1(x))
        # S2：汇聚层
        output = self.pool2(output)
        # C3：卷积层+激活函数
        output = F.relu(self.conv3(output))
        # S4：汇聚层
        output = self.pool4(output)
        # C5：卷积层+激活函数
        output = F.relu(self.conv5(output))
        # 输入层将数据拉平[B,C,H,W] -> [B,CxHxW]
        output = paddle.squeeze(output, axis=[2,3])
        # F6：全连接层
        output = F.relu(self.linear6(output))
        # F7：全连接层
        output = self.linear7(output)
        return output

# 残差单元
class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        """
        残差单元
        输入：
            - in_channels：输入通道数
            - out_channels：输出通道数
            - stride：残差单元的步长，通过调整残差单元中第一个卷积层的步长来控制
            - use_residual：用于控制是否使用残差连接
        """
        super(ResBlock, self).__init__()
        self.stride = stride
        self.use_residual = use_residual
        # 第一个卷积层，卷积核大小为3×3，可以设置不同输出通道数以及步长
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, padding=1, stride=self.stride, bias_attr=False)
        # 第二个卷积层，卷积核大小为3×3，不改变输入特征图的形状，步长为1
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1, bias_attr=False)
        
        # 如果conv2的输出和此残差块的输入数据形状不一致，则use_1x1conv = True
        # 当use_1x1conv = True，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if in_channels != out_channels or stride != 1:
            self.use_1x1conv = True
        else:
            self.use_1x1conv = False
        # 当残差单元包裹的非线性层输入和输出通道数不一致时，需要用1×1卷积调整通道数后再进行相加运算
        if self.use_1x1conv:
            self.shortcut = nn.Conv2D(in_channels, out_channels, 1, stride=self.stride, bias_attr=False)

        # 每个卷积层后会接一个批量规范化层，批量规范化的内容在7.5.1中会进行详细介绍
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)
        if self.use_1x1conv:
            self.bn3 = nn.BatchNorm2D(out_channels)

    def forward(self, inputs):
        y = F.relu(self.bn1(self.conv1(inputs)))
        y = self.bn2(self.conv2(y))
        if self.use_residual:
            if self.use_1x1conv:  # 如果为真，对inputs进行1×1卷积，将形状调整成跟conv2的输出y一致
                shortcut = self.shortcut(inputs)
                shortcut = self.bn3(shortcut)
            else: # 否则直接将inputs和conv2的输出y相加
                shortcut = inputs
            y = paddle.add(shortcut, y)
        out = F.relu(y)
        return out

# ResNet-18
def make_first_module(in_channels):
    # 模块一：7*7卷积、批量规范化、汇聚
    m1 = nn.Sequential(nn.Conv2D(in_channels, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2D(64), nn.ReLU(),
                    nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
    return m1

def resnet_module(input_channels, out_channels, num_res_blocks, stride=1, use_residual=True):
    blk = []
    # 根据num_res_blocks，循环生成残差单元
    for i in range(num_res_blocks):
        if i == 0: # 创建模块中的第一个残差单元
            blk.append(ResBlock(input_channels, out_channels,
                                stride=stride, use_residual=use_residual))
        else:      # 创建模块中的其他残差单元
            blk.append(ResBlock(out_channels, out_channels, use_residual=use_residual))
    return blk

def make_modules(use_residual):
    # 模块二：包含两个残差单元，输入通道数为64，输出通道数为64，步长为1，特征图大小保持不变
    m2 = nn.Sequential(*resnet_module(64, 64, 2, stride=1, use_residual=use_residual))
    # 模块三：包含两个残差单元，输入通道数为64，输出通道数为128，步长为2，特征图大小缩小一半。
    m3 = nn.Sequential(*resnet_module(64, 128, 2, stride=2, use_residual=use_residual))
    # 模块四：包含两个残差单元，输入通道数为128，输出通道数为256，步长为2，特征图大小缩小一半。
    m4 = nn.Sequential(*resnet_module(128, 256, 2, stride=2, use_residual=use_residual))
    # 模块五：包含两个残差单元，输入通道数为256，输出通道数为512，步长为2，特征图大小缩小一半。
    m5 = nn.Sequential(*resnet_module(256, 512, 2, stride=2, use_residual=use_residual))
    return m2, m3, m4, m5

# 定义完整网络
class Model_ResNet18(nn.Layer):
    def __init__(self, in_channels=3, num_classes=10, use_residual=True):
        super(Model_ResNet18,self).__init__()
        m1 = make_first_module(in_channels)
        m2, m3, m4, m5 = make_modules(use_residual)
        # 封装模块一到模块6
        self.net = nn.Sequential(m1, m2, m3, m4, m5,
                        # 模块六：汇聚层、全连接层
                        nn.AdaptiveAvgPool2D(1), nn.Flatten(), nn.Linear(512, num_classes) )

    def forward(self, x):
        return self.net(x)

