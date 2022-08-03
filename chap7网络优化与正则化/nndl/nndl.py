import math
import copy
import paddle
from paddle.metric import Metric

def make_moons(n_samples=1000, shuffle=True, noise=None):

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = paddle.cos(paddle.linspace(0, math.pi, n_samples_out))
    outer_circ_y = paddle.sin(paddle.linspace(0, math.pi, n_samples_out))
    inner_circ_x = 1 - paddle.cos(paddle.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 1 - paddle.sin(paddle.linspace(0, math.pi, n_samples_in)) - 0.5

    X = paddle.stack(
        [paddle.concat([outer_circ_x, inner_circ_x]), 
        paddle.concat([outer_circ_y, inner_circ_y])],
        axis=1
    )
    y = paddle.concat(
        [paddle.zeros(shape=[n_samples_out]), paddle.ones(shape=[n_samples_in])]
    )

    if shuffle:
        idx = paddle.randperm(X.shape[0])
        X_new = copy.deepcopy(X)
        y_new = copy.deepcopy(y)
        for i in range(X.shape[0]):
            X_new[i] = X[idx[i]]
            y_new[i] = y[idx[i]]
        X = X_new
        y = y_new

    if noise is not None:
        X += paddle.normal(mean=0.0, std=noise, shape=X.shape)
    
    return X, y

def accuracy(preds, labels): 
    """
    输入:
        - preds:预测值，二分类时，shape=[N, 1]，N为样本数量，多分类时，shape=[N, C]，C为类别数量
        - labels:真实标签，shape=[N, 1]
    输出:
        - 准确率:shape=[1]
    """ 
    #判断是二分类任务还是多分类任务，preds.shape[1]=1时为二分类任务，preds.shape[1]>1时为多分类任务
    if preds.shape[1] == 1: 
        #二分类时，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0 
        #使用'paddle.cast'将preds的数据类型转换为float32类型
        preds = paddle.cast((preds>=0.5), dtype='float32')
    else: 
        #多分类时，使用'paddle.argmax'计算最大元素索引作为类别 
        preds = paddle.argmax(preds, axis=1, dtype='int32')
    return paddle.mean(paddle.cast(paddle.equal(preds, labels), dtype='float32'))

class RunnerV3(object):
    def __init__(self, model, optimizer, loss_fn, metric, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric #只用于计算评价指标

        #记录训练过程中的评价指标变化情况
        self.dev_scores = []

        #记录训练过程中的损失函数变化情况
        self.train_epoch_losses = [] # 一个epoch记录一次loss
        self.train_step_losses = []  # 一个step记录一次loss
        self.dev_losses = []
        
        #记录全局最优指标
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, **kwargs):
        #将模型切换为训练模式
        self.model.train()

        #传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        #传入log打印频率，如果没有传入值则默认为100
        log_steps = kwargs.get("log_steps", 100)
        # 评价频率
        eval_steps = kwargs.get("eval_steps", 0)

        #传入模型保存路径，如果没有传入值则默认为"best_model.pdparams"
        save_path = kwargs.get("save_path", "best_model.pdparams")

        custom_print_log = kwargs.get("custom_print_log", None) 
       
        # 训练总的步数
        num_training_steps = num_epochs * len(train_loader)

        if eval_steps:
            if self.metric is None:
                raise RuntimeError('Error: Metric can not be None!')
            if dev_loader is None:
                raise RuntimeError('Error: dev_loader can not be None!')
            
        # 运行的step数目
        global_step = 0

        #进行num_epochs轮训练
        for epoch in range(num_epochs):
            #用于统计训练集的损失
            total_loss = 0
            for step, data in enumerate(train_loader):
                X, y = data
                #获取模型预测
                logits = self.model(X)
                loss = self.loss_fn(logits, y) # 默认求mean
                total_loss += loss 

                # 训练过程中，每个step的loss进行保存
                self.train_step_losses.append((global_step,loss.item()))

                if log_steps and global_step%log_steps==0:
                    print(f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")
                
                # 梯度反向传播，计算每个参数的梯度值
                loss.backward() 

                if custom_print_log:
                   custom_print_log(self)
                
                # 小批量梯度下降进行参数更新
                self.optimizer.step()
                # 梯度归零
                self.optimizer.clear_grad()

                # 判断是否需要评价
                if eval_steps>0 and global_step!=0 and \
                    (global_step%eval_steps == 0 or global_step==(num_training_steps-1)):

                    dev_score, dev_loss = self.evaluate(dev_loader, global_step=global_step)
                    print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}") 

                    #将模型切换为训练模式
                    self.model.train()

                    #如果当前指标为最优指标，保存该模型
                    if dev_score > self.best_score:
                        self.save_model(save_path)
                        print(f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        self.best_score = dev_score

                global_step += 1
            
            # 当前epoch 训练loss累计值 
            trn_loss = (total_loss/len(train_loader)).item()
            # epoch粒度的训练loss保存
            self.train_epoch_losses.append(trn_loss)
            
        print("[Train] Training done!")

    #模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def evaluate(self, dev_loader, **kwargs):
        assert self.metric is not None

        #将模型设置为评估模式
        self.model.eval()

        global_step = kwargs.get("global_step", -1) 

        #用于统计训练集的损失
        total_loss = 0

        # 重置评价
        self.metric.reset() 
        
        # 遍历验证集每个批次    
        for batch_id, data in enumerate(dev_loader):
            X, y = data
    
            #计算模型输出
            logits = self.model(X)
            
            #计算损失函数
            loss = self.loss_fn(logits, y).item()
            # 累积损失
            total_loss += loss 

            # 累积评价
            self.metric.update(logits, y)

        dev_loss = (total_loss/len(dev_loader))
        self.dev_losses.append((global_step, dev_loss))

        dev_score = self.metric.accumulate() 
        self.dev_scores.append(dev_score)
        
        return dev_score, dev_loss
    
    #模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def predict(self, x, **kwargs):
        #将模型设置为评估模式
        self.model.eval()
        #运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        model_state_dict = paddle.load(model_path)
        self.model.set_state_dict(model_state_dict)

class Accuracy(Metric):
    def __init__(self, is_logist=True):
        """
        输入：
           - is_logist: outputs是logits还是激活后的值
        """

        # 用于统计正确的样本个数
        self.num_correct = 0
        # 用于统计样本的总数
        self.num_count = 0
        self.is_logist = is_logist

    def update(self, outputs, labels):
        """
        输入：
           - outputs: 预测值, shape=[N,class_num]
           - labels: 标签值, shape=[N,1]
        """

        #判断是二分类任务还是多分类任务，shape[1]=1时为二分类任务，shape[1]>1时为多分类任务
        if outputs.shape[1] == 1:
            outputs = paddle.squeeze(outputs)
            if self.is_logist:
                # logits判断是否大于0
                preds = paddle.cast((outputs>=0), dtype='float32')
            else:
                # 如果不是logits，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
                preds = paddle.cast((outputs>=0.5), dtype='float32')
        else:
            #多分类时，使用'paddle.argmax'计算最大元素索引作为类别
            preds = paddle.argmax(outputs, axis=1, dtype='int64')

        # 获取本批数据中预测正确的样本个数
        labels = paddle.squeeze(labels, axis=-1)
        batch_correct = paddle.sum(paddle.cast(preds==labels, dtype="float32")).numpy()[0]
        batch_count = len(labels)

        # 更新num_correct 和 num_count
        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        # 使用累计的数据，计算总的指标
        if self.num_count == 0:
            return 0
        return self.num_correct / self.num_count

    def reset(self):
        self.num_correct = 0
        self.num_count = 0

    def name(self):
        return "Accuracy"