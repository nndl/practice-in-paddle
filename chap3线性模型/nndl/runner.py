import paddle

#新增RunnerV2类
class RunnerV2(object):
    def __init__(self, model, optimizer, metric, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        #记录训练过程中的评价指标变化情况
        self.train_scores = []
        self.dev_scores = []
        #记录训练过程中的损失函数变化情况
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        #传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        #传入log打印频率，如果没有传入值则默认为100
        log_epochs = kwargs.get("log_epochs", 100)
        #传入模型保存路径，如果没有传入值则默认为"best_model.pdparams"
        save_path = kwargs.get("save_path", "best_model.pdparams")
        #梯度打印函数，如果没有传入则默认为"None"
        print_grads = kwargs.get("print_grads", None)
        #记录全局最优指标
        best_score = 0
        #进行num_epochs轮训练
        for epoch in range(num_epochs):
            X, y = train_set
            #获取模型预测
            logits = self.model(X)
            #计算交叉熵损失
            trn_loss = self.loss_fn(logits, y).item()
            self.train_loss.append(trn_loss)
            #计算评价指标
            trn_score = self.metric(logits, y).item()
            self.train_scores.append(trn_score)
            #计算参数梯度
            self.model.backward(y)
            if print_grads is not None:
                #打印每一层的梯度
                print_grads(self.model)
            #更新模型参数
            self.optimizer.step()
            dev_score, dev_loss = self.evaluate(dev_set)
            #如果当前指标为最优指标，保存该模型
            if dev_score > best_score:
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
            if epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}, loss: {trn_loss}, score: {trn_score}")
                print(f"[Dev] epoch: {epoch}, loss: {dev_loss}, score: {dev_score}")
                
    def evaluate(self, data_set):
        X, y = data_set
        #计算模型输出
        logits = self.model(X)
        #计算损失函数
        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        #计算评价指标
        score = self.metric(logits, y).item()
        self.dev_scores.append(score)
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_path):
        paddle.save(self.model.params, save_path)

    def load_model(self, model_path):
        self.model.params = paddle.load(model_path)