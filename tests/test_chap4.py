"""chap4 前馈神经网络: MLP + Accuracy + RunnerV3 on synthetic moons / iris-like data."""
import paddle
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter


def main():
    r = TestReporter("chap4 前馈神经网络")
    pkg = push_chapter_pkg("chap4前馈神经网络")
    try:
        from nndl.dataset import make_moons, load_data
        from nndl.metric import accuracy, Accuracy
        from nndl.runner import RunnerV3

        def t_make_moons():
            X, y = make_moons(n_samples=40, shuffle=False, noise=None)
            assert X.shape == [40, 2]
            assert y.shape == [40]
            # half class-0, half class-1
            assert int(y.sum().item()) == 20

        def t_iris_load():
            X, y = load_data(shuffle=False)
            assert X.shape == [150, 4]
            assert y.shape == [150]

        def t_mlp_forward_backward():
            # Two-layer MLP for binary classification on make_moons.
            paddle.seed(0)
            class MLP(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.fc1 = paddle.nn.Linear(2, 8)
                    self.fc2 = paddle.nn.Linear(8, 2)
                def forward(self, x):
                    return self.fc2(paddle.nn.functional.relu(self.fc1(x)))
            model = MLP()
            X, y = make_moons(n_samples=64, shuffle=False, noise=0.1)
            y = y.cast('int64')
            opt = paddle.optimizer.SGD(learning_rate=0.05, parameters=model.parameters())
            loss_fn = paddle.nn.CrossEntropyLoss()
            losses = []
            for _ in range(20):
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                opt.clear_grad()
                losses.append(loss.item())
            # loss should decrease meaningfully
            assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"

        def t_runner_v3_full_loop():
            paddle.seed(0)
            class MLP(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.fc1 = paddle.nn.Linear(4, 6)
                    self.fc2 = paddle.nn.Linear(6, 3)
                def forward(self, x):
                    return self.fc2(paddle.nn.functional.relu(self.fc1(x)))
            X, y = load_data(shuffle=True)
            train_X, train_y = X[:120], y[:120].reshape([-1, 1]).cast('int64')
            dev_X, dev_y = X[120:], y[120:].reshape([-1, 1]).cast('int64')
            class IrisDS(paddle.io.Dataset):
                def __init__(self, X, y):
                    self.X, self.y = X, y
                def __len__(self): return self.X.shape[0]
                def __getitem__(self, i): return self.X[i], self.y[i]
            train_loader = paddle.io.DataLoader(IrisDS(train_X, train_y), batch_size=32, shuffle=True)
            dev_loader = paddle.io.DataLoader(IrisDS(dev_X, dev_y), batch_size=32)
            model = MLP()
            opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
            runner = RunnerV3(model, opt, paddle.nn.CrossEntropyLoss(), Accuracy(is_logist=True))
            runner.train(train_loader, dev_loader, num_epochs=3, log_steps=100, eval_steps=4,
                         save_path="_tmp_chap4.pdparams")
            assert runner.best_score > 0.3, f"best dev score too low: {runner.best_score}"

        r.run("make_moons synthetic data", t_make_moons)
        r.run("iris load_data", t_iris_load)
        r.run("MLP train loop: loss decreases", t_mlp_forward_backward)
        r.run("RunnerV3 full train+eval on iris", t_runner_v3_full_loop)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
