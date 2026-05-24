"""chap7 网络优化与正则化: optimizer comparison + BatchNorm + Dropout + L2 weight decay."""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter


def main():
    r = TestReporter("chap7 网络优化与正则化")
    pkg = push_chapter_pkg("chap7网络优化与正则化")
    try:
        import nndl.nndl as nlib

        def t_optimizer_compare():
            """Train same tiny model with SGD vs Adam, assert both reduce loss."""
            paddle.seed(0)
            def make():
                m = nn.Linear(4, 2)
                return m
            X = paddle.randn([32, 4])
            y = paddle.randint(0, 2, shape=[32]).cast('int64')
            results = {}
            for name, opt_cls in [('SGD', paddle.optimizer.SGD),
                                   ('Adam', paddle.optimizer.Adam),
                                   ('AdamW', paddle.optimizer.AdamW)]:
                m = make()
                opt = opt_cls(learning_rate=0.01, parameters=m.parameters())
                initial, final = None, None
                for _ in range(30):
                    loss = F.cross_entropy(m(X), y)
                    if initial is None:
                        initial = loss.item()
                    loss.backward()
                    opt.step()
                    opt.clear_grad()
                    final = loss.item()
                results[name] = (initial, final)
            for name, (initial, final) in results.items():
                assert final < initial, f"{name} didn't reduce loss: {initial} -> {final}"

        def t_batchnorm():
            bn = nn.BatchNorm2D(num_features=4)
            x = paddle.randn([8, 4, 6, 6])
            y = bn(x)
            assert y.shape == x.shape

        def t_layernorm():
            ln = nn.LayerNorm(normalized_shape=8)
            x = paddle.randn([4, 8])
            y = ln(x)
            assert y.shape == x.shape

        def t_dropout_train_vs_eval():
            paddle.seed(0)
            drop = nn.Dropout(p=0.9)
            x = paddle.ones([100])
            drop.train()
            y_train = drop(x)
            drop.eval()
            y_eval = drop(x)
            # In eval mode, dropout is identity
            assert paddle.allclose(y_eval, x).item()
            # In train mode, ~10% of values survive (with rescaling)
            num_zero = int((y_train == 0).cast('int64').sum().item())
            assert num_zero > 50

        def t_l2_weight_decay():
            """Verify L2Decay regularizer is accepted and training stays stable."""
            paddle.seed(0)
            m = nn.Linear(4, 2)
            opt = paddle.optimizer.SGD(
                learning_rate=0.05,
                parameters=m.parameters(),
                weight_decay=paddle.regularizer.L2Decay(coeff=0.01))
            X = paddle.randn([16, 4])
            y = paddle.randint(0, 2, shape=[16]).cast('int64')
            for _ in range(20):
                loss = F.cross_entropy(m(X), y)
                loss.backward()
                opt.step()
                opt.clear_grad()
            assert paddle.isfinite(m.weight).all().item()

        def t_nndl_accuracy_and_runner():
            # exercise the accuracy + Accuracy + RunnerV3 from nndl.nndl
            preds = paddle.to_tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.2], [3.0, 0.0, 0.1]])
            labels = paddle.to_tensor([0, 1, 0], dtype='int64')
            assert nlib.accuracy(preds, labels).item() == 1.0
            assert hasattr(nlib, 'RunnerV3')

        def t_lr_scheduler():
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=5, gamma=0.5)
            assert abs(scheduler.get_lr() - 0.1) < 1e-9
            for _ in range(5):
                scheduler.step()
            assert abs(scheduler.get_lr() - 0.05) < 1e-9

        r.run("optimizer comparison (SGD/Adam/AdamW)", t_optimizer_compare)
        r.run("BatchNorm2D forward", t_batchnorm)
        r.run("LayerNorm forward", t_layernorm)
        r.run("Dropout train vs eval", t_dropout_train_vs_eval)
        r.run("L2Decay weight decay regularizer", t_l2_weight_decay)
        r.run("nndl.nndl accuracy/RunnerV3 surface", t_nndl_accuracy_and_runner)
        r.run("learning rate scheduler StepDecay", t_lr_scheduler)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
