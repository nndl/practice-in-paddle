"""chap3 线性模型: logistic regression (Linear+sigmoid) + softmax regression on iris."""
import paddle
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter


def main():
    r = TestReporter("chap3 线性模型")
    pkg = push_chapter_pkg("chap3线性模型")
    try:
        from nndl.op import Linear, model_SR, MultiCrossEntropyLoss
        from nndl.metric import accuracy
        from nndl.activation import softmax
        from nndl.dataset import make_moons
        import paddle.nn.functional as F
        import numpy as np
        from sklearn.datasets import load_iris

        def _iris():
            X = paddle.to_tensor(np.array(load_iris().data, dtype=np.float32))
            y = paddle.to_tensor(np.array(load_iris().target, dtype=np.int64))
            return X, y

        def t_linear_forward():
            paddle.seed(0)
            m = Linear(dimension=4)
            X = paddle.randn([8, 4])
            assert m(X).shape == [8, 1]

        def t_softmax_op():
            x = paddle.to_tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
            s = softmax(x)
            assert paddle.allclose(s.sum(axis=1), paddle.ones([2]), atol=1e-5).item()

        def t_model_sr_forward_backward():
            paddle.seed(0)
            m = model_SR(input_dim=4, output_dim=3)
            X = paddle.randn([10, 4])
            out = m(X)
            assert out.shape == [10, 3]
            assert paddle.allclose(out.sum(axis=1), paddle.ones([10]), atol=1e-5).item()
            # NOTE: docstring claims [N,1] but math only works with [N]
            labels = paddle.to_tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype='int64')
            m.backward(labels)
            assert 'W' in m.grads and 'b' in m.grads
            assert m.grads['W'].shape == [4, 3], f"got {m.grads['W'].shape}"

        def t_multi_ce_loss():
            loss_fn = MultiCrossEntropyLoss()
            # probabilities, well-distributed
            preds = paddle.to_tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
            labels = paddle.to_tensor([[0], [1], [2]])
            loss = loss_fn(preds, labels).item()
            # average over 3 samples of -log(p_correct)
            import math
            expected = -(math.log(0.7) + math.log(0.6) + math.log(0.5)) / 3
            assert abs(loss - expected) < 1e-4

        def t_accuracy_func():
            preds = paddle.to_tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.2], [3.0, 0.0, 0.1]])
            labels = paddle.to_tensor([0, 1, 0], dtype='int64')
            score = accuracy(preds, labels).item()
            assert score == 1.0

        def t_iris_fit_softmax_one_step():
            X, y = _iris()
            # Train one manual step of softmax regression
            paddle.seed(1)
            m = model_SR(input_dim=4, output_dim=3)
            out = m(X)
            m.backward(y)  # labels shape [N]
            # SGD step on internal params
            lr = 0.1
            m.params['W'] = m.params['W'] - lr * m.grads['W']
            m.params['b'] = m.params['b'] - lr * m.grads['b']
            out2 = m(X)
            # After one step, predictions should change; pick any index to compare
            assert not paddle.allclose(out, out2).item()

        r.run("Linear forward", t_linear_forward)
        r.run("softmax activation sums to 1", t_softmax_op)
        r.run("model_SR forward + backward", t_model_sr_forward_backward)
        r.run("MultiCrossEntropyLoss numerical check", t_multi_ce_loss)
        r.run("accuracy() multiclass", t_accuracy_func)
        r.run("Softmax regression: 1 manual SGD step on iris", t_iris_fit_softmax_one_step)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
