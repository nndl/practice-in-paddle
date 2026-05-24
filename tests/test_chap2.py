"""chap2 机器学习概述: linear regression + closed-form optimizer + polynomial fitting."""
import paddle
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter


def main():
    r = TestReporter("chap2 机器学习概述")
    pkg = push_chapter_pkg("chap2机器学习概述")
    try:
        from nndl.op import Linear
        from nndl.opitimizer import optimizer_lsm

        def t_linear_forward():
            paddle.seed(0)
            m = Linear(input_size=3)
            X = paddle.randn([5, 3])
            y = m(X)
            assert y.shape == [5, 1]

        def t_least_squares_recovers_weights():
            # Synthesize y = X @ w_true + b_true, then verify closed-form fit.
            paddle.seed(42)
            N, D = 200, 2
            w_true = paddle.to_tensor([2.0, -1.5]).reshape([D, 1])
            b_true = paddle.to_tensor([0.5])
            X = paddle.randn([N, D])
            y = (paddle.matmul(X, w_true) + b_true).reshape([N])
            m = Linear(input_size=D)
            m = optimizer_lsm(m, X, y, reg_lambda=0)
            w_fit = m.params['w'].reshape([D])
            b_fit = m.params['b'].reshape([1])
            assert paddle.allclose(w_fit, w_true.reshape([D]), atol=1e-3).item()
            assert paddle.allclose(b_fit, b_true, atol=1e-3).item()

        def t_least_squares_with_ridge():
            paddle.seed(7)
            N, D = 50, 3
            X = paddle.randn([N, D])
            y = paddle.randn([N])
            m = Linear(input_size=D)
            m = optimizer_lsm(m, X, y, reg_lambda=0.5)
            # just verify it runs and shape is right
            assert m.params['w'].numel() == D

        def t_polynomial_design_matrix():
            # In §2.x, polynomial regression builds a Vandermonde-style feature matrix.
            x = paddle.linspace(-1, 1, 20).reshape([20, 1])
            degree = 4
            feats = paddle.concat([x ** k for k in range(1, degree + 1)], axis=1)
            assert feats.shape == [20, degree]

        def t_mse():
            # MSE used in chapter
            y_true = paddle.to_tensor([1.0, 2.0, 3.0])
            y_pred = paddle.to_tensor([1.1, 2.1, 2.9])
            mse = paddle.mean((y_true - y_pred) ** 2)
            assert abs(mse.item() - 0.01) < 1e-6

        r.run("Linear forward", t_linear_forward)
        r.run("closed-form least squares recovers weights", t_least_squares_recovers_weights)
        r.run("least squares with L2 ridge", t_least_squares_with_ridge)
        r.run("polynomial feature matrix", t_polynomial_design_matrix)
        r.run("mean squared error", t_mse)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
