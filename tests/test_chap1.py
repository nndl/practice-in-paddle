"""chap1 实践基础: tensor operations + custom Op forward/backward chain.

Virtualizes the chapter's main demos: tensor broadcasting/reshape,
and the add/multiply/exponential Op classes composing into a tiny graph.
"""
import math
import paddle
from _common import TestReporter


def main():
    r = TestReporter("chap1 实践基础")

    def t_tensor_create():
        x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float32")
        assert x.shape == [2, 2]
        assert abs(x[0, 0].item() - 1.1) < 1e-5

    def t_broadcast_arith():
        a = paddle.ones([2, 3, 1, 5])
        b = paddle.ones([3, 2, 5])
        # broadcast to [2,3,2,5]
        out = a + b
        assert out.shape == [2, 3, 2, 5]

    def t_matmul_high_dim():
        # [10,1,5,2] x [3,2,5] -> [10,3,5,5]
        x = paddle.ones([10, 1, 5, 2])
        y = paddle.ones([3, 2, 5])
        out = paddle.matmul(x, y)
        assert out.shape == [10, 3, 5, 5]

    def t_reshape_squeeze_concat():
        x = paddle.arange(0, 12, dtype='float32').reshape([3, 4])
        assert x.shape == [3, 4]
        y = paddle.unsqueeze(x, axis=0)
        assert y.shape == [1, 3, 4]
        z = paddle.squeeze(y, axis=0)
        assert z.shape == [3, 4]
        cat = paddle.concat([x, z], axis=0)
        assert cat.shape == [6, 4]

    # Custom Op classes from chap1 §1.5
    class Op:
        def __init__(self): pass
        def __call__(self, *args): return self.forward(*args)
        def forward(self, *args): raise NotImplementedError
        def backward(self, *args): raise NotImplementedError

    class add(Op):
        def forward(self, x, y):
            self.x, self.y = x, y
            return x + y
        def backward(self, grads):
            return grads * 1, grads * 1

    class multiply(Op):
        def forward(self, x, y):
            self.x, self.y = x, y
            return x * y
        def backward(self, grads):
            return grads * self.y, grads * self.x

    class exponential(Op):
        def forward(self, x):
            self.x = x
            return math.exp(x)
        def backward(self, grads):
            return grads * math.exp(self.x)

    def t_op_chain_forward():
        a, b, c, d = 2.0, 3.0, 4.0, 5.0
        mul1, mul2, addz, expe = multiply(), multiply(), add(), exponential()
        e = mul1(a, b)
        f = mul2(c, d)
        z = addz(e, f)
        result = expe(z)
        assert math.isclose(result, math.exp(2*3 + 4*5), rel_tol=1e-9)

    def t_op_chain_backward():
        a, b, c, d = 2.0, 3.0, 4.0, 5.0
        mul1, mul2, addz, expe = multiply(), multiply(), add(), exponential()
        e = mul1(a, b)
        f = mul2(c, d)
        z = addz(e, f)
        _ = expe(z)
        # walk gradient back: d/dz exp(z) = exp(z); add splits 1,1
        # multiply gives partials e.g., d/da (a*b) = b
        g_z = expe.backward(1.0)
        g_e, g_f = addz.backward(g_z)
        g_a, g_b = mul1.backward(g_e)
        g_c, g_d = mul2.backward(g_f)
        # check da: d/da exp(a*b + c*d) = b * exp(a*b + c*d)
        expected = b * math.exp(a*b + c*d)
        assert math.isclose(g_a, expected, rel_tol=1e-9)

    def t_nn_layer():
        class Linear(paddle.nn.Layer):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.fc = paddle.nn.Linear(in_f, out_f)
            def forward(self, x):
                return self.fc(x)
        m = Linear(3, 2)
        x = paddle.randn([4, 3])
        y = m(x)
        assert y.shape == [4, 2]
        # autograd works
        loss = y.sum()
        loss.backward()
        assert m.fc.weight.grad is not None

    r.run("tensor creation + indexing", t_tensor_create)
    r.run("broadcasting arithmetic", t_broadcast_arith)
    r.run("high-dim matmul", t_matmul_high_dim)
    r.run("reshape/squeeze/concat", t_reshape_squeeze_concat)
    r.run("Op chain forward (add/mul/exp)", t_op_chain_forward)
    r.run("Op chain backward", t_op_chain_backward)
    r.run("nn.Layer + autograd", t_nn_layer)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
