"""chap8 注意力机制: scaled dot-product attention + multi-head + simple seq model."""
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter, load_file_module
from pathlib import Path


def main():
    r = TestReporter("chap8 注意力机制")
    pkg = push_chapter_pkg("chap8注意力机制")
    try:
        nndl = load_file_module("chap8_nndl", Path(pkg) / "nndl.py")

        def t_scaled_dot_product_attention():
            """The chapter's core formula: softmax(QK^T/sqrt(d)) V."""
            paddle.seed(0)
            B, T, D = 2, 4, 8
            Q = paddle.randn([B, T, D])
            K = paddle.randn([B, T, D])
            V = paddle.randn([B, T, D])
            scores = paddle.matmul(Q, K.transpose([0, 2, 1])) / math.sqrt(D)
            weights = F.softmax(scores, axis=-1)
            out = paddle.matmul(weights, V)
            assert out.shape == [B, T, D]
            # attention weights must sum to 1 along the key axis
            assert paddle.allclose(weights.sum(axis=-1), paddle.ones([B, T]), atol=1e-5).item()

        def t_masked_attention():
            """Causal masking via paddle.where, used in chap8."""
            paddle.seed(0)
            T, D = 4, 6
            Q = paddle.randn([1, T, D])
            K = paddle.randn([1, T, D])
            scores = paddle.matmul(Q, K.transpose([0, 2, 1]))
            # Lower-triangular mask: True where keep, False where mask
            mask = paddle.tril(paddle.ones([T, T])).cast('bool').unsqueeze(0)
            neg_inf = paddle.full([1, T, T], -1e9)
            scores = paddle.where(mask, scores, neg_inf)
            weights = F.softmax(scores, axis=-1)
            # Upper triangle of weights (excluding diag) must be ~0
            upper = paddle.triu(weights[0], diagonal=1)
            assert upper.abs().max().item() < 1e-5

        def t_multihead_attention_layer():
            mha = nn.MultiHeadAttention(embed_dim=16, num_heads=4)
            x = paddle.randn([2, 5, 16])
            y = mha(x, x, x)
            assert y.shape == [2, 5, 16]

        def t_embedding():
            emb = nn.Embedding(num_embeddings=10, embedding_dim=8)
            idx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            out = emb(idx)
            assert out.shape == [2, 3, 8]

        def t_attention_classifier_train():
            """Tiny attention-based seq classifier: embed -> self-attn -> mean -> linear."""
            paddle.seed(0)
            class AttnClf(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(num_embeddings=20, embedding_dim=16)
                    self.attn = nn.MultiHeadAttention(embed_dim=16, num_heads=2)
                    self.fc = nn.Linear(16, 3)
                def forward(self, x):
                    e = self.embed(x)
                    a = self.attn(e, e, e)
                    return self.fc(a.mean(axis=1))
            m = AttnClf()
            opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=m.parameters())
            X = paddle.randint(0, 20, shape=[6, 5])
            y = paddle.to_tensor([0, 1, 2, 0, 1, 2], dtype='int64')
            initial, final = None, None
            for _ in range(30):
                loss = F.cross_entropy(m(X), y)
                if initial is None:
                    initial = loss.item()
                loss.backward()
                opt.step()
                opt.clear_grad()
                final = loss.item()
            assert final < initial, f"loss did not decrease: {initial} -> {final}"

        def t_chap8_accuracy():
            acc = nndl.Accuracy(is_logist=True)
            outputs = paddle.to_tensor([[2.0, -1.0], [-0.5, 1.5], [3.0, 0.0]])
            labels = paddle.to_tensor([[0], [1], [0]], dtype='int64')
            acc.update(outputs, labels)
            assert acc.accumulate() == 1.0

        r.run("scaled dot-product attention shape + weights sum", t_scaled_dot_product_attention)
        r.run("causal mask via paddle.where", t_masked_attention)
        r.run("nn.MultiHeadAttention forward", t_multihead_attention_layer)
        r.run("nn.Embedding forward", t_embedding)
        r.run("Attention classifier training (loss decreases)", t_attention_classifier_train)
        r.run("chap8.Accuracy metric", t_chap8_accuracy)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
