"""chap6 循环神经网络: custom SimpleRNN cell + paddle nn.SimpleRNN/LSTM forward+backward."""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter, load_file_module
from pathlib import Path


def main():
    r = TestReporter("chap6 循环神经网络")
    pkg = push_chapter_pkg("chap6循环神经网络")
    try:
        # chap6 has nndl.py at top level (no nndl/ subpackage)
        nndl = load_file_module("chap6_nndl", Path(pkg) / "nndl.py")

        # Custom Simple Recurrent Network cell (chap6 §6.1)
        class SRN(nn.Layer):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.W = self.create_parameter([input_size, hidden_size])
                self.U = self.create_parameter([hidden_size, hidden_size])
                self.b = self.create_parameter([hidden_size])

            def forward(self, inputs):
                # inputs: [B, T, D]
                B, T, _ = inputs.shape
                h = paddle.zeros([B, self.hidden_size])
                for t in range(T):
                    x_t = inputs[:, t, :]
                    h = paddle.tanh(paddle.matmul(x_t, self.W) + paddle.matmul(h, self.U) + self.b)
                return h

        def t_custom_srn_forward():
            paddle.seed(0)
            rnn = SRN(input_size=4, hidden_size=6)
            x = paddle.randn([3, 5, 4])
            h = rnn(x)
            assert h.shape == [3, 6]

        def t_custom_srn_backward():
            paddle.seed(0)
            rnn = SRN(input_size=4, hidden_size=6)
            head = nn.Linear(6, 2)
            opt = paddle.optimizer.SGD(0.01, parameters=list(rnn.parameters()) + list(head.parameters()))
            x = paddle.randn([3, 5, 4])
            y = paddle.to_tensor([0, 1, 0], dtype='int64')
            initial, final = None, None
            for _ in range(10):
                h = rnn(x)
                logits = head(h)
                loss = F.cross_entropy(logits, y)
                if initial is None:
                    initial = loss.item()
                loss.backward()
                opt.step()
                opt.clear_grad()
                final = loss.item()
            assert final < initial

        def t_paddle_simple_rnn():
            rnn = nn.SimpleRNN(input_size=4, hidden_size=8)
            x = paddle.randn([2, 6, 4])
            y, _ = rnn(x)
            assert y.shape == [2, 6, 8]

        def t_paddle_lstm_bidirectional():
            rnn = nn.LSTM(input_size=4, hidden_size=8, direction='bidirectional')
            x = paddle.randn([2, 6, 4])
            y, _ = rnn(x)
            assert y.shape == [2, 6, 16]  # bidirectional doubles last dim

        def t_seq_classification_full_loop():
            """Tiny sequence classifier: LSTM + linear head, sigmoid-loss path."""
            paddle.seed(0)
            class SeqClassifier(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=8)
                    self.lstm = nn.LSTM(input_size=8, hidden_size=16, direction='forward')
                    self.fc = nn.Linear(16, 3)
                def forward(self, x):
                    e = self.embedding(x)
                    out, _ = self.lstm(e)
                    return self.fc(out[:, -1, :])
            model = SeqClassifier()
            opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
            X = paddle.randint(0, 20, shape=[8, 5])
            y = paddle.to_tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype='int64')
            initial, final = None, None
            for _ in range(20):
                logits = model(X)
                loss = F.cross_entropy(logits, y)
                if initial is None:
                    initial = loss.item()
                loss.backward()
                opt.step()
                opt.clear_grad()
                final = loss.item()
            assert final < initial, f"loss did not decrease: {initial} -> {final}"

        def t_accuracy_metric():
            acc = nndl.Accuracy(is_logist=True)
            outputs = paddle.to_tensor([[2.0, -1.0], [-0.5, 1.5], [3.0, 0.0]])
            labels = paddle.to_tensor([[0], [1], [0]], dtype='int64')
            acc.update(outputs, labels)
            assert acc.accumulate() == 1.0

        r.run("custom SRN forward", t_custom_srn_forward)
        r.run("custom SRN training loop (loss decreases)", t_custom_srn_backward)
        r.run("paddle nn.SimpleRNN forward", t_paddle_simple_rnn)
        r.run("paddle nn.LSTM bidirectional forward", t_paddle_lstm_bidirectional)
        r.run("LSTM seq classifier full training loop", t_seq_classification_full_loop)
        r.run("chap6.Accuracy metric", t_accuracy_metric)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
