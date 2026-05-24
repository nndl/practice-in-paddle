"""chap5 卷积神经网络: custom Conv2D + Paddle_LeNet + ResBlock + ResNet18 (tiny inputs)."""
import paddle
from _common import push_chapter_pkg, pop_chapter_pkg, TestReporter


def main():
    r = TestReporter("chap5 卷积神经网络")
    pkg = push_chapter_pkg("chap5卷积神经网络")
    try:
        from nndl.op import Conv2D, Pool2D, Model_LeNet, Paddle_LeNet, ResBlock, Model_ResNet18

        def t_custom_conv2d_forward():
            paddle.seed(0)
            conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3)
            x = paddle.randn([2, 1, 6, 6])
            y = conv(x)
            assert y.shape == [2, 2, 4, 4]

        def t_pool2d():
            pool = Pool2D(size=(2, 2), mode='max', stride=2)
            x = paddle.arange(0, 16, dtype='float32').reshape([1, 1, 4, 4])
            y = pool(x)
            assert y.shape == [1, 1, 2, 2]

        def t_paddle_lenet_forward():
            # LeNet expects 32x32 input
            model = Paddle_LeNet(in_channels=1, num_classes=10)
            x = paddle.randn([2, 1, 32, 32])
            y = model(x)
            assert y.shape == [2, 10]

        def t_paddle_lenet_train_step():
            paddle.seed(0)
            model = Paddle_LeNet(in_channels=1, num_classes=3)
            opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            loss_fn = paddle.nn.CrossEntropyLoss()
            X = paddle.randn([4, 1, 32, 32])
            y = paddle.to_tensor([0, 1, 2, 0], dtype='int64')
            initial_loss = None
            for step in range(8):
                logits = model(X)
                loss = loss_fn(logits, y)
                if initial_loss is None:
                    initial_loss = loss.item()
                loss.backward()
                opt.step()
                opt.clear_grad()
            final_loss = loss.item()
            assert final_loss < initial_loss

        def t_resblock_forward():
            block = ResBlock(in_channels=4, out_channels=8, stride=2, use_residual=True)
            block.eval()  # BatchNorm in eval mode for stable test
            x = paddle.randn([2, 4, 8, 8])
            y = block(x)
            assert y.shape == [2, 8, 4, 4]

        def t_resnet18_forward():
            # tiny ResNet-18 forward on small input
            model = Model_ResNet18(in_channels=3, num_classes=5, use_residual=True)
            model.eval()
            x = paddle.randn([2, 3, 32, 32])
            y = model(x)
            assert y.shape == [2, 5]

        def t_resnet18_backward():
            paddle.seed(0)
            model = Model_ResNet18(in_channels=3, num_classes=4, use_residual=True)
            opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
            loss_fn = paddle.nn.CrossEntropyLoss()
            X = paddle.randn([2, 3, 32, 32])
            y = paddle.to_tensor([0, 3], dtype='int64')
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            # check at least one parameter got a gradient
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert len(grads) > 0

        r.run("custom Conv2D forward", t_custom_conv2d_forward)
        r.run("custom Pool2D forward", t_pool2d)
        r.run("Paddle_LeNet forward (32x32)", t_paddle_lenet_forward)
        r.run("Paddle_LeNet 1-step training loss decreases", t_paddle_lenet_train_step)
        r.run("ResBlock forward (stride=2)", t_resblock_forward)
        r.run("Model_ResNet18 forward", t_resnet18_forward)
        r.run("Model_ResNet18 backward (gradients)", t_resnet18_backward)
    finally:
        pop_chapter_pkg(pkg)
    return r.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
