import matplotlib.pyplot as plt

#新增绘制图像方法
def plot(runner,fig_name):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    epochs = [i for i in range(len(runner.train_scores))]
    #绘制训练损失变化曲线
    plt.plot(epochs, runner.train_loss, color='#e4007f', label="Train loss")
    #绘制评价损失变化曲线
    plt.plot(epochs, runner.dev_loss, color='#f19ec2', linestyle='--', label="Dev loss")
    #绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.subplot(1,2,2)
    #绘制训练准确率变化曲线
    plt.plot(epochs, runner.train_scores, color='#e4007f', label="Train accuracy")
    #绘制评价准确率变化曲线
    plt.plot(epochs, runner.dev_scores, color='#f19ec2', linestyle='--', label="Dev accuracy")
    #绘制坐标轴和图例
    plt.ylabel("score", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

# 绘制训练集和验证集的损失变化以及验证集上的准确率变化曲线
def plot_training_loss_acc(runner, fig_name, 
    fig_size=(16, 6), 
    sample_step=20, 
    loss_legend_loc="upper right", 
    acc_legend_loc="lower right",
    train_color="#e4007f",
    dev_color='#f19ec2',
    fontsize='large',
    train_linestyle="-",
    dev_linestyle='--'):

    plt.figure(figsize=fig_size)

    plt.subplot(1,2,1)
    train_items = runner.train_step_losses[::sample_step]
    train_steps=[x[0] for x in train_items]
    train_losses = [x[1] for x in train_items]

    plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle, label="Train loss")
    if len(runner.dev_losses)>0:
        dev_steps=[x[0] for x in runner.dev_losses]
        dev_losses = [x[1] for x in runner.dev_losses]
        plt.plot(dev_steps, dev_losses, color=dev_color, linestyle=dev_linestyle, label="Dev loss")
    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize=fontsize)
    plt.xlabel("step", fontsize=fontsize)
    plt.legend(loc=loss_legend_loc, fontsize='x-large')

    # 绘制评价准确率变化曲线
    if len(runner.dev_scores)>0:
        plt.subplot(1,2,2)
        plt.plot(dev_steps, runner.dev_scores,
            color=dev_color, linestyle=dev_linestyle, label="Dev accuracy")
    
        # 绘制坐标轴和图例
        plt.ylabel("score", fontsize=fontsize)
        plt.xlabel("step", fontsize=fontsize)
        plt.legend(loc=acc_legend_loc, fontsize='x-large')

    plt.savefig(fig_name)
    plt.show()