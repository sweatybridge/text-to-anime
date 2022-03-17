import matplotlib.pyplot as plt
import numpy as np


def read_loss(path):
    train, val = [], []
    with open(path, "r") as f:
        iter_loss = []
        for line in f:
            if line.startswith("Epoch"):
                train.append(np.mean(iter_loss))
                iter_loss = []
            elif line.startswith("Validation loss"):
                loss = float(line.split(":")[-1].strip())
                val.append(loss)
            elif line.startswith("Train loss"):
                loss = float(line.split(" ")[3].strip())
                iter_loss.append(loss)
    return train, val


def render(train, val, title):
    plt.plot([i + 1 for i in range(len(train))], train)
    plt.plot([(i + 1) * 5 for i in range(len(val))], val)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.legend(["Train", "Val"])
    plt.tight_layout()
    plt.show()


def compare(train, title, legend):
    for data in train:
        plt.plot([i + 1 for i in range(len(data))], data)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.legend(legend)
    plt.tight_layout()
    plt.show()


def lips_vs_full():
    train, val = read_loss("report/loss_all.txt")
    render(train, val, "Model A (68 landmarks)")
    train, val = read_loss("report/loss_lips.txt")
    render(train, val, "Model B (lips only)")


def pretrain_vs_no():
    train, val = read_loss("loss_60.txt")
    train_p, val_p = read_loss("loss_60_pretrain.txt")
    compare([train, train_p], "Train loss (lips only)", ["No Pretrain", "Pretrain"])
    compare([val, val_p], "Val loss (lips only)", ["No Pretrain", "Pretrain"])


def pre_post_none():
    train_none, val_none = read_loss("loss/none.txt")
    train_pre, val_pre = read_loss("loss/pre.txt")
    train_post, val_post = read_loss("loss/post.txt")
    legend = ["No Pre / Post", "Only Pre", "Pre + Post"]
    compare([train_none, train_pre, train_post], "Training loss", legend)
    compare([val_none, val_pre, val_post], "Validation loss", legend)


def none_500():
    train, val = read_loss("loss/500_none.txt")
    render(train, val, "No Pre/Post (lips only)")


if __name__ == "__main__":
    # lips_vs_full()
    none_500()
    # pre_post_none()
