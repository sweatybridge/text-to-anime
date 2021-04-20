import numpy as np
import matplotlib.pyplot as plt


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


def main():
    train, val = read_loss("report/loss_all.txt")
    render(train, val, "Model A (68 landmarks)")
    train, val = read_loss("report/loss_lips.txt")
    render(train, val, "Model B (lips only)")


if __name__ == "__main__":
    main()
