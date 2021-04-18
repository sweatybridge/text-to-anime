from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, cm


def normalize(df):
    pos = [(df[f"X_{i}"], df[f"Y_{i}"], df[f"Z_{i}"]) for i in range(68)]
    T = (df["pose_Tx"], df["pose_Ty"], df["pose_Tz"])
    pitch, yaw, roll = (df["pose_Rx"], df["pose_Ry"], df["pose_Rz"])
    # Left handed coordinate system
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), np.sin(pitch)],
            [0, -np.sin(pitch), np.cos(pitch)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(yaw), 0, -np.sin(yaw)],
            [0, 1, 0],
            [np.sin(yaw), 0, np.cos(yaw)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(roll), np.sin(roll), 0],
            [-np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ]
    )
    R = R_x @ R_y @ R_z
    # print(T, R, pos[0], sep="\n")
    norm = -R @ (np.array(pos) - T).T
    return norm.T


def render(df):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    norm = normalize(df)
    # X, Y, Z = zip(*norm)
    ax.scatter(norm[:, 0], norm[:, 1], norm[:, 2], marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=100, azim=-90)
    plt.tight_layout()
    plt.show()


def load_frames(root: Path):
    processed = root / "all_frames.csv"
    if processed.exists():
        return pd.read_csv(str(processed))

    data = [
        pd.read_csv(fp, sep=", ", engine="python").nlargest(1, ["confidence"])
        for fp in sorted(glob(str(root / "frame_*.csv")))
    ]
    df = pd.concat(data)
    df.to_csv(str(root / "all_frames.csv"))
    return df


def main():
    root = Path("data") / "1BHOflzxPjI" / "processed"
    df = load_frames(root)
    # return render(df.iloc[0])
    print(df["confidence"].mean(), (df["confidence"] < 0.7).sum())

    start = None
    cv2.namedWindow("frame")
    for fp in sorted(glob(str(root / "*.jpg"))):
        frame = int(fp.split("_")[-1].split(".")[0])
        if start is None:
            start = frame

        if df.iloc[frame - start]["confidence"] < 0.7:
            continue

        img = cv2.imread(fp, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("frame", img)
        k = cv2.waitKey(0)
        if k == 27:
            break


def get_bounds(val):
    return [val.min() - 1, val.max() + 1]


def update_surface(frame, ax, limits):
    ax.clear()
    ax.set_xlim3d(limits[0])
    ax.set_ylim3d(limits[1])
    ax.set_zlim3d(limits[2])
    surf = ax.plot_trisurf(frame[:, 0], frame[:, 1], frame[:, 2], cmap=cm.coolwarm)
    markers = ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], marker="o")
    return surf, markers


def animate(data, save=False):
    if len(data.shape) == 2:
        dim = (data.shape[0] // 3, 3, -1)
        data = data.reshape(dim).transpose(2, 0, 1)

    print(data[0].shape)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2], marker="o")
    ax.plot_trisurf(data[0, :, 0], data[0, :, 1], data[0, :, 2], cmap=cm.coolwarm)

    limits = [get_bounds(data[:, :, i]) for i in range(3)]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=100, azim=-90)
    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig=fig,
        frames=data,
        func=update_surface,
        fargs=(ax, limits),
        interval=40,
    )
    if save:
        anim.save("landmarks_color.mp4")
    else:
        plt.show()


def show_ground_truth():
    # df = pd.read_csv("clean/trainval/1BHOflzxPjI/00002.csv")
    df = pd.read_csv("clean/trainval/0d6iSvF1UmA/00009.csv")
    data = np.empty((len(df), 20, 3))
    for i, row in df.iterrows():
        lips = normalize(row)[48:]
        lips -= lips.mean(axis=0)
        data[i] = lips
    animate(data, save=True)


def main():
    # df = pd.read_csv("clean/trainval/1BHOflzxPjI/00002.csv")
    df = pd.read_csv("clean/trainval/0d6iSvF1UmA/00009.csv")
    lips = normalize(df.iloc[0])[48:]
    lips -= lips.mean(axis=0)
    # data = np.array([lips])
    norm = lips.reshape(-1)
    residual = np.load("output_post_res.npy")
    data = (residual.T + norm).T
    # df = pd.read_csv("clean/trainval/1BHOflzxPjI/00002.csv")
    # df = pd.read_csv("clean/pretrain/1BHOflzxPjI/00008.csv")
    # data = np.array([normalize(row)[48:] for _, row in df.iterrows()])
    animate(data, save=True)


if __name__ == "__main__":
    # show_ground_truth()
    main()
