import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import Animation, FuncAnimation


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
    ax = fig.add_subplot(projection="3d")

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
        for fp in sorted(root.glob("frame_*.csv"))
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
    for fp in sorted(root.glob("*.jpg")):
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


def get_jaw(val):
    return val[:17]


def get_left_eye(val):
    eye = np.zeros(shape=(7, 3))
    eye[:6, :] = val[36:42]
    eye[6, :] = val[36]
    return eye


def get_left_brow(val):
    return val[17:22]


def get_right_eye(val):
    eye = np.zeros(shape=(7, 3))
    eye[:6, :] = val[42:48]
    eye[6, :] = val[42]
    return eye


def get_right_brow(val):
    return val[22:27]


def get_nose(val):
    nose = np.zeros(shape=(10, 3))
    nose[:9, :] = val[27:36]
    nose[9, :] = val[30]
    return nose


def get_upper_lip(val):
    lip = np.zeros(shape=(13, 3))
    lip[:7, :] = val[48:55]
    lip[7:12, :] = np.flip(val[60:65], axis=0)
    lip[12, :] = val[48]
    return lip


def get_lower_lip(val):
    lip = np.zeros(shape=(11, 3))
    lip[:6, :] = val[54:60]
    lip[6, :] = val[48]
    lip[7:10, :] = np.flip(val[65:], axis=0)
    lip[10, :] = val[54]
    return lip


def get_face(val):
    return [
        get_jaw(val),
        get_left_brow(val),
        get_left_eye(val),
        get_right_brow(val),
        get_right_eye(val),
        get_nose(val),
        get_upper_lip(val),
        get_lower_lip(val),
    ]


def get_bounds(val):
    return [val.min() - 1, val.max() + 1]


def snapshot(data):
    if len(data.shape) == 2:
        shape = (data.shape[0], data.shape[1] // 3, 3)
        data = data.reshape(shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2], marker="o")
    # ax.plot_trisurf(data[0, :, 0], data[0, :, 1], data[0, :, 2], cmap=cm.coolwarm)
    for line in get_face(data[0]):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="C0")

    limits = [get_bounds(data[0, :, i]) for i in range(3)]
    ax.set_xlim3d(limits[0])
    ax.set_ylim3d(limits[1])
    ax.set_zlim3d(limits[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=100, azim=-90)
    plt.tight_layout()
    plt.show()


def update_surface(frame, ax, limits):
    ax.clear()
    ax.set_xlim3d(limits[0])
    ax.set_ylim3d(limits[1])
    ax.set_zlim3d(limits[2])
    # surf = ax.plot_trisurf(frame[:, 0], frame[:, 1], frame[:, 2], cmap=cm.coolwarm)
    # markers = ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], marker="o")
    # return surf, markers
    markers = [
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="C0" if i < 6 else "C3")
        for i, line in enumerate(get_face(frame))
    ]
    # markers = ax.plot(frame[48:, 0], frame[48:, 1], frame[48:, 2])
    return markers


def create_anime(data: np.ndarray) -> Animation:
    if len(data.shape) == 2:
        shape = (data.shape[0], data.shape[1] // 3, 3)
        data = data.reshape(shape)
    # Expected input shape: (n, 68, 3)
    print(f"Input shape: {data.shape}")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    limits = [get_bounds(data[:, 48:, i]) for i in range(3)]
    # limits[1][0] -= 10
    # limits[1][1] += 10

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=100, azim=-90)
    plt.tight_layout()

    return FuncAnimation(
        fig=fig,
        frames=data,
        func=update_surface,
        fargs=(ax, limits),
        interval=13,
    )


def show_ground_truth():
    ref = Path("clean/trainval/0d6iSvF1UmA/00009.csv")
    df = pd.read_csv(str(ref))
    data = np.empty((len(df), 68, 3))
    for i, row in df.iterrows():
        lips = normalize(row)
        lips -= lips.mean(axis=0)
        data[i] = lips
    # Interpolate to 12.5 ms frame hop (ie. 80 fps)
    video = ref.parent.stem
    with open("video/fps.json", "r") as f:
        fps = json.load(f)[video]
    xp = np.arange(data.shape[0]) / fps * 80
    frames = int(data.shape[0] / fps * 80)
    print(f"Frames: {frames}")
    xs = np.arange(frames)
    interpolated = np.zeros(shape=(frames, data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        interpolated[:, i, 0] = np.interp(xs, xp, data[:, i, 0])
        interpolated[:, i, 1] = np.interp(xs, xp, data[:, i, 1])
        interpolated[:, i, 2] = np.interp(xs, xp, data[:, i, 2])
        # print(data[:, i, 0], interpolated[:, i, 0])
    anim = create_anime(interpolated)
    anim.save("landmarks_68.mp4")
    # plt.show()


if __name__ == "__main__":
    show_ground_truth()
    # main()
