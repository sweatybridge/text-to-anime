from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    norm = [-R @ (np.array(p) - T) for p in pos]
    return np.array(norm)


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
        pd.read_csv(fp, sep=", ").nlargest(1, ["confidence"])
        for fp in sorted(glob(str(root / "frame_*.csv")))
    ]
    df = pd.concat(data)
    df.to_csv(str(root / "all_frames.csv"))
    return df


def render():
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


def main(video_id, train=True):
    label_dir = "pretrain" if train else "trainval"
    clean = Path("clean") / label_dir / video_id
    clean.mkdir(parents=True, exist_ok=True)

    path = Path("labels") / label_dir / video_id
    for fp in sorted(path.glob("*")):
        df = load_frames(fp / "processed")
        invalid = (df["confidence"] < 0.7).sum()
        print(f"{fp}: {invalid}")
        if not invalid:
            cp = (clean / fp.stem).with_suffix(".csv")
            df.to_csv(str(cp))


if __name__ == "__main__":
    main(video_id="1BHOflzxPjI", train=False)
