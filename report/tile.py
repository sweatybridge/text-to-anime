from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def extract(path: Path, skip: int = 2, total: int = 7):
    output = []
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"{path}: {fps}")

    count = 0
    while capture.isOpened():
        if count >= total:
            break
        _, src = capture.read()
        if src is None:
            break
        frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        # frame = int(frame / fps * 25)
        if frame % skip:
            continue
        cropped = src[50:-50, 130:-130, :]
        output.append(cropped)
        count += 1

    capture.release()
    return np.concatenate(output, axis=1)


def main(root: Path):
    merged = [extract(p, skip=15) for p in sorted(root.glob("lips_*.mp4"))]
    # merged.append(merged[0][:, 2660:, :])
    # merged[0] = merged[0][:, :2660, :]
    img = np.concatenate(merged, axis=0)
    output = root / "landmarks_68.png"
    cv2.imwrite(str(output), img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        dest="root",
        type=Path,
        help="Path to source videos",
        default="output/00009",
    )
    args = parser.parse_args()
    main(**vars(args))
