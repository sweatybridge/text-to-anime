from io import StringIO
from pathlib import Path

import cv2
import pandas as pd


def parse_data(fp):
    with open(fp, "r") as f:
        contents = f.read()
    parts = contents.split("\n\n")

    meta = {}
    for line in parts[0].split("\n"):
        key, value = line.split(":")
        meta[key] = value.strip()

    bbox = pd.read_csv(StringIO(parts[1]), sep="\s+")
    text = (
        pd.read_csv(StringIO(parts[2]), sep="\s+") if len(parts) > 2 else pd.DataFrame()
    )
    return meta, bbox, text


def export_frames(video_id, train=True):
    capture = cv2.VideoCapture(f"data/{video_id}.mp4")
    cv2.namedWindow(winname="frame")
    fps = capture.get(cv2.CAP_PROP_FPS)

    label_root = Path("lrs3_v0.4")
    label_dir = "pretrain" if train else "trainval"
    label_path = label_root / label_dir / video_id
    for fp in sorted(label_path.glob("*.txt")):
        print(f"Processing: {fp}")
        meta, bbox, text = parse_data(fp)
        bbox["FRAME"] *= fps / 25
        bbox.set_index("FRAME", inplace=True)
        if not text.empty:
            text.set_index("START", inplace=True)

        start = bbox.index.min()
        end = bbox.index.max()
        offset = start / fps
        while capture.isOpened():
            _, src = capture.read()
            frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if frame < start:
                continue
            if frame > end:
                break

            # TODO: filter out frames without speech
            path = (
                Path("labels")
                / label_dir
                / video_id
                / fp.stem
                / f"frame_{int(frame):05}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), src)

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    export_frames(video_id="1BHOflzxPjI", train=False)
