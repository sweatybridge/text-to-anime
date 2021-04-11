from glob import glob
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
    text = pd.read_csv(StringIO(parts[2]), sep="\s+")
    return meta, bbox, text


def main(save=False):
    fp = sorted(glob("lrs3_v0.4/pretrain/0Bhk65bYSI0/*.txt"))[0]
    meta, bbox, text = parse_data(fp)
    capture = cv2.VideoCapture(f"data/{meta['Ref']}.mp4")
    cv2.namedWindow(winname="frame")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps, text.columns)

    bbox["FRAME"] *= fps / 25
    bbox.set_index("FRAME", inplace=True)
    text.set_index("START", inplace=True)

    start = bbox.index.min()
    offset = start / fps
    while capture.isOpened():
        _, src = capture.read()
        frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if frame < start:
            continue

        if not save:
            idx = bbox.index.get_loc(frame, method="nearest")
            curr = bbox.iloc[idx]

            left = int(curr["X"] * width)
            right = left + int(curr["W"] * width)
            top = int(curr["Y"] * height)
            bottom = top + int(curr["H"] * height)
            src = cv2.rectangle(
                img=src,
                pt1=(left, top),
                pt2=(right, bottom),
                color=(0, 255, 0),
                thickness=3,
            )

        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - offset
        if timestamp > text.index[-1]:
            break

        if timestamp >= text.index[0]:
            idx = text.index.get_loc(timestamp, method="pad")
            curr = text.iloc[idx]
            if timestamp <= curr["END"]:
                src = cv2.putText(
                    img=src,
                    text=curr["WORD"],
                    org=(int(width / 2), int(height - 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

        if save:
            path = Path("data") / meta["Ref"]
            path.mkdir(exist_ok=True)
            cv2.imwrite(str(path / f"frame_{int(frame):04}.png"), src)
        else:
            cv2.imshow("frame", src)
            k = cv2.waitKey(0)
            if k == 27:  # esc key
                break
            elif k == ord("s"):
                cv2.imwrite("saved.png", src)
                continue

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main(save=True)
