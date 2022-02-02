from glob import glob
from pathlib import Path

import cv2

from preprocess import parse_data


def main(video_id, save=False):
    capture = cv2.VideoCapture(f"data/{video_id}.mp4")
    cv2.namedWindow(winname="frame")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # fp = sorted(glob("lrs3_v0.4/pretrain/1BHOflzxPjI/*.txt"))[0]
    fp = sorted(glob(f"lrs3_v0.4/trainval/{video_id}/*.txt"))[0]
    meta, bbox, text = parse_data(fp)
    print(fps, text.columns)
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

        if not text.empty:
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - offset
            if timestamp > text.index[-1]:
                continue

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
            path = Path("valdata" if "val" in fp.split("/")[1] else "data") / video_id
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
    main(video_id="1BHOflzxPjI", save=True)
