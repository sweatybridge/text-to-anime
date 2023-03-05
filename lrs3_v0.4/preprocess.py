from pathlib import Path

import cv2

from constants import LANDMARK_NOISY_DIR, RAW_VIDEO_DIR
from label import parse_data


def export_frames(path: Path, train=True):
    video_id = path.stem
    capture = cv2.VideoCapture(str(path))
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
        # offset = start / fps
        while capture.isOpened():
            _, src = capture.read()
            frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if frame < start:
                continue
            if frame > end:
                break

            # TODO: filter out frames without speech
            path = (
                LANDMARK_NOISY_DIR
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
    # export_frames(path=Path("data") / "1BHOflzxPjI.mp4", train=False)
    videos = RAW_VIDEO_DIR.glob("*.mp4")
    for path in sorted(videos):
        export_frames(path)
        export_frames(path, train=False)
