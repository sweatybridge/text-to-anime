from enum import Enum
from pathlib import Path

import cv2
from fire import Fire


class Emotion(Enum):
    neutral = 1
    calm = 2
    happy = 3
    sad = 4
    angry = 5
    fearful = 6
    disgust = 7
    surprised = 8


def export_frames(path: Path):
    print("Processing", path)
    video_id = path.stem
    ids = video_id.split("-")
    emo = Emotion(int(ids[2]))
    # Create output directory
    noisy = Path("noisy") / emo.name / f"Actor_{ids[-1]}" / video_id
    noisy.mkdir(parents=True, exist_ok=True)
    # Process video
    capture = cv2.VideoCapture(str(path))
    cv2.namedWindow(winname="frame")
    # fps = capture.get(cv2.CAP_PROP_FPS)
    while capture.isOpened():
        _, src = capture.read()
        if src is None:
            break
        frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        path = noisy / f"frame_{int(frame):05}.png"
        cv2.imwrite(str(path), src)
    # Cleanup resources
    cv2.destroyAllWindows()
    capture.release()


def main(emotion: str):
    emo = Emotion[emotion]
    videos = Path("ravdess").glob(f"**/02-01-0{emo.value}-01-*.mp4")
    for fp in sorted(videos):
        export_frames(fp)


if __name__ == "__main__":
    Fire(main)
