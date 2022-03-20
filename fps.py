import json
from pathlib import Path

import cv2


def main():
    root = Path("video")
    fps = {}
    for video in sorted(root.glob("**/*.mp4")):
        cap = cv2.VideoCapture(str(video))
        fps[video.stem] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    with open(root / "fps.json", "w") as f:
        json.dump(fps, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
