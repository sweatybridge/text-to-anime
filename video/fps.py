import json

import cv2

from constants import RAW_VIDEO_DIR


def main():
    fps = {}
    videos = RAW_VIDEO_DIR.glob("**/*.mp4")
    for video in sorted(videos):
        cap = cv2.VideoCapture(str(video))
        fps[video.stem] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    with open(RAW_VIDEO_DIR / "fps.json", "w") as f:
        json.dump(fps, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
