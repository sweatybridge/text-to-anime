from pathlib import Path

from face import load_frames, normalize


def merge(video_id, train=True):
    label_dir = "pretrain" if train else "trainval"
    clean = Path("clean") / label_dir / video_id
    clean.mkdir(parents=True, exist_ok=True)

    path = Path("noisy") / label_dir / video_id
    for fp in sorted(path.glob("*")):
        df = load_frames(fp / "processed")
        invalid = (df["confidence"] < 0.7).sum()
        print(f"{fp}: {invalid}")
        if not invalid:
            cp = (clean / fp.stem).with_suffix(".csv")
            df.to_csv(str(cp))


if __name__ == "__main__":
    # merge(video_id="1BHOflzxPjI", train=False)
    noisy = Path("noisy") / "pretrain"
    for path in sorted(noisy.glob("*")):
        merge(video_id=path.stem)
        merge(video_id=path.stem, train=False)
