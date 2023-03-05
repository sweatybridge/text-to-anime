from pathlib import Path

from fire import Fire

from animate.face import load_frames


def merge_frames(fp: Path):
    df = load_frames(fp)
    invalid = (df["confidence"] < 0.7).sum()
    print(f"{fp}: {invalid}")
    if not invalid:
        out = Path("clean").joinpath(*fp.parts[1:4])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out.with_suffix(".csv")))


def main(emotion: str = "*", actor: str = "*"):
    processed = Path("noisy").glob(f"{emotion}/{actor}/*/processed")
    for fp in sorted(processed):
        merge_frames(fp)


if __name__ == "__main__":
    Fire(main)
