from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from animate import create_anime, normalize
from model import HParams, TextLandmarkModel
from text import text_to_sequence


def load_model(path: Path) -> TextLandmarkModel:
    # xyz coordinates * 20 lip landmarks
    params = HParams(n_landmark_xyz=60, pretrain=False)
    model = TextLandmarkModel(params)
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_lips(path: Path) -> np.ndarray:
    # Load reference facial landmarks
    df = pd.read_csv(path)
    lips = normalize(df.iloc[0])
    lips -= lips.mean(axis=0)
    return lips.reshape(-1)


def predict(model: TextLandmarkModel, face: np.ndarray, text: str) -> np.ndarray:
    # Convert input text to embeddings
    sequence = text_to_sequence(text, ["english_cleaners"])
    sequence = torch.IntTensor(sequence)[None, :].long()
    if torch.cuda.is_available():
        sequence = sequence.cuda()
    mel_outputs = model.inference(sequence)[0]
    # Animate lips only
    residual = mel_outputs.float().data.cpu().numpy()[0]
    data = np.zeros(shape=(residual.shape[1], face.shape[0]))
    for i in range(data.shape[0]):
        data[i, :] = face
    data[:, 144:] += residual.T
    return data


def main(
    artefact: Path,
    face: Path,
    output: Path,
    text: Optional[str] = None,
    file: Optional[Path] = None,
) -> None:
    torch.manual_seed(1234)
    model = load_model(artefact)
    lips = load_lips(face)
    # Load input text
    if file:
        text = file.read_text()
    assert text, "Input text is empty"
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(text.split("\n")):
        if not line:
            continue
        data = predict(model, lips, line)
        anime = create_anime(data)
        # Save each line as individual video
        path = output / f"line_{i}.mp4"
        anime.save(path)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        help="Input text to generate lip movements",
    )
    group.add_argument(
        "--file",
        type=Path,
        help="File containing line separated input text",
    )
    # Optional args
    parser.add_argument(
        "--artefact",
        help="Path to model weights checkpoint file",
        type=Path,
        default="artefact/best-lips.pt",
    )
    parser.add_argument(
        "--face",
        help="Path to reference face model",
        type=Path,
        # default="clean/trainval/0d6iSvF1UmA/00009.csv",
        default="clean/trainval/Actor_01/02-01-05-01-01-01-01.csv",
    )
    parser.add_argument(
        "--output",
        help="Path to video output directory",
        type=Path,
        default="output",
    )
    args = parser.parse_args()
    main(**vars(args))
