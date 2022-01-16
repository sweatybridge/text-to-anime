from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pandas as pd
import torch

from face import animate, normalize
from model import Tacotron2
from text import text_to_sequence
from utils import HParams


def main(artefact: str, text: Optional[str] = None, file: Optional[str] = None):
    # xyz coordinates * 20 lip landmarks, 8 seconds * 30 fps
    model = Tacotron2(HParams(n_mel_channels=60, max_decoder_steps=240))
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(artefact, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    # Convert input text to embeddings
    sequence = text_to_sequence(text, ["english_cleaners"])
    sequence = torch.IntTensor(sequence)[None, :].long()
    if torch.cuda.is_available():
        sequence = sequence.cuda()
    mel_outputs = model.inference(sequence)[0]
    # Load base facial landmarks
    df = pd.read_csv("clean/trainval/0d6iSvF1UmA/00009.csv")
    lips = normalize(df.iloc[0])
    lips -= lips.mean(axis=0)
    norm = lips.reshape(-1)
    # Animate lips only
    residual = mel_outputs.float().data.cpu().numpy()[0]
    data = np.zeros(shape=(residual.shape[1], norm.shape[0]))
    for i in range(data.shape[0]):
        data[i, :] = norm
    data[:, 144:] += residual.T
    animate(data.T, save=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Input text to generate lip movements")
    group.add_argument("--file", help="File containing line separated input text")
    parser.add_argument(
        "--artefact",
        help="Path to model weights checkpoint file",
        default="artefact/best-lips.pt",
    )
    args = parser.parse_args()
    main(**vars(args))
