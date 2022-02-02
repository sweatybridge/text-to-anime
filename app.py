from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from face import create_anime, normalize
from model import Tacotron2
from text import text_to_sequence
from utils import HParams


@st.cache
def load_model(path: str) -> torch.nn.Module:
    # xyz coordinates * 20 lip landmarks, 8 seconds * 30 fps
    model = Tacotron2(HParams(n_mel_channels=60, max_decoder_steps=240))
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model


@st.cache
def load_face(path: Optional[str] = None) -> np.ndarray:
    if not path:
        return np.genfromtxt("artefact/face.csv")
    # Load base facial landmarks
    df = pd.read_csv(path)
    lips = normalize(df.iloc[0])
    lips -= lips.mean(axis=0)
    return lips.reshape(-1)


def main():
    st.title("Text to lip movements")
    model = load_model("artefact/best-lips.pt")
    face = load_face()
    text = st.text_input(
        "Enter a short phrase or sentence:",
        max_chars=140,
        placeholder="Hello World!",
    )
    if not text:
        return
    with st.spinner("Running model inference..."):
        # Convert input text to embeddings
        sequence = text_to_sequence(text, ["english_cleaners"])
        sequence = torch.IntTensor(sequence)[None, :].long()
        if torch.cuda.is_available():
            sequence = sequence.cuda()
        mel_outputs = model.inference(sequence)[0]
    with st.spinner("Rendering output video..."):
        # Animate lips only
        residual = mel_outputs.float().data.cpu().numpy()[0]
        data = np.zeros(shape=(residual.shape[1], face.shape[0]))
        for i in range(data.shape[0]):
            data[i, :] = face
        data[:, 144:] += residual.T
        anime = create_anime(data.T)
        # Render output video
        video = anime.to_html5_video()
    st.write(video, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
