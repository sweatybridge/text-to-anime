from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from face import create_anime
from model import TextLandmarkModel
from score import load_lips, load_model, predict


@st.cache
def init() -> Tuple[TextLandmarkModel, np.ndarray]:
    model = load_model(Path("artefact/best.pt"))
    lips = load_lips(Path("clean/trainval/0d6iSvF1UmA/00009.csv"))
    return model, lips


def main() -> None:
    st.title("Text to lip movements")
    model, lips = init()
    text = st.text_input(
        "Enter a short phrase or sentence:",
        max_chars=140,
        placeholder="Hello World!",
    )
    if not text:
        return
    with st.spinner("Running model inference..."):
        data = predict(model, lips, text)
    with st.spinner("Rendering output video..."):
        anime = create_anime(data)
        video = anime.to_jshtml()
    # FIXME: video doesn't reload when input text changes
    components.html(video, height=600)


if __name__ == "__main__":
    main()
