from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from face import normalize
from preprocess import parse_data
from text import text_to_sequence


class TextLandmarkLoader(Dataset):
    """
    1) loads video, text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    4) normalizes facial landmarks from openface
    """

    def __init__(self, train=True):
        self.label_dir = "pretrain" if train else "trainval"
        label_path = Path("clean") / self.label_dir
        self.landmark_paths = list(sorted(label_path.glob("*/*.csv")))
        self.text_cleaners = ["english_cleaners"]

    def get_landmarks(self, path):
        df = pd.read_csv(str(path))
        norm = [normalize(row).reshape(-1) for _, row in df.iterrows()]
        return torch.FloatTensor(norm).t()

    def get_text(self, path):
        meta, _, _ = parse_data(path)
        norm = text_to_sequence(meta["Text"], self.text_cleaners)
        return torch.IntTensor(norm)

    def __getitem__(self, index):
        fp = self.landmark_paths[index]
        landmarks = self.get_landmarks(fp)
        clip = Path("lrs3_v0.4") / self.label_dir / fp.parent.stem / fp.stem
        text = self.get_text(clip.with_suffix(".txt"))
        return (text, landmarks)

    def __len__(self):
        return len(self.landmark_paths)


class TextLandmarkCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self):
        self.n_frames_per_step = 1

    def __call__(self, batch):
        """Collate's training batch from normalized text and facial landmarks
        PARAMS
        ------
        batch: [text_normalized, landmarks_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad landmark positions
        num_landmarks = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # Include landmark padded and gate padded
        landmarks_padded = torch.FloatTensor(len(batch), num_landmarks, max_target_len)
        landmarks_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            landmarks_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, landmarks_padded, gate_padded, output_lengths


if __name__ == "__main__":
    trainset = TextLandmarkLoader()
    text, landmarks = trainset[0]
    print(text.shape, landmarks.shape)
    collate_fn = TextLandmarkCollate()
    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    for batch in train_loader:
        for item in batch:
            print(item.shape)
        break
