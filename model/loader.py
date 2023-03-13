import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from animate import normalize
from label import parse_data
from text import text_to_sequence


class TextLandmarkLoader(Dataset):
    """
    1) loads video, text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    4) normalizes facial landmarks from openface
    """

    def __init__(self, train=True):
        self.label_dir = "pretrain_2" if train else "trainval_2"
        label_path = Path("landmark/clean") / self.label_dir
        self.landmark_paths = list(sorted(label_path.glob("*/*.csv")))
        with open("video/fps.json", "r") as f:
            self.fps = json.load(f)
        self.speaker_embedding = {}
        for path in self.landmark_paths:
            speaker = path.stem
            if speaker not in self.speaker_embedding:
                self.speaker_embedding[speaker] = []
            df = pd.read_csv(str(path)).iloc[0]
            norm = self.normalize_landmarks(df)
            self.speaker_embedding[speaker].append(norm)
        for k, v in self.speaker_embedding.items():
            self.speaker_embedding[k] = np.mean(v, axis=0)
        self.text_cleaners = ["english_cleaners"]

    def normalize_landmarks(self, df, video_id=None):
        lips = normalize(df)[48:]
        lips -= lips.mean(axis=0)
        result = lips.reshape(-1)
        if video_id is not None:
            result -= self.speaker_embedding[video_id]
        return result

    def get_landmarks(self, path):
        video_id = path.stem
        df = pd.read_csv(str(path))
        # Only load lip positions
        norm = np.zeros(shape=(df.shape[0], 60))
        for i, row in df.iterrows():
            norm[i] = self.normalize_landmarks(row, video_id)
        # Interpolate to 12.5 ms frame hop (ie. 80 fps)
        fps = self.fps[video_id]
        xp = np.arange(norm.shape[0]) / fps * 80
        frames = int(norm.shape[0] / fps * 80)
        xs = np.arange(frames)
        interpolated = np.zeros(shape=(frames, norm.shape[1]))
        for i in range(norm.shape[1]):
            interpolated[:, i] = np.interp(xs, xp, norm[:, i])
        return torch.FloatTensor(interpolated).t()

    def get_text(self, path):
        # meta, _, _ = parse_data(path)
        # norm = text_to_sequence(meta["Text"], self.text_cleaners)
        video_id = path.stem
        statement_id = video_id[13]
        if statement_id == "1":
            statement = text_to_sequence("KIDS ARE TALKING BY THE DOOR")
        elif statement_id == "2":
            statement = text_to_sequence("DOGS ARE SITTING BY THE DOOR")
        return torch.IntTensor(statement)
    
    def get_emotion(self, path):
        video_id = path.stem
        emotion_id = video_id[7]
        if emotion_id == "5":
            emotion = text_to_sequence("angry", self.text_cleaners)
        elif emotion_id == "8":
            emotion = text_to_sequence("surprised", self.text_cleaners)
        return torch.IntTensor(emotion)
    
    def __getitem__(self, index):
        fp = self.landmark_paths[index]
        landmarks = self.get_landmarks(fp)
        clip = Path("ravdess") / self.label_dir / fp.parent.stem / fp.stem
        text = self.get_text(clip.with_suffix(".txt"))
        emotion = self.get_emotion(clip.with_suffix(".txt"))
        return (text, landmarks, emotion)

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
        batch: [text_normalized, landmarks_normalized, emotion_normalized]
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

        # Right zero-pad emotion
        max_emotion_len = max([x[2].size(0) for x in batch])
        emotion_padded = torch.LongTensor(len(batch), max_emotion_len)
        emotion_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            emotion = batch[ids_sorted_decreasing[i]][2]
            emotion_padded[i, : emotion.size(0)] = emotion

        return text_padded, input_lengths, landmarks_padded, gate_padded, output_lengths, emotion_padded


if __name__ == "__main__":
    trainset = TextLandmarkLoader()
    print(f"Samples: {len(trainset)}")
    text, landmarks = trainset[0]
    print(f"Batch: {text.shape}, {landmarks.shape}")
    collate_fn = TextLandmarkCollate()
    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=8,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Steps: {len(train_loader)}")
    for batch in train_loader:
        for item in batch:
            print(item.shape)
        break
