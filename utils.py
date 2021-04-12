from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from text import symbols


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


@dataclass(frozen=True)
class HParams:
    ################################
    # Experiment Parameters        #
    ################################
    epochs: int = 500
    iters_per_checkpoint: int = 1000
    seed: int = 1234
    fp16_run: bool = False
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = False
    ignore_layers: List[str] = field(default_factory=lambda: ["embedding.weight"])
    ################################
    # Data Parameters             #
    ################################
    training_files: str = "filelists/ljs_audio_text_train_filelist.txt"
    validation_files: str = "filelists/ljs_audio_text_val_filelist.txt"
    text_cleaners: List[str] = field(default_factory=lambda: ["english_cleaners"])
    ################################
    # Landmark Parameters          #
    ################################
    n_mel_channels: int = 204
    ################################
    # Model Parameters             #
    ################################
    n_symbols: int = len(symbols)
    symbols_embedding_dim: int = 512
    # Encoder parameters
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512
    # Decoder parameters
    n_frames_per_step: int = 1  # currently only 1 is supported
    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1
    # Attention parameters
    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    # Location Layer parameters
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    # Mel-post processing network parameters
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5
    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate: bool = False
    scheduler_step: int = 4000
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    batch_size: int = 64
    mask_padding: bool = True  # set model's padded outputs to padded values
