from dataclasses import dataclass

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
    epochs = 500
    iters_per_checkpoint = 1000
    seed = 1234
    fp16_run = False
    cudnn_enabled = True
    cudnn_benchmark = False
    ignore_layers = ["embedding.weight"]
    ################################
    # Data Parameters             #
    ################################
    training_files = "filelists/ljs_audio_text_train_filelist.txt"
    validation_files = "filelists/ljs_audio_text_val_filelist.txt"
    text_cleaners = ["english_cleaners"]
    ################################
    # Landmark Parameters          #
    ################################
    n_mel_channels = 204
    ################################
    # Model Parameters             #
    ################################
    n_symbols = len(symbols)
    symbols_embedding_dim = 512
    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512
    # Decoder parameters
    n_frames_per_step = 1  # currently only 1 is supported
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1
    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128
    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31
    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5
    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate = False
    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    batch_size = 64
    mask_padding = True  # set model's padded outputs to padded values
