import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from loader import TextLandmarkCollate, TextLandmarkLoader
from model import Tacotron2
from utils import HParams


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse(mel_out, mel_target)
        post_loss = self.mse(mel_out_postnet, mel_target)
        gate_loss = self.bce(gate_out, gate_target)
        return mel_loss + post_loss + gate_loss


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    scaler.load_state_dict(checkpoint_dict["scaler"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return learning_rate, iteration


def save_checkpoint(model, optimizer, scaler, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(model, criterion, valset, batch_size, collate_fn):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(
            valset,
            sampler=None,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    return val_loss


def main(hparams, checkpoint_path=None):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    hparams (object): comma separated list of "name=value" pairs.
    checkpoint_path(string): checkpoint path
    """
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = Tacotron2(hparams).cuda()
    # Initialise with pretrained weights and freeze
    pretrained = torch.hub.load("nvidia/DeepLearningExamples:torchhub", "nvidia_tacotron2").cuda()
    model.embedding.weight = pretrained.embedding.weight
    model.embedding.weight.requires_grad = False

    model.encoder.load_state_dict(pretrained.encoder.state_dict())
    for param in model.encoder.parameters():
        param.requires_grad = False

    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = np.finfo("float16").min

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=hparams.fp16_run)
    criterion = Tacotron2Loss()

    # Setup data loaders
    trainset = TextLandmarkLoader()
    valset = TextLandmarkLoader(train=False)
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

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        _learning_rate, iteration = load_checkpoint(
            checkpoint_path, model, optimizer, scaler
        )
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    best = 100
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.zero_grad()
            with torch.cuda.amp.autocast():
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss = criterion(y_pred, y)

            reduced_loss = loss.item()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh
            )
            is_overflow = math.isnan(grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            iteration += 1
            if is_overflow:
                continue

            duration = time.perf_counter() - start
            print(
                f"Train loss {iteration} {reduced_loss:.6f} "
                f"Grad Norm {grad_norm:.6f} {duration:.2f}s/it"
            )

            if iteration % hparams.iters_per_checkpoint == 0:
                val_loss = validate(
                    model, criterion, valset, hparams.batch_size, collate_fn
                )
                print(f"Validation loss {iteration}: {val_loss:9f}")
                if val_loss < best:
                    save_checkpoint(
                        model, optimizer, scaler, learning_rate, iteration, "best"
                    )


if __name__ == "__main__":
    hparams = HParams(batch_size=1)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    main(hparams)
