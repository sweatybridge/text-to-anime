import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import (
    HParams,
    TextLandmarkCollate,
    TextLandmarkLoader,
    TextLandmarkLoss,
    TextLandmarkModel,
)


def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler):
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    scaler.load_state_dict(checkpoint_dict["scaler"])
    scheduler.load_state_dict(checkpoint_dict["scheduler"])
    iteration = checkpoint_dict["iteration"]
    val_loss = checkpoint_dict["val_loss"]
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return val_loss, iteration


def save_checkpoint(model, optimizer, scaler, scheduler, iteration, val_loss, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save(
        {
            "val_loss": val_loss,
            "iteration": iteration,
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
        },
        filepath,
    )


def validate(model, criterion, valset, batch_size, collate_fn):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(
            dataset=valset,
            sampler=None,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        val_loss = 0.0
        steps = len(val_loader)
        for batch in val_loader:
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss /= steps

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

    # Initialise model with pretrained weights and freeze
    model = TextLandmarkModel(hparams)
    if torch.cuda.is_available():
        model = model.cuda()

    if hparams.fp16_run:
        model.decoder.xyz.attention_layer.score_mask_value = np.finfo("float16").min
        model.decoder.mel.attention_layer.score_mask_value = np.finfo("float16").min

    # Setup data loaders
    trainset = TextLandmarkLoader()
    valset = TextLandmarkLoader(train=False)
    collate_fn = TextLandmarkCollate()
    train_loader = DataLoader(
        dataset=trainset,
        num_workers=1,
        shuffle=True,
        sampler=None,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Load checkpoint if one exists
    best = 100
    iteration = 0
    epoch_offset = 0
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=hparams.epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=hparams.fp16_run)
    criterion = TextLandmarkLoss()

    if checkpoint_path is not None:
        best, iteration = load_checkpoint(
            checkpoint_path, model, optimizer, scaler, scheduler
        )
        epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    val_loss_arr = []
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")
        for batch in train_loader:
            start = time.perf_counter()
            model.zero_grad()
            with torch.cuda.amp.autocast():
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss = criterion(y_pred, y)

            reduced_loss = loss.item()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=hparams.grad_clip_thresh,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            iteration += 1
            duration = time.perf_counter() - start
            print(
                f"Train loss {iteration} {reduced_loss:.6f} "
                f"Grad Norm {grad_norm:.6f} {duration:.2f}s/it"
            )

            if iteration % hparams.iters_per_checkpoint == 0:
                val_loss = validate(
                    model, criterion, valset, hparams.batch_size, collate_fn
                )
                val_loss_arr.append(val_loss)
                print(f"Validation loss {iteration}: {val_loss:9f}")
                if val_loss < best and not math.isnan(grad_norm):
                    save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        iteration,
                        val_loss,
                        "best.pt",
                    )
                    best = val_loss
    print("validation loss:")
    print(val_loss_arr)


if __name__ == "__main__":
    hparams = HParams(
        n_landmark_xyz=60,
        # max_decoder_steps=240,
        # epochs=50,
        iters_per_checkpoint=50,
        learning_rate=2e-3,
        batch_size=8,
        fp16_run=True,
    )

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    main(hparams)
