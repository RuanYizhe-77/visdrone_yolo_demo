from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics or {},
        },
        path,
    )


def load_model_weights(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt

