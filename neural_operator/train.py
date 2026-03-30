"""Training script for FNO classification models.

Usage:
    python -m neural_operator.train --config neural_operator/config.json --model fno --data-dir data/train
    python -m neural_operator.train --config neural_operator/config.json --model hourglass --epochs 200 --lr 1e-3
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from .models import NaiveFNO, FNOImageClassifier, HourglassFNO
from .losses import FocalLoss
from .dataloader import get_dataloaders
from .metrics import evaluate

MODEL_REGISTRY = {
    "NaiveFNO": NaiveFNO,
    "FNOImageClassifier": FNOImageClassifier,
    "HourglassFNO": HourglassFNO,
}


def build_model(model_config):
    """Instantiate a model from its config dict."""
    cls_name = model_config["class"]
    cls = MODEL_REGISTRY[cls_name]

    # Filter to only valid constructor args
    import inspect
    valid_params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in model_config.items()
              if k in valid_params and k not in ("self", "class", "loss")}
    return cls(**kwargs)


def build_criterion(model_config):
    """Build loss function from config."""
    loss_type = model_config.get("loss", "cross_entropy")
    if loss_type == "focal":
        return FocalLoss(gamma=2.0)
    return nn.CrossEntropyLoss()


def build_scheduler(optimizer, training_config):
    """Build LR scheduler from config."""
    sched_type = training_config.get("scheduler", "StepLR")
    params = training_config.get("scheduler_params", {})

    if sched_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.get("T_max", training_config["num_epochs"]),
            eta_min=params.get("eta_min", 1e-6)
        )
    else:  # StepLR
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 10),
            gamma=params.get("gamma", 0.75)
        )


def train(args):
    # Load config
    with open(args.config) as f:
        config = json.load(f)

    data_config = config["data"]
    training_config = config["training"]
    model_config = config["models"][args.model]

    # CLI overrides
    if args.epochs:
        training_config["num_epochs"] = args.epochs
    if args.lr:
        training_config["lr"] = args.lr
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.data_dir:
        data_config["train_dir"] = args.data_dir
    if args.val_dir:
        data_config["val_dir"] = args.val_dir

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Data
    loaders, class_names = get_dataloaders(
        train_dir=data_config["train_dir"],
        image_size=data_config["image_size"],
        batch_size=training_config["batch_size"],
        val_dir=data_config.get("val_dir"),
        train_split=data_config.get("train_split", 0.9),
        num_workers=data_config.get("num_workers", 2),
        seed=data_config.get("seed", 42),
    )
    print(f"Classes: {class_names}")
    print(f"Train: {len(loaders['train'].dataset)}, Test: {len(loaders['test'].dataset)}")
    if 'val' in loaders:
        print(f"Val: {len(loaders['val'].dataset)}")

    # Model
    model = build_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config['class']} ({total_params:,} params)")

    # Resume from checkpoint
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    # Optimizer, scheduler, criterion
    optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"],
                            weight_decay=training_config.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, training_config)
    criterion = build_criterion(model_config)

    # wandb (optional)
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={
            **training_config, **model_config,
            "model_name": args.model, "data": data_config,
        })

    # Checkpoint tracking
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    top_k = training_config.get("top_k_checkpoints", 2)
    top_checkpoints = []  # list of (val_f1, epoch, filepath)

    num_epochs = training_config["num_epochs"]

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Evaluate test ---
        test_loss, test_acc, test_preds, test_labels, _ = evaluate(
            model, loaders['test'], criterion, device)

        # --- Evaluate val (if available) ---
        eval_loader_key = 'val' if 'val' in loaders else 'test'
        if eval_loader_key == 'val':
            val_loss, val_acc, val_preds, val_labels, _ = evaluate(
                model, loaders['val'], criterion, device)
        else:
            val_loss, val_acc, val_preds, val_labels = test_loss, test_acc, test_preds, test_labels

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

        # --- Logging ---
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/accuracy": train_acc,
            "test/loss": test_loss, "test/accuracy": test_acc, "test/f1_macro": test_f1,
            "val/loss": val_loss, "val/accuracy": val_acc, "val/f1_macro": val_f1,
            "lr": current_lr,
        }

        # --- Top-K checkpoint management ---
        min_f1 = top_checkpoints[-1][0] if len(top_checkpoints) == top_k else -1.0
        if val_f1 > min_f1:
            ckpt_path = os.path.join(save_dir, f"fno_epoch{epoch+1}_f1{val_f1:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)

            top_checkpoints.append((val_f1, epoch + 1, ckpt_path))
            top_checkpoints.sort(key=lambda x: x[0], reverse=True)

            if len(top_checkpoints) > top_k:
                _, removed_epoch, removed_path = top_checkpoints.pop()
                if os.path.exists(removed_path):
                    os.remove(removed_path)

            print(f"  [Checkpoint] Epoch {epoch+1}: val F1 = {val_f1:.4f}")

        # --- Per-class metrics every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            for split, preds, labels_arr in [("test", test_preds, test_labels), ("val", val_preds, val_labels)]:
                prec = precision_score(labels_arr, preds, average="macro", zero_division=0)
                rec = recall_score(labels_arr, preds, average="macro", zero_division=0)
                log_dict[f"{split}/precision_macro"] = prec
                log_dict[f"{split}/recall_macro"] = rec

                f1_per = f1_score(labels_arr, preds, average=None, zero_division=0)
                for i, cname in enumerate(class_names):
                    log_dict[f"{split}/f1_{cname}"] = f1_per[i]

        if wandb_run:
            import wandb
            wandb.log(log_dict)

        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test F1: {test_f1:.4f} | Val F1: {val_f1:.4f}")

    # --- Summary ---
    if top_checkpoints:
        best_f1, best_epoch, best_path = top_checkpoints[0]
        print(f"\nTraining complete. Best val F1: {best_f1:.4f} (epoch {best_epoch})")
        print(f"Top-{top_k} checkpoints:")
        for f1_val, ep, path in top_checkpoints:
            print(f"  Epoch {ep}: F1={f1_val:.4f} -> {path}")

    if wandb_run:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train FNO classification models")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--model", required=True, choices=["naive_fno", "fno", "fno_higher_modes", "hourglass"],
                        help="Model name from config")
    parser.add_argument("--data-dir", type=str, default=None, help="Override train data directory")
    parser.add_argument("--val-dir", type=str, default=None, help="Override val data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
