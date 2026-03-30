"""Validation script for FNO classification models.

Usage:
    python -m neural_operator.val --config neural_operator/config.json --model fno --weights weights/fno_ce_light_f1_9555.pth
    python -m neural_operator.val --config neural_operator/config.json --model fno --weights weights/fno_ce_light_f1_9555.pth --splits train test val --plot-roc --plot-cm
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from .models import NaiveFNO, FNOImageClassifier, HourglassFNO
from .dataloader import get_dataloaders
from .metrics import (
    evaluate, compute_metrics, compute_roc_auc,
    plot_roc_curves, plot_confusion_matrix, print_metrics,
)

MODEL_REGISTRY = {
    "NaiveFNO": NaiveFNO,
    "FNOImageClassifier": FNOImageClassifier,
    "HourglassFNO": HourglassFNO,
}


def build_model(model_config):
    """Instantiate a model from its config dict."""
    import inspect
    cls_name = model_config["class"]
    cls = MODEL_REGISTRY[cls_name]
    valid_params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in model_config.items()
              if k in valid_params and k not in ("self", "class", "loss")}
    return cls(**kwargs)


def run_val(args):
    with open(args.config) as f:
        config = json.load(f)

    data_config = config["data"]
    model_config = config["models"][args.model]

    if args.data_dir:
        data_config["train_dir"] = args.data_dir
    if args.val_dir:
        data_config["val_dir"] = args.val_dir

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Build model and load weights
    model = build_model(model_config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config['class']} ({total_params:,} params)")
    print(f"Weights: {args.weights}")

    # Data
    batch_size = config["training"].get("batch_size", 32)
    loaders, class_names = get_dataloaders(
        train_dir=data_config["train_dir"],
        image_size=data_config["image_size"],
        batch_size=batch_size,
        val_dir=data_config.get("val_dir"),
        train_split=data_config.get("train_split", 0.9),
        num_workers=data_config.get("num_workers", 2),
        seed=data_config.get("seed", 42),
    )

    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.output_dir, exist_ok=True)

    # Map split names to loader keys
    split_loader_map = {
        'train': 'train_eval',
        'test': 'test',
        'val': 'val',
    }

    for split in args.splits:
        loader_key = split_loader_map.get(split, split)
        if loader_key not in loaders:
            print(f"\nSkipping '{split}' — no dataloader available (provide --val-dir for val split)")
            continue

        print(f"\n{'#'*60}")
        print(f"# Evaluating on {split.upper()}")
        print(f"{'#'*60}")

        loss, acc, preds, labels, probs = evaluate(model, loaders[loader_key], criterion, device)
        metrics = compute_metrics(preds, labels, class_names)
        print_metrics(metrics, split.upper(), class_names)

        # ROC-AUC
        fpr, tpr, roc_auc = compute_roc_auc(labels, probs, class_names)
        print(f"\nAUC scores:")
        print(f"  Macro AUC: {roc_auc['macro']:.4f}")
        print(f"  Micro AUC: {roc_auc['micro']:.4f}")
        for i, cname in enumerate(class_names):
            print(f"  {cname}: {roc_auc[i]:.4f}")

        if args.plot_roc:
            save_path = os.path.join(args.output_dir, f"roc_{args.model}_{split}.png")
            plot_roc_curves(fpr, tpr, roc_auc, class_names,
                            title=f"{model_config['class']} - {split.upper()} ROC Curves",
                            save_path=save_path)

        if args.plot_cm:
            save_path = os.path.join(args.output_dir, f"cm_{args.model}_{split}.png")
            plot_confusion_matrix(preds, labels, class_names,
                                  title=f"{model_config['class']} - {split.upper()} Confusion Matrix",
                                  save_path=save_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FNO classification models")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--model", required=True, choices=["naive_fno", "fno", "fno_higher_modes", "hourglass"],
                        help="Model name from config")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override train data directory")
    parser.add_argument("--val-dir", type=str, default=None, help="Override val data directory")
    parser.add_argument("--splits", nargs='+', default=["val"],
                        choices=["train", "test", "val"], help="Splits to evaluate")
    parser.add_argument("--plot-roc", action="store_true", help="Save ROC curve plots")
    parser.add_argument("--plot-cm", action="store_true", help="Save confusion matrix plots")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for plots")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    run_val(args)


if __name__ == "__main__":
    main()
