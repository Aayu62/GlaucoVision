"""
Training Pipeline for CA-Net
Implements the training protocol described in Section V-A of:
"CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification"
(IEEE Transactions on Instrumentation and Measurement, 2023)

Key settings:
- Initial LR: 0.005
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
- Loss: Cross-Entropy
- Optimizer: Adam
- Batch size: 16
- Epochs: 50 (configurable)
- Metrics: Accuracy, Precision, Recall, F1, multi-class AUC
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.ca_net import CANet
from dataset.dataset import build_dataloaders

def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    except Exception:
        auc = 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc)
    }

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train CA-Net for Multistage Glaucoma Classification")
    parser.add_argument("--dataset", type=str, default="lmg", choices=["lmg", "harvard"],
                        help="Dataset to train on (lmg or harvard)")
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=["densenet121", "resnet50", "vgg19", "mobilenet_v2"])
    parser.add_argument("--attention", type=str, default="cam",
                        choices=["cam", "tcab", "sab", "none"],
                        help="'cam' is CA-Net, 'none' is Baseline without attention")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Starting CA-Net Training Pipeline ===")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()} | Backbone: {args.backbone} | Attention: {args.attention.upper()}")
    print(f"Batch size: {args.batch_size} | Initial LR: {args.lr} | Epochs: {args.epochs}")

    os.makedirs(args.save_dir, exist_ok=True)
    exp_name = f"{args.dataset}_{args.backbone}_{args.attention}"
    best_model_path = os.path.join(args.save_dir, f"best_{exp_name}.pth")

    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size
    )
    print(f"Loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Model
    model = CANet(
        backbone=args.backbone,
        num_classes=3,
        pretrained=True,
        attention=args.attention
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_decay, patience=args.patience, min_lr=1e-6)

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": [],
        "lr": []
    }

    best_val_f1 = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Scheduler step based on validation F1-score / accuracy
        scheduler.step(val_metrics["f1"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
              f"F1: {val_metrics['f1']*100:.2f}% AUC: {val_metrics['auc']:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(args)
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val F1: {best_val_f1*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")

    # Save training history
    history_file = os.path.join(args.save_dir, f"history_{exp_name}.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved history log to {history_file}")

    # Evaluate best model on Test Set
    print("\n=== Evaluating Best Checkpoint on Unseen Test Set ===")
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Set Results -> Acc: {test_metrics['accuracy']*100:.2f}%, "
          f"Prec: {test_metrics['precision']*100:.2f}%, "
          f"Recall: {test_metrics['recall']*100:.2f}%, "
          f"F1: {test_metrics['f1']*100:.2f}%, "
          f"AUC: {test_metrics['auc']:.4f}")

    test_res_file = os.path.join(args.save_dir, f"test_results_{exp_name}.json")
    with open(test_res_file, "w") as f:
        json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    main()
