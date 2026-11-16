"""
Evaluation, ROC Analysis, and Explainability Visualization for CA-Net
Recreates the evaluation plots from the paper:
- Multi-class ROC curves with AUC (Figures 4 and 5)
- Confusion Matrix
- Per-class Precision, Recall, F1
- Grad-CAM and Grad-CAM++ visualizations (Figure 6)
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
import torch.nn.functional as F

from models.ca_net import CANet
from dataset.dataset import build_dataloaders
from explainability.gradcam import CAMExplainer

CLASS_NAMES = ["Normal", "Early-stage", "Advanced-stage"]

def evaluate_and_plot(model_path,
                      dataset_type="lmg",
                      output_dir="evaluation_results",
                      device=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    backbone = config.get("backbone", "densenet121")
    attention = config.get("attention", "cam")

    print(f"Loading model checkpoint: {model_path}")
    print(f"Config: Backbone={backbone}, Attention={attention}, Dataset={dataset_type}")

    model = CANet(backbone=backbone, attention=attention, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(dataset_type=dataset_type, batch_size=16)

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Classification Report & Confusion Matrix
    print("\n=== Classification Report ===")
    report_dict = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    print(report_str)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({dataset_type.upper()})")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Multi-Class ROC Curves (Reproducing Fig. 4 / Fig. 5 of the paper)
    plt.figure(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(len(CLASS_NAMES)):
        y_binary = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves - {backbone.upper()} with {attention.upper()} ({dataset_type.upper()})", fontsize=13)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # Save metrics JSON
    metrics_summary = {
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"\nSaved confusion matrix to: {cm_path}")
    print(f"Saved ROC curves to: {roc_path}")

    # 3. Generate Grad-CAM and Grad-CAM++ visualizations for sample test cases (Fig. 6)
    print("\nGenerating Grad-CAM & Grad-CAM++ sample visualizations...")
    explainer = CAMExplainer(model)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    # Pick 1 sample for each class
    sample_indices = []
    for cls_idx in range(3):
        idxs = np.where(all_targets == cls_idx)[0]
        if len(idxs) > 0:
            sample_indices.append(idxs[0])

    for row, idx in enumerate(sample_indices):
        img_tensor, target = test_loader.dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Grad-CAM
        hm_gcam, _, _ = explainer.generate_heatmap(input_tensor, target_class=target, method="gradcam")
        # Grad-CAM++
        hm_gcampp, _, _ = explainer.generate_heatmap(input_tensor, target_class=target, method="gradcam++")

        # Denormalize image for display
        raw_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = np.clip(raw_img * std + mean, 0, 1)

        overlay_gcam = CAMExplainer.overlay_heatmap_on_image(rgb, hm_gcam)
        overlay_gcampp = CAMExplainer.overlay_heatmap_on_image(rgb, hm_gcampp)

        # Plot row
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Original ({CLASS_NAMES[target]})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hm_gcam, cmap="jet")
        axes[row, 1].set_title("Grad-CAM Heatmap")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_gcam)
        axes[row, 2].set_title("Grad-CAM Overlay")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay_gcampp)
        axes[row, 3].set_title("Grad-CAM++ Overlay")
        axes[row, 3].axis("off")

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "gradcam_visualizations.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM/Grad-CAM++ visualizations to: {viz_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CA-Net checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--dataset", type=str, default="lmg", choices=["lmg", "harvard"])
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    evaluate_and_plot(args.model_path, dataset_type=args.dataset, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
