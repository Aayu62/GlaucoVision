# GlaucoVision: CA-Net for Multistage Glaucoma Classification

An end-to-end implementation and web detection platform reproducing the research paper:
> **CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification Using Fundus Images**  
> *Dipankar Das, Deepak Ranjan Nayak, and Ram Bilas Pachori*  
> **IEEE Transactions on Instrumentation and Measurement**, Vol. 72, 2023.

---

## 🌟 Key Features

1. **Exact Paper Architecture**:
   - **Backbone**: Pretrained DenseNet-121 (ImageNet weights) extracting high-level feature maps $F_{in} \in \mathbb{R}^{7 \times 7 \times 1024}$.
   - **Cascaded Attention Module (CAM)**:
     - **Triplet Channel Attention Block (TCAB)**: Computes 1D Channel, 2D Channel-Width (cross-height), and 2D Channel-Height (cross-width) attention weights with learnable scalar parameters $w_1, w_2, w_3$.
     - **Spatial Attention Block (SAB)**: Focuses on optic nerve head (ONH) and cup-to-disc ratio (CDR) lesion areas using Cross-Channel Average Pooling (CCAP) and $1 \times 1$ convolution.
   - **Classifier**: Global Average Pooling (GAP) + Softmax-activated Fully Connected layer for 3-class classification (**Normal**, **Early-Stage**, **Advanced-Stage**).

2. **Dataset Reconstruction**:
   - Reconstructs the exact **1,582-image LMG Dataset**:
     - **Harvard Dataverse V1**: 1,542 images (786 Normal, 289 Early-Stage, 467 Advanced-Stage)
     - **RIM-ONE r1**: 40 images (14 Normal, 12 Early-Stage, 14 Advanced-Stage)
   - Adheres to the paper's stratified **70:30 (Train:Test)** and inner **70:30 (Train:Val)** split protocol.

3. **Explainability**:
   - Integrated **Grad-CAM** and **Grad-CAM++** to generate visual attention heatmaps localized over the optic cup and neuroretinal rim.

4. **Web-Based Clinical System**:
   - **FastAPI backend** for real-time model inference and heatmap generation.
   - **Modern Glassmorphic Web Dashboard** with live drag-and-drop, sample benchmark cases, stage probabilities, and interactive opacity blending.

---

## 📂 Repository Structure

```
c:/Users/KIIT0001/GlaucoVision/
├── dataset/
│   ├── prepare_lmg.py          # Assembles LMG & Harvard splits (70:30 train/test, 70:30 train/val)
│   └── dataset.py              # PyTorch Dataset with on-the-fly augmentations (flips, rotations, scaling)
├── models/
│   ├── __init__.py
│   ├── tcab.py                 # Triplet Channel Attention Block (3 branches + learnable weights)
│   ├── sab.py                  # Spatial Attention Block (CCAP + 1x1 conv + LayerNorm)
│   ├── cam.py                  # Cascaded Attention Module (TCAB -> SAB)
│   └── ca_net.py               # Complete CA-Net model & baseline ablations
├── explainability/
│   ├── __init__.py
│   └── gradcam.py              # Grad-CAM and Grad-CAM++ implementations
├── app/
│   ├── backend.py              # FastAPI server (inference & heatmap generation)
│   └── static/
│       ├── index.html          # Web application UI
│       ├── style.css           # Glassmorphism & medical theme styling
│       └── app.js              # Interactive frontend logic
├── data_splits/                # Generated CSV manifests (train, val, test)
├── train.py                    # Training pipeline (Adam, initial LR=0.005, ReduceLROnPlateau)
├── evaluate.py                 # Evaluation pipeline (ROC curves, Confusion Matrix, Grad-CAM++)
├── run_app.py                  # Web application launcher
└── test_api.py                 # Programmatic API verification test
```

---

## 🚀 Quick Start Guide

### 1. Launch the Web Application
Start the local FastAPI server:
```bash
python run_app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Click on any benchmark test case (**Normal Control**, **Early Glaucoma**, or **Advanced Glaucoma**) or drag-and-drop any fundus photograph.
- Click **Run CA-Net Analysis** to view the predicted glaucoma stage, confidence breakdown, and Grad-CAM++ lesion heatmaps.

### 2. Dataset Split Generation
To regenerate or inspect the LMG dataset splits:
```bash
python dataset/prepare_lmg.py
```
This generates the CSV manifests under `data_splits/` matching the paper's 1,582 image total.

### 3. Training the Model
Train CA-Net using the paper's exact hyperparameters (Adam optimizer, initial LR=0.005, ReduceLROnPlateau, batch size 16):
```bash
# Train CA-Net on the LMG dataset
python train.py --dataset lmg --backbone densenet121 --attention cam --epochs 50 --batch_size 16 --lr 0.005

# Train the Baseline DenseNet-121 (without attention) for comparison
python train.py --dataset lmg --backbone densenet121 --attention none --epochs 50 --batch_size 16 --lr 0.005
```

### 4. Evaluating Checkpoints
To generate confusion matrices, multi-class ROC curves, and Grad-CAM++ figure overlays:
```bash
python evaluate.py --model_path checkpoints/best_lmg_densenet121_cam.pth --dataset lmg
```
Artifacts are automatically saved in `evaluation_results/`:
- `roc_curves.png` (reproducing Figure 5 of the paper)
- `confusion_matrix.png`
- `gradcam_visualizations.png` (reproducing Figure 6 of the paper)
- `evaluation_metrics.json`
