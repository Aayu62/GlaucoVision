# In-Depth Guide: CA-Net Architecture, Tensor Lifecycle, and Training Mechanics

> **Scope & Purpose**: This document provides an exhaustive, step-by-step breakdown of how **CA-Net** is trained and how tensors transform from raw image pixels into clinical stage classifications. To make abstract matrix algebra intuitive, every mathematical operation is demonstrated using a concrete **toy numerical grid walkthrough**, accompanied by exact line-by-line references to the project source code.

---

## Table of Contents
1. [End-to-End Training Lifecycle Overview](#1-end-to-end-training-lifecycle-overview)
2. [From Raw Image to Numerical Tensor (`dataset/dataset.py`)](#2-from-raw-image-to-numerical-tensor-datasetdatasetpy)
3. [The Forward Pass & Tensor Dimensional Journey](#3-the-forward-pass--tensor-dimensional-journey)
   - [Backbone Feature Extraction (`models/ca_net.py`)](#part-1-backbone-feature-extraction-modelsca_netpy)
   - [Deep Dive into TCAB with Numerical Toy Grid (`models/tcab.py`)](#part-2-deep-dive-into-tcab-with-a-toy-numerical-grid)
   - [Deep Dive into SAB with Numerical Toy Grid (`models/sab.py`)](#part-3-deep-dive-into-sab-with-a-toy-numerical-grid)
   - [Classification Head & Softmax Probabilities](#part-4-the-classification-head)
4. [Loss Calculation & Error Measurement](#4-loss-calculation--error-measurement)
5. [The Backward Pass & Weight Optimization](#5-the-backward-pass--weight-optimization)
   - [Gradient Backpropagation](#gradient-backpropagation)
   - [Adam Optimizer Update Rules](#adam-optimizer-update-rules)
   - [Adaptive Learning Rate Decay (`ReduceLROnPlateau`)](#adaptive-learning-rate-decay-reducelronplateau)
6. [Step-by-Step Code Execution Trace](#6-step-by-step-code-execution-trace)

---

## 1. End-to-End Training Lifecycle Overview

The training engine is implemented in [train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py). The training loop runs for **50 epochs** across **774 training images** and evaluates against **333 validation images** at each epoch.

```
For each Epoch (1 to 50):
  │
  ├─► Training Phase: Iterate through 49 batches of size 16
  │     1. Load batch images (16, 3, 224, 224) & targets (16,)
  │     2. Forward pass through CA-Net ──► Raw logits (16, 3)
  │     3. Compute Categorical Cross-Entropy Loss
  │     4. Backward pass: compute gradients ∇_θ L
  │     5. Adam optimizer updates all parameters & learnable weights w1, w2, w3
  │
  ├─► Validation Phase: Evaluate on 333 unseen validation images
  │     1. Forward pass (torch.no_grad)
  │     2. Compute Validation Accuracy, Macro F1, Precision, Recall, and AUC
  │     3. If Validation F1 > best_val_f1:
  │           Save checkpoint ──► checkpoints/best_lmg_densenet121_cam.pth
  │
  └─► Scheduler Step:
        Pass Validation F1 to ReduceLROnPlateau(patience=5, factor=0.5).
        If F1 does not improve for 5 consecutive epochs, cut learning rate in half.
```

### Key Training Hyperparameters:
* **Batch Size**: `16` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L112))
* **Total Epochs**: `50` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L113))
* **Initial Learning Rate**: `0.005` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L114))
* **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L146))
* **Scheduler**: `ReduceLROnPlateau(mode="max", factor=0.5, patience=5)` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L147))

---

## 2. From Raw Image to Numerical Tensor (`dataset/dataset.py`)

A raw fundus photo stored on disk as a PNG or BMP file is transformed into a floating-point PyTorch tensor inside [dataset/dataset.py](file:///c:/Users/KIIT0001/GlaucoVision/dataset/dataset.py).

```
 Raw Disk Image (e.g. 240 x 240 RGB)
            │
            ▼   [T.Resize((224, 224))]
 Resized Image: (224 x 224 x 3)
            │
            ▼   [Data Augmentation (Train only)]
 Random Horizontal Flip (p=0.5)
 Random Vertical Flip (p=0.5)
 Random Rotation (±20°)
 Random Scaling (0.9x to 1.1x)
            │
            ▼   [T.ToTensor()]
 Float Tensor: Values in range [0.0, 1.0], Shape: (3, 224, 224)
            │
            ▼   [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
 Standardized Tensor: Mean ≈ 0.0, Variance ≈ 1.0
            │
            ▼   [DataLoader batching: batch_size=16]
 Input Mini-Batch Tensor: Shape (16, 3, 224, 224)
```

In code:
* Augmentation pipeline is defined in `get_transforms()` ([dataset/dataset.py](file:///c:/Users/KIIT0001/GlaucoVision/dataset/dataset.py#L20-L36)).
* Loading and transformation occur in `GlaucomaDataset.__getitem__()` ([dataset/dataset.py](file:///c:/Users/KIIT0001/GlaucoVision/dataset/dataset.py#L48-L62)).

---

## 3. The Forward Pass & Tensor Dimensional Journey

Let us trace the complete lifecycle of an input tensor as it moves through the network layers.

### Part 1: Backbone Feature Extraction (`models/ca_net.py`)

When the mini-batch of images $X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ enters `CANet.forward()` ([models/ca_net.py](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L85-L108)):
1. **DenseNet-121 Feature Extractor** ([models/ca_net.py](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L88)):
   * Passes through an initial $7\times 7$ convolution, max pooling, and 4 dense blocks interleaved with transition layers (which downsample via $1\times 1$ conv and $2\times 2$ average pooling).
   * Dense connections concatenate every preceding layer's output to all subsequent layers.
2. **Backbone Output Feature Map ($F_{in}$)**:
   * Shape: **$(B, 1024, 7, 7)$**
   * $B = 16$ (batch size), $K = 1024$ (feature channels), $H = 7$ (height), $W = 7$ (width).
   * A final ReLU non-linearity is applied ([models/ca_net.py](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L90)).

---

### Part 2: Deep Dive into TCAB with a Toy Numerical Grid

To see exactly what happens inside **TCAB** ([models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py)), let us create a **simplified "toy" feature map**:
* Let batch size $B = 1$.
* Let number of channels $K = 2$ (Channel 0 and Channel 1).
* Let spatial resolution be a tiny $2 \times 2$ grid ($H = 2, W = 2$).

Our toy tensor $F_{in}$ of shape **$(1, 2, 2, 2)$** is:

$$\text{Channel 0} = \begin{bmatrix} 2.0 & 4.0 \\ 6.0 & 8.0 \end{bmatrix}, \quad \text{Channel 1} = \begin{bmatrix} 1.0 & 3.0 \\ 5.0 & 7.0 \end{bmatrix}$$

Now let us trace all 3 parallel branches of TCAB:

---

#### Branch 1: 1D Channel-Wise Attention
*Goal: Figure out how much importance to give Channel 0 vs. Channel 1.*
* Code: [models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L58-L63)

1. **Global Average Pooling (GAP) across spatial dimensions $(H, W)$**:
   $$\text{GAP}(x) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x(i, j)$$
   * For Channel 0: $\text{mean}(2.0, 4.0, 6.0, 8.0) = \frac{20.0}{4} = \mathbf{5.0}$
   * For Channel 1: $\text{mean}(1.0, 3.0, 5.0, 7.0) = \frac{16.0}{4} = \mathbf{4.0}$
   * GAP tensor shape: $(1, 2, 1, 1) = [[5.0], [4.0]]$.

2. **Channel Excitation (Two $1\times 1$ Convolutions + LayerNorm + Sigmoid)**:
   * The GAP values pass through a squeeze layer of size $K/r$, LayerNorm, ReLU, an expand layer to size $K$, LayerNorm, and finally a **Sigmoid** activation function:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   * Sigmoid bounds the output between `0.0` and `1.0`. Suppose the learned weights yield attention values:
     $$\text{Attention Weights } W_1 = [0.80, \quad 0.30]$$
     *(Interpretation: Channel 0 contains important optic cup cues $\to$ boost by 0.80; Channel 1 has background noise $\to$ scale down to 0.30).*

3. **Element-Wise Multiplication**:
   $$F_{ch1} = F_{in} \otimes W_1$$
   * For Channel 0 ($0.80 \times \text{Channel 0}$):
     $$F_{ch1}^{(0)} = 0.80 \times \begin{bmatrix} 2.0 & 4.0 \\ 6.0 & 8.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.6 & 3.2 \\ 4.8 & 6.4 \end{bmatrix}}$$
   * For Channel 1 ($0.30 \times \text{Channel 1}$):
     $$F_{ch1}^{(1)} = 0.30 \times \begin{bmatrix} 1.0 & 3.0 \\ 5.0 & 7.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.3 & 0.9 \\ 1.5 & 2.1 \end{bmatrix}}$$
   * $F_{ch1}$ has shape **$(1, 2, 2, 2)$**.

---

#### Branch 2: 2D Channel-Width Cross-Attention
*Goal: Model interactions between feature channels and the horizontal width ($W$) of the optic nerve head.*
* Code: [models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L65-L72)

1. **Permutation along Width Axis**:
   * Original tensor $F_{in}$ has axes $(B, C, H, W)$ with shape $(1, 2, 2, 2)$.
   * `x.permute(0, 2, 1, 3)` swaps the Channel and Height axes $\to (B, H, C, W)$.
   * Now Height ($H$) acts as the depth dimension!

2. **Cross-Height Average Pooling (CHAP) over $H$**:
   $$\tilde{Y} = \text{CHAP}(F'_{in}) = \frac{1}{H} \sum_{h=1}^H F'_{in}(h, c, w)$$
   * For each channel $c$ and column $w$, average across rows $H$:
     * Channel 0, Column 0: $\text{mean}(2.0, 6.0) = \mathbf{4.0}$
     * Channel 0, Column 1: $\text{mean}(4.0, 8.0) = \mathbf{6.0}$
     * Channel 1, Column 0: $\text{mean}(1.0, 5.0) = \mathbf{3.0}$
     * Channel 1, Column 1: $\text{mean}(3.0, 7.0) = \mathbf{5.0}$
   * Shape of $\tilde{Y}$: **$(1, 1, 2, 2)$** (1 depth, spatial dimensions $C \times W$).

3. **$1\times 1$ Conv + LayerNorm + Sigmoid**:
   * Generates a 2D cross-attention map over $(C, W)$, for example:
     $$W_2 = \begin{bmatrix} 0.70 & 0.90 \\ 0.40 & 0.60 \end{bmatrix}$$
   * Multiply element-wise with the permuted tensor, then permute back to $(B, C, H, W)$.
   * $F_{ch2}$ has shape **$(1, 2, 2, 2)$**.

---

#### Branch 3: 2D Channel-Height Cross-Attention
*Goal: Model interactions between feature channels and the vertical height ($H$) of the optic nerve head (critical for detecting vertical cup elongation).*
* Code: [models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L74-L81)

1. **Permutation along Height Axis**:
   * `x.permute(0, 3, 2, 1)` permutes $(B, C, H, W) \to (B, W, H, C)$.
   * Now Width ($W$) acts as the depth dimension.

2. **Cross-Width Average Pooling (CWAP) over $W$**:
   $$\tilde{Z} = \text{CWAP}(F''_{in}) = \frac{1}{W} \sum_{w=1}^W F''_{in}(w, h, c)$$
   * Average across columns $W$ for each height $h$ and channel $c$:
     * Channel 0, Row 0: $\text{mean}(2.0, 4.0) = \mathbf{3.0}$
     * Channel 0, Row 1: $\text{mean}(6.0, 8.0) = \mathbf{7.0}$
     * Channel 1, Row 0: $\text{mean}(1.0, 3.0) = \mathbf{2.0}$
     * Channel 1, Row 1: $\text{mean}(5.0, 7.0) = \mathbf{6.0}$
   * Shape of $\tilde{Z}$: **$(1, 1, 2, 2)$** (spatial dimensions $H \times C$).

3. **$1\times 1$ Conv + LayerNorm + Sigmoid**:
   * Generates 2D cross-attention weights over $(H, C)$, multiplies with permuted tensor, and permutes back to $(B, C, H, W)$.
   * $F_{ch3}$ has shape **$(1, 2, 2, 2)$**.

---

#### TCAB Weighted Aggregation
* Code: [models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L83)

The network merges all three branches using three learnable scalar parameters $w_1, w_2, w_3$:
$$F_{ch} = w_1 F_{ch1} + w_2 F_{ch2} + w_3 F_{ch3}$$
* At initialization, $w_1 = w_2 = w_3 = \frac{1}{3} \approx 0.333$ ([models/tcab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L46-L48)).
* During training, the network automatically learns which dimension (pure channel, cross-width, or cross-height) provides the strongest signal.
* Result: $F_{ch} \in \mathbb{R}^{B \times 1024 \times 7 \times 7}$.

---

### Part 3: Deep Dive into SAB with a Toy Numerical Grid

Now the aggregated channel tensor $F_{ch}$ enters the **Spatial Attention Block (SAB)** ([models/sab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/sab.py)).
* Goal: Create a 2D spatial spotlight over the $(H, W)$ coordinates of the eye, illuminating the optic disc and silencing irrelevant retinal background.

Suppose our toy $F_{ch}$ has shape $(1, 2, 2, 2)$:
$$\text{Channel 0} = \begin{bmatrix} 2.0 & 8.0 \\ 1.0 & 6.0 \end{bmatrix}, \quad \text{Channel 1} = \begin{bmatrix} 4.0 & 6.0 \\ 3.0 & 4.0 \end{bmatrix}$$

1. **Cross-Channel Average Pooling (CCAP) over all $K$ channels**:
   * Code: `v_ccap = torch.mean(x, dim=1, keepdim=True)` ([models/sab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/sab.py#L28))
   * For every coordinate $(i, j)$, calculate the average across all channels:
     $$\tilde{V}(i, j) = \frac{1}{K} \sum_{k=1}^K F_{ch}(k, i, j)$$
     * Top-Left $(0, 0)$: $\text{mean}(2.0, 4.0) = \mathbf{3.0}$
     * Top-Right $(0, 1)$: $\text{mean}(8.0, 6.0) = \mathbf{7.0}$  *(High activation $\to$ Optic Cup!)*
     * Bottom-Left $(1, 0)$: $\text{mean}(1.0, 3.0) = \mathbf{2.0}$
     * Bottom-Right $(1, 1)$: $\text{mean}(6.0, 4.0) = \mathbf{5.0}$
   * Tensor shape: **$(1, 1, 2, 2)$**:
     $$\tilde{V} = \begin{bmatrix} 3.0 & 7.0 \\ 2.0 & 5.0 \end{bmatrix}$$

2. **$1\times 1$ Convolution + LayerNorm + Sigmoid**:
   * Code: `spatial_att = self.sigmoid(self.ln(self.conv(v_ccap)))` ([models/sab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/sab.py#L30))
   * Maps each pooled value to an attention factor in range $[0.0, 1.0]$. For example:
     $$\text{Spatial Attention Map } M_{sp} = \begin{bmatrix} 0.20 & 0.95 \\ 0.15 & 0.65 \end{bmatrix}$$
     *(Notice: Top-Right gets 0.95 spotlight; Top-Left background gets scaled down to 0.20!)*

3. **Element-Wise Spatial Modulation**:
   * Code: `f_sp = x * spatial_att` ([models/sab.py](file:///c:/Users/KIIT0001/GlaucoVision/models/sab.py#L32))
   * Every channel in $F_{ch}$ is multiplied by this 2D stencil:
     $$F_{sp}^{(0)} = \begin{bmatrix} 2.0 \times 0.20 & 8.0 \times 0.95 \\ 1.0 \times 0.15 & 6.0 \times 0.65 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.40 & 7.60 \\ 0.15 & 3.90 \end{bmatrix}}$$
   * The optic cup region ($7.60$) is preserved and amplified, while background noise ($0.40, 0.15$) is suppressed.
   * Output shape: $F_{sp} \in \mathbb{R}^{B \times 1024 \times 7 \times 7}$.

---

### Part 4: The Classification Head

In [models/ca_net.py](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L103-L107):
1. **Global Average Pooling (GAP)**:
   * Averages each of the 1,024 spatial grids of size $7\times 7$ into a single number:
     $$\text{GAP}: (B, 1024, 7, 7) \longrightarrow (B, 1024)$$
2. **Linear Classification Layer**:
   * Multiplies the 1,024-dimensional feature vector by weight matrix $W \in \mathbb{R}^{3 \times 1024}$ and adds bias $b \in \mathbb{R}^3$:
     $$\text{logits} = z = \text{GAP}(F_{sp}) W^T + b \in \mathbb{R}^{B \times 3}$$
3. **Softmax Normalization**:
   * Converts raw logits $z = [z_0, z_1, z_2]$ into valid probabilities that sum to 1:
     $$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=0}^2 e^{z_j}}, \quad c \in \{0: \text{Normal}, 1: \text{Early}, 2: \text{Advanced}\}$$

---

## 4. Loss Calculation & Error Measurement

Let us walk through a concrete example of how training calculates the error penalty on a single image.

### Step 1: True Label vs. Predicted Probabilities
Suppose the current image is an **Early-Stage Glaucoma** patient:
* Ground-truth integer label: $y = 1$
* One-hot target vector:
  $$\mathbf{y} = [0, \quad 1, \quad 0]$$

Suppose CA-Net outputs raw logits $z = [1.2, \quad 2.1, \quad 0.8]$.
Applying Softmax:
$$e^{1.2} \approx 3.32, \quad e^{2.1} \approx 8.17, \quad e^{0.8} \approx 2.23 \implies \sum = 13.72$$
$$\hat{y}_0 (\text{Normal}) = \frac{3.32}{13.72} = \mathbf{0.242} \quad (24.2\%)$$
$$\hat{y}_1 (\text{Early}) = \frac{8.17}{13.72} = \mathbf{0.595} \quad (59.5\%)$$
$$\hat{y}_2 (\text{Advanced}) = \frac{2.23}{13.72} = \mathbf{0.163} \quad (16.3\%)$$

### Step 2: Categorical Cross-Entropy Loss Formula
In [train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L59), `criterion = nn.CrossEntropyLoss()` computes:
$$\mathcal{L} = -\sum_{c=0}^2 y_c \log(\hat{y}_c) = - [0 \times \log(0.242) + 1 \times \log(0.595) + 0 \times \log(0.163)]$$
$$\mathcal{L} = -\log(0.595) \approx \mathbf{0.5192}$$

* If the model had been 99% confident in the correct answer ($\hat{y}_1 = 0.99$), the loss would be tiny: $-\log(0.99) = 0.010$.
* If the model had been totally wrong ($\hat{y}_1 = 0.05$), the loss would be huge: $-\log(0.05) = 2.996$.
* Across a batch of 16 images, the batch loss is the mean of the individual losses:
  $$\mathcal{L}_{\text{batch}} = \frac{1}{16} \sum_{b=1}^{16} \mathcal{L}_b$$

---

## 5. The Backward Pass & Weight Optimization

Once the scalar loss $\mathcal{L}_{\text{batch}}$ is calculated, the training engine updates the network weights.

### Gradient Backpropagation
Code: `loss.backward()` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L60))

Using the chain rule of calculus, PyTorch traces backward through the computation graph:
$$\frac{\partial \mathcal{L}}{\partial \text{Linear Weights}} \longleftarrow \frac{\partial \mathcal{L}}{\partial F_{sp}} \longleftarrow \frac{\partial \mathcal{L}}{\partial \text{SAB Conv}} \longleftarrow \frac{\partial \mathcal{L}}{\partial F_{ch}} \longleftarrow \frac{\partial \mathcal{L}}{\partial (w_1, w_2, w_3)} \longleftarrow \frac{\partial \mathcal{L}}{\partial \text{TCAB Convs}} \longleftarrow \frac{\partial \mathcal{L}}{\partial \text{DenseNet Layers}}$$

Every single parameter $\theta$ receives a gradient:
$$g_t = \nabla_\theta \mathcal{L}$$
This gradient vector points in the direction of greatest error increase; the optimizer steps in the opposite direction to reduce error.

---

### Adam Optimizer Update Rules
Code: `optimizer.step()` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L61))

Standard Gradient Descent uses a fixed step size: $\theta \leftarrow \theta - \eta g_t$.
The Adam optimizer (Equations 11–15 in the paper) maintains two running histories for every single parameter:

1. **First Moment Vector $x_t$ (Momentum)**:
   Keeps a moving average of recent gradient directions so the model does not oscillate wildly:
   $$x_t = \beta_1 x_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9)$$

2. **Second Moment Vector $y_t$ (RMSprop variance)**:
   Keeps a moving average of gradient squared magnitudes to give rarely-updated parameters bigger steps and frequently-updated parameters smaller, stable steps:
   $$y_t = \beta_2 y_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)$$

3. **Bias Correction**:
   Adjusts for the fact that $x_0$ and $y_0$ are initialized at zero:
   $$\hat{x}_t = \frac{x_t}{1 - \beta_1^t}, \quad \hat{y}_t = \frac{y_t}{1 - \beta_2^t}$$

4. **Parameter Update**:
   $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{y}_t} + \epsilon} \hat{x}_t \quad (\eta = 0.005, \epsilon = 10^{-8})$$

This update applies to all 7.09 million weights—including the three scalar attention weights $w_1, w_2, w_3$.

---

### Adaptive Learning Rate Decay (`ReduceLROnPlateau`)
Code: `scheduler.step(val_metrics["f1"])` ([train.py](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L166))

* At the start, the model moves quickly at $\eta = 0.005$.
* As the model converges, taking large steps can cause it to overshoot the optimal weights.
* `ReduceLROnPlateau` monitors the validation F1-score after each epoch.
* If 5 epochs pass without setting a new record score (`patience=5`), it automatically cuts the learning rate in half (`factor=0.5`):
  $$\eta \leftarrow \eta \times 0.5$$
* In your 50-epoch run, the learning rate stepped down from $0.005 \to 0.0025 \to 0.00125 \dots$, enabling the network to settle into its optimal validation accuracy of **79.28%**.

---

## 6. Step-by-Step Code Execution Trace

Here is the exact file-and-line call stack that executes on every training step:

```
[train.py] Line 158: for epoch in range(1, 51):
   │
   ├─► [train.py] Line 162: train_metrics = train_one_epoch(...)
   │      │
   │      ├─► [dataset/dataset.py] Line 48: GlaucomaDataset.__getitem__(idx)
   │      │      ├── Opens fundus PIL Image (Line 55)
   │      │      └── Applies get_transforms(is_train=True): Resize, Flips, Rotations, Norm (Line 59)
   │      │
   │      ├─► [train.py] Line 58: logits = model(images)
   │      │      │
   │      │      ├─► [models/ca_net.py] Line 88: feat = self.features(x)
   │      │      │      └── DenseNet-121 extracts F_in (16, 1024, 7, 7)
   │      │      │
   │      │      ├─► [models/ca_net.py] Line 95: feat = self.attention(feat)
   │      │      │      │
   │      │      │      ├─► [models/cam.py] Line 20: f_ch = self.tcab(x)
   │      │      │      │      ├── Branch 1 GAP + 1x1 Conv (models/tcab.py:59-63)
   │      │      │      │      ├── Branch 2 Permute W + CHAP over H (models/tcab.py:65-72)
   │      │      │      │      ├── Branch 3 Permute H + CWAP over W (models/tcab.py:74-81)
   │      │      │      │      └── Weighted Aggregation w1*F1 + w2*F2 + w3*F3 (models/tcab.py:83)
   │      │      │      │
   │      │      │      └─► [models/cam.py] Line 21: f_sp = self.sab(f_ch)
   │      │      │             ├── Cross-Channel Average Pooling CCAP (models/sab.py:28)
   │      │      │             ├── 1x1 Conv + LayerNorm + Sigmoid (models/sab.py:30)
   │      │      │             └── Spatial Modulation x * spatial_att (models/sab.py:32)
   │      │      │
   │      │      ├─► [models/ca_net.py] Line 104: gap = torch.mean(feat, dim=(2, 3))
   │      │      │      └── Reduces (16, 1024, 7, 7) ──► (16, 1024)
   │      │      │
   │      │      └─► [models/ca_net.py] Line 107: logits = self.classifier(gap)
   │      │             └── Fully connected layer (16, 1024) ──► (16, 3)
   │      │
   │      ├─► [train.py] Line 59: loss = criterion(logits, targets)
   │      │      └── Categorical Cross-Entropy penalty
   │      │
   │      ├─► [train.py] Line 60: loss.backward()
   │      │      └── Computes gradients dL/dw for all 7.09M weights
   │      │
   │      └─► [train.py] Line 61: optimizer.step()
   │             └── Adam updates weights with momentum and variance
   │
   ├─► [train.py] Line 163: val_metrics = evaluate(model, val_loader, ...)
   │      └── Evaluates on 333 validation images without gradient tracking
   │
   ├─► [train.py] Line 166: scheduler.step(val_metrics["f1"])
   │      └── Decays learning rate if validation progress stalls for 5 epochs
   │
   └─► [train.py] Lines 184-194: if val_metrics["f1"] > best_val_f1:
          └── Saves best model weights to checkpoints/best_lmg_densenet121_cam.pth
```

---

### Summary Checklist

| Component | Code Location | Input Shape | Output Shape | What It Does |
| :--- | :--- | :--- | :--- | :--- |
| **Augmentation** | [dataset/dataset.py:20](file:///c:/Users/KIIT0001/GlaucoVision/dataset/dataset.py#L20) | Raw Image | `(3, 224, 224)` | Resizes, rotates, flips, and normalizes pixels |
| **Backbone** | [models/ca_net.py:88](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L88) | `(16, 3, 224, 224)` | `(16, 1024, 7, 7)` | Extracts 1,024 dense hierarchical feature maps |
| **TCAB** | [models/tcab.py:51](file:///c:/Users/KIIT0001/GlaucoVision/models/tcab.py#L51) | `(16, 1024, 7, 7)` | `(16, 1024, 7, 7)` | Models 1D channel, channel-width, and channel-height interactions |
| **SAB** | [models/sab.py:22](file:///c:/Users/KIIT0001/GlaucoVision/models/sab.py#L22) | `(16, 1024, 7, 7)` | `(16, 1024, 7, 7)` | Spotlights the optic nerve disc; suppresses background vessels |
| **GAP + Classifier** | [models/ca_net.py:104](file:///c:/Users/KIIT0001/GlaucoVision/models/ca_net.py#L104) | `(16, 1024, 7, 7)` | `(16, 3)` | Averages spatial grid and outputs 3 class logits |
| **Loss** | [train.py:59](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L59) | `(16, 3)` & targets | Scalar $\mathcal{L}$ | Measures penalty via Categorical Cross-Entropy |
| **Optimizer** | [train.py:61](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L61) | Gradients $\nabla \mathcal{L}$ | Updated Weights | Adjusts parameters using Adam momentum and scaling |
| **Scheduler** | [train.py:166](file:///c:/Users/KIIT0001/GlaucoVision/train.py#L166) | Validation F1 | Learning Rate $\eta$ | Halves learning rate when progress plateaus |

