"""
Visual Explainability for CA-Net: Grad-CAM and Grad-CAM++
From:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV 2017
- Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks", WACV 2018
- CA-Net paper Section V-E: "Visualization Results using Grad-CAM and Grad-CAM++"
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

class CAMExplainer:
    def __init__(self, model):
        """
        Args:
            model: Instance of CANet
        """
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=None, method="gradcam++"):
        """
        Generate visual attention heatmap using Grad-CAM or Grad-CAM++.
        Args:
            input_tensor (torch.Tensor): Shape (1, 3, H, W)
            target_class (int or None): Target class index. If None, uses argmax prediction.
            method (str): 'gradcam' or 'gradcam++'
        Returns:
            heatmap (np.ndarray): Shape (H, W) in range [0, 1]
            predicted_class (int)
            probabilities (list of float)
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        if target_class is None:
            target_class = int(np.argmax(probs))

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.model.get_cam_gradients()   # (1, C, H_feat, W_feat)
        activations = self.model.get_cam_activations() # (1, C, H_feat, W_feat)

        if gradients is None or activations is None:
            raise RuntimeError("Activations or gradients not captured by the model hook.")

        grads = gradients.detach().cpu().numpy()[0]     # (C, H_feat, W_feat)
        acts = activations.detach().cpu().numpy()[0]     # (C, H_feat, W_feat)

        method = method.lower()
        if method == "gradcam":
            # Global Average Pooling of gradients
            weights = np.mean(grads, axis=(1, 2))  # (C,)
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]
            cam = np.maximum(cam, 0) # ReLU

        elif method == "gradcam++":
            # Grad-CAM++: alpha weights computed with positive gradients & scaling
            grads_power_2 = grads ** 2
            grads_power_3 = grads ** 3
            sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)
            eps = 1e-7

            # Denominator: 2 * (grad^2) + sum(acts * grad^3)
            aij = grads_power_2 / (2.0 * grads_power_2 + sum_acts * grads_power_3 + eps)
            # Only consider positive gradients
            aij = np.where(grads != 0, aij, 0)

            weights = np.maximum(grads, 0) * aij
            alpha_c = np.sum(weights, axis=(1, 2)) # (C,)

            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(alpha_c):
                cam += w * acts[i]
            cam = np.maximum(cam, 0) # ReLU
        else:
            raise ValueError(f"Unknown explainability method: {method}")

        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to input resolution (H, W)
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized, target_class, probs.tolist()

    @staticmethod
    def overlay_heatmap_on_image(rgb_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays a normalized heatmap [0, 1] onto an RGB uint8 image [0, 255].
        Returns:
            overlay (np.ndarray): uint8 RGB image of shape (H, W, 3)
        """
        # Ensure image is in uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(colored_heatmap, alpha, rgb_image, 1 - alpha, 0)
        return overlay
