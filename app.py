"""
GlaucoVision - CA-Net Multistage Glaucoma Diagnosis & Explainability
Hugging Face Spaces Native Gradio Application (100% Free Tier Supported)
Paper: IEEE Transactions on Instrumentation and Measurement, Vol. 72, 2023
"""

# 1. Hugging Face ZeroGPU requires importing spaces before any other CUDA package
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import gradio as gr

from models.ca_net import CANet
from dataset.dataset import get_transforms
from explainability.gradcam import CAMExplainer

# Path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_lmg_densenet121_cam.pth")

CLASS_NAMES = ["Normal", "Early-stage", "Advanced-stage"]
CLASS_DESCRIPTIONS = {
    "Normal": "**Normal Control**: No observable glaucomatous neuroretinal rim thinning or optic cup excavation. CDR is within normal physiological limits.",
    "Early-stage": "**Early-Stage Glaucoma**: Subtle structural alterations detected in the optic nerve head, including mild neuroretinal rim loss or vertical cup enlargement.",
    "Advanced-stage": "**Advanced-Stage Glaucoma**: Severe optic nerve head excavation, significant neuroretinal rim thinning, and extensive retinal nerve fiber layer defect patterns."
}

# 2. Initialize model on CPU on startup (ZeroGPU will move to CUDA inside the function)
model = CANet(backbone="densenet121", attention="cam", pretrained=True)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading trained weights from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
else:
    print("Warning: Trained weights checkpoint not found. Operating with pretrained ImageNet weights.")

model.eval()
explainer = CAMExplainer(model)
transform = get_transforms(is_train=False, img_size=224)

@gpu_decorator
def diagnose_fundus(input_image):
    if input_image is None:
        return None, None, None, "Please upload or select an eye fundus image."

    # Determine execution device dynamically (CUDA on ZeroGPU, CPU otherwise)
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(curr_device)

    # Convert to PIL RGB
    if isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(input_image).convert("RGB")
    else:
        pil_img = input_image.convert("RGB")

    orig_resized = np.array(pil_img.resize((224, 224)))
    input_tensor = transform(pil_img).unsqueeze(0).to(curr_device)

    # 1. Forward Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_stage = CLASS_NAMES[pred_idx]

    # Format dictionary for gr.Label
    probabilities = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
    }

    # 2. Grad-CAM++ (Paper Primary)
    hm_gcampp, _, _ = explainer.generate_heatmap(input_tensor, target_class=pred_idx, method="gradcam++")
    overlay_gcampp = CAMExplainer.overlay_heatmap_on_image(orig_resized, hm_gcampp, alpha=0.55)

    # 3. Standard Grad-CAM
    hm_gcam, _, _ = explainer.generate_heatmap(input_tensor, target_class=pred_idx, method="gradcam")
    overlay_gcam = CAMExplainer.overlay_heatmap_on_image(orig_resized, hm_gcam, alpha=0.55)

    # Description text
    desc_markdown = f"""
### Diagnostic Verdict: **{pred_stage}** (Confidence: {probs[pred_idx]*100:.1f}%)

{CLASS_DESCRIPTIONS[pred_stage]}

* **Input Resolution**: 224×224 pixels
* **Architecture**: DenseNet-121 + Cascaded Attention Module (TCAB + SAB)
* **Heatmap Focus**: Grad-CAM++ concentrates thermal intensity directly on the optic cup and neuroretinal rim margin.
"""

    return probabilities, overlay_gcampp, overlay_gcam, desc_markdown

# Sample images for testing
sample_dir = os.path.join(BASE_DIR, "app", "static", "samples")
sample_images = []
for name in ["normal_sample.png", "early_sample.png", "advanced_sample.png"]:
    sp = os.path.join(sample_dir, name)
    if os.path.exists(sp):
        sample_images.append([sp])

# Build Gradio Interface
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    padding: 20px 0;
}
.disclaimer-box {
    margin-top: 25px;
    padding: 12px 18px;
    border-radius: 8px;
    border: 1px solid #ff4d4f;
    background-color: rgba(255, 77, 79, 0.08);
    font-size: 0.85em;
}
"""

with gr.Blocks(title="GlaucoVision: CA-Net Glaucoma Diagnosis") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1 style="font-size: 2.3em; margin-bottom: 6px;">GlaucoVision: CA-Net Diagnostic Platform</h1>
        <p style="font-size: 1.1em; color: #888;">
            Deep learning multistage glaucoma detection in retinal fundus photography with Grad-CAM++ visual explainability.<br>
            <em>Implementation of IEEE Transactions on Instrumentation and Measurement (2023)</em>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 1. Upload Retinal Fundus Photo")
            input_img = gr.Image(type="pil", label="Input Fundus Image")
            btn_analyze = gr.Button("Run CA-Net Analysis", variant="primary", size="lg")

            if sample_images:
                gr.Markdown("#### Or click a benchmark clinical test case:")
                gr.Examples(
                    examples=sample_images,
                    inputs=input_img,
                    label="Benchmark Test Cases"
                )

        with gr.Column(scale=6):
            gr.Markdown("### 2. Multi-Stage Diagnosis & Confidence")
            output_label = gr.Label(num_top_classes=3, label="Stage Probability Distribution")
            output_desc = gr.Markdown()

            gr.Markdown("### 3. Visual Explainability (Lesion Attention Heatmaps)")
            with gr.Row():
                output_gcampp = gr.Image(label="Grad-CAM++ (Paper Primary - Neuroretinal Rim Focus)")
                output_gcam = gr.Image(label="Standard Grad-CAM Overlay")

    gr.HTML("""
    <div class="disclaimer-box">
        <strong>Medical Research Disclaimer:</strong> This software is an implementation of academic research (Das et al., IEEE TIM 2023) developed strictly for educational and investigative purposes. It does not provide certified medical diagnoses and must not substitute for an in-person examination by a qualified ophthalmologist.
    </div>
    """)

    btn_analyze.click(
        fn=diagnose_fundus,
        inputs=input_img,
        outputs=[output_label, output_gcampp, output_gcam, output_desc]
    )

if __name__ == "__main__":
    demo.launch(css=custom_css)
