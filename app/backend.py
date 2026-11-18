"""
FastAPI Backend Server for CA-Net Glaucoma Classification & Explainability
Exposes:
- POST /api/predict: Predicts glaucoma stage (Normal, Early, Advanced) + confidence scores
- POST /api/explain: Generates Grad-CAM and Grad-CAM++ attention heatmaps & overlays
- GET /api/sample-images: Serves sample test images for quick demonstration
- GET /api/status: Returns model architecture and hardware details
- Serves static web frontend at root /
"""

import os
import io
import base64
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from models.ca_net import CANet
from dataset.dataset import get_transforms
from explainability.gradcam import CAMExplainer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
STATIC_DIR = os.path.join(APP_DIR, "static")

CLASS_NAMES = ["Normal", "Early-stage", "Advanced-stage"]
CLASS_DESCRIPTIONS = {
    "Normal": "No observable glaucomatous optic nerve head cupping or neuroretinal rim thinning. Physiological CDR is within normal range.",
    "Early-stage": "Early structural changes detected in the optic nerve head, including mild neuroretinal rim loss or subtle vertical cup enlargement.",
    "Advanced-stage": "Severe optic disc cupping, significant neuroretinal rim thinning, and extensive retinal nerve fiber layer (RNFL) defect patterns."
}

app = FastAPI(title="CA-Net Glaucoma Diagnosis System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & explainer cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
explainer = None
transform = get_transforms(is_train=False, img_size=224)

def load_or_init_model():
    global model, explainer
    best_weights_path = os.path.join(CHECKPOINTS_DIR, "best_lmg_densenet121_cam.pth")
    
    # Instantiate CA-Net
    model = CANet(backbone="densenet121", attention="cam", pretrained=True).to(device)
    
    if os.path.exists(best_weights_path):
        print(f"Loading trained weights from {best_weights_path}...")
        ckpt = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Trained weights checkpoint not found yet. Operating in feature-inference demo mode with ImageNet pretrained DenseNet-121 backbone.")
        
    model.eval()
    explainer = CAMExplainer(model)

# Initialize on startup
load_or_init_model()

def pil_to_base64_data_url(image_np_or_pil, format="PNG"):
    if isinstance(image_np_or_pil, np.ndarray):
        pil_img = Image.fromarray(image_np_or_pil)
    else:
        pil_img = image_np_or_pil
        
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

@app.get("/api/status")
async def get_status():
    w1, w2, w3 = 0.33, 0.33, 0.33
    if hasattr(model, "attention") and hasattr(model.attention, "tcab"):
        w1 = float(model.attention.tcab.w1.item())
        w2 = float(model.attention.tcab.w2.item())
        w3 = float(model.attention.tcab.w3.item())

    return {
        "status": "online",
        "device": str(device),
        "backbone": "DenseNet-121",
        "attention": "Cascaded Attention Module (TCAB + SAB)",
        "input_resolution": "224x224",
        "tcab_weights": {"w1_channel": round(w1, 3), "w2_cross_width": round(w2, 3), "w3_cross_height": round(w3, 3)},
        "classes": CLASS_NAMES
    }

@app.post("/api/predict")
async def predict_stage(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    start_t = time.time()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    latency_ms = round((time.time() - start_t) * 1000, 1)
    pred_class_idx = int(np.argmax(probs))
    pred_class_name = CLASS_NAMES[pred_class_idx]

    return {
        "prediction": pred_class_name,
        "class_index": pred_class_idx,
        "confidence": round(float(probs[pred_class_idx]) * 100, 2),
        "description": CLASS_DESCRIPTIONS[pred_class_name],
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "latency_ms": latency_ms
    }

@app.post("/api/explain")
async def explain_prediction(file: UploadFile = File(...), target_class: Optional[int] = Form(None)):
    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Prepare input tensor
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    orig_resized = np.array(pil_img.resize((224, 224)))

    # 1. Grad-CAM++ (Paper primary)
    hm_gcampp, pred_idx, probs = explainer.generate_heatmap(input_tensor, target_class=target_class, method="gradcam++")
    overlay_gcampp = CAMExplainer.overlay_heatmap_on_image(orig_resized, hm_gcampp, alpha=0.55)

    # 2. Grad-CAM (Standard comparison)
    hm_gcam, _, _ = explainer.generate_heatmap(input_tensor, target_class=pred_idx, method="gradcam")
    overlay_gcam = CAMExplainer.overlay_heatmap_on_image(orig_resized, hm_gcam, alpha=0.55)

    # Convert to Data URLs
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "class_index": pred_idx,
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))
        },
        "gradcam_pp_overlay": pil_to_base64_data_url(overlay_gcampp),
        "gradcam_pp_heatmap": pil_to_base64_data_url(np.uint8(hm_gcampp * 255)),
        "gradcam_overlay": pil_to_base64_data_url(overlay_gcam),
        "original_image": pil_to_base64_data_url(orig_resized)
    }

@app.get("/api/samples")
async def get_sample_images():
    """
    Returns base64 data URLs for 3 representative sample images (Normal, Early, Advanced)
    for one-click testing. Works in local environments and deployed cloud containers.
    """
    samples = {
        "Normal": [
            os.path.join(STATIC_DIR, "samples", "normal_sample.png"),
            "c:/Users/KIIT0001/GlaucoVision/processed_data(Harvard)/normal_control/1.png"
        ],
        "Early-stage": [
            os.path.join(STATIC_DIR, "samples", "early_sample.png"),
            "c:/Users/KIIT0001/GlaucoVision/processed_data(Harvard)/early_glaucoma/1.png"
        ],
        "Advanced-stage": [
            os.path.join(STATIC_DIR, "samples", "advanced_sample.png"),
            "c:/Users/KIIT0001/GlaucoVision/processed_data(Harvard)/advanced_glaucoma/1.png"
        ]
    }
    
    result = []
    for stage, paths in samples.items():
        for p in paths:
            if os.path.exists(p):
                try:
                    im = Image.open(p).convert("RGB").resize((224, 224))
                    result.append({
                        "stage": stage,
                        "data_url": pil_to_base64_data_url(im)
                    })
                    break
                except Exception:
                    continue
    return result

# Serve static frontend files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "CA-Net API Online. Static files not yet mounted."})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend:app", host=host, port=port, reload=True)
