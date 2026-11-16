import requests
import json

url_predict = "http://127.0.0.1:8000/api/predict"
url_explain = "http://127.0.0.1:8000/api/explain"
img_path = "c:/Users/KIIT0001/GlaucoVision/processed_data(Harvard)/early_glaucoma/1.png"

with open(img_path, "rb") as f:
    files = {"file": ("early_1.png", f, "image/png")}
    res_pred = requests.post(url_predict, files=files)
    print("Prediction status:", res_pred.status_code)
    print("Prediction payload:", json.dumps(res_pred.json(), indent=2))

with open(img_path, "rb") as f:
    files = {"file": ("early_1.png", f, "image/png")}
    res_exp = requests.post(url_explain, files=files)
    print("\nExplainability status:", res_exp.status_code)
    exp_json = res_exp.json()
    print("Predicted stage:", exp_json.get("predicted_class"))
    print("Confidence:", exp_json.get("confidence"))
    print("Grad-CAM++ base64 length:", len(exp_json.get("gradcam_pp_overlay", "")))
