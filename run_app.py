"""
GlaucoVision - CA-Net Launch Utility
Runs the FastAPI web server with Uvicorn.
Access via browser at http://127.0.0.1:8000
"""

import os
import sys
import subprocess
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_lmg_densenet121_cam.pth")

# Ensure checkpoint is a real binary and not an unpulled Git LFS pointer
if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) < 1000:
    print("Notice: Checkpoint is a Git LFS pointer. Running 'git lfs checkout' to restore weights binary...")
    try:
        subprocess.run(["git", "lfs", "checkout"], cwd=BASE_DIR, check=True)
    except Exception as e:
        print(f"Warning: Could not automatically checkout LFS objects: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print("=" * 60)
    print(" GlaucoVision | CA-Net Multi-Stage Glaucoma Detection System")
    print(" IEEE Transactions on Instrumentation and Measurement, 2023")
    print(f" Server running on http://{host}:{port}")
    print("=" * 60)

    # Use app_dir="app" to load backend:app cleanly without module collision with app.py
    app_dir = os.path.join(BASE_DIR, "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    uvicorn.run("backend:app", host=host, port=port, reload=True, app_dir=app_dir)
