"""
GlaucoVision - CA-Net Launch Utility
Runs the FastAPI web server with Uvicorn.
Access via browser at http://127.0.0.1:8000
"""

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print("=" * 60)
    print(" GlaucoVision | CA-Net Multi-Stage Glaucoma Detection System")
    print(" IEEE Transactions on Instrumentation and Measurement, 2023")
    print(f" Server running on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run("app.backend:app", host=host, port=port, reload=True)
