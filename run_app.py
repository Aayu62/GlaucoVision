"""
GlaucoVision - CA-Net Launch Utility
Runs the FastAPI web server with Uvicorn.
Access via browser at http://127.0.0.1:8000
"""

import uvicorn
import os

if __name__ == "__main__":
    print("=" * 60)
    print(" GlaucoVision | CA-Net Multi-Stage Glaucoma Detection System")
    print(" IEEE Transactions on Instrumentation and Measurement, 2023")
    print(" Server URL: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8000, reload=True)
