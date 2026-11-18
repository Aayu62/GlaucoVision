# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules and trained model weights
COPY models/ ./models/
COPY explainability/ ./explainability/
COPY dataset/ ./dataset/
COPY checkpoints/best_lmg_densenet121_cam.pth ./checkpoints/best_lmg_densenet121_cam.pth
COPY checkpoints/history_lmg_densenet121_cam.json ./checkpoints/history_lmg_densenet121_cam.json
COPY checkpoints/test_results_lmg_densenet121_cam.json ./checkpoints/test_results_lmg_densenet121_cam.json
COPY app/ ./app/

# Expose default port
EXPOSE 7860

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "7860"]

