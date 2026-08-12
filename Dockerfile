FROM python:3.13-slim

# System libraries required by opencv-python-headless, mediapipe, and the
# ffmpeg transcode step in wellness_core.py. Railway's default Railpack/
# Nixpacks builder ships a minimal image without these, which is why
# `import cv2` previously failed with:
#   ImportError: libxcb.so.1: cannot open shared object file
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
