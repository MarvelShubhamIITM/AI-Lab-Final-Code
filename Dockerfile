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
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway's own checkout of this repo does not resolve Git LFS pointers, so
# the *.pt/*.pth/*.task/*.csv files copied above are tiny LFS pointer stubs
# ("version https://git-lfs.github.com/spec/v1...") instead of the real
# checkpoints -- torch.load then fails with "invalid load key, 'v'." (the
# pointer text's leading byte). Re-fetch the real binaries with a throwaway
# clone that has git-lfs installed and actually smudges them.
RUN git clone --depth 1 https://github.com/MarvelShubhamIITM/AI-Lab-Final-Code.git /tmp/lfs-src \
    && cp -f /tmp/lfs-src/models/* models/ \
    && rm -rf /tmp/lfs-src

EXPOSE 7860

CMD ["python", "app.py"]
