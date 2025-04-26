FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    ultralytics==8.1.27 \
    deep-sort-realtime==1.3.2 \
    opencv-python-headless==4.8.1.78  # Headless version avoids GUI dependencies

WORKDIR /workspace