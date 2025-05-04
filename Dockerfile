FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libxcb1 \
    libxcb-util1 \
    libglu1-mesa \
    libegl1-mesa \
    libgl1-mesa-glx \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    ultralytics==8.1.27 \
    deep-sort-realtime==1.3.2 \
    opencv-python \
    scikit-learn

WORKDIR /workspace