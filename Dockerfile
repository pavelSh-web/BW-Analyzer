# Base image
FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# System dependencies for audio processing
# - ffmpeg: decoding/encoding support used by librosa/torchaudio
# - libsndfile1: required by soundfile (pysoundfile)
# - build-essential & gcc: some wheels may need compilation (scipy stack)
# - libgl1: avoid OpenL3/NumPy/OpenCV-related GL import errors in some environments
# - wget: for downloading PANNs data files
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       build-essential \
       gcc \
       libgl1 \
       wget \
       git \
    && rm -rf /var/lib/apt/lists/*

# Create PANNs data directory and download required files
RUN mkdir -p /root/panns_data \
    && wget -O /root/panns_data/class_labels_indices.csv \
       https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python deps
# Note: torch/torchaudio CPU wheels come from the extra index url in requirements.txt
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Environment variables
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command: run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]