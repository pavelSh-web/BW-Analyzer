# Base image
FROM python:3.10-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# System dependencies for audio processing
# - ffmpeg: decoding/encoding support used by librosa/torchaudio
# - libsndfile1: required by soundfile (pysoundfile)
# - build-essential & gcc: some wheels may need compilation (scipy stack)
# - libgl1: avoid OpenL3/NumPy/OpenCV-related GL import errors in some environments
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       build-essential \
       gcc \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python deps
# Note: torch/torchaudio CPU wheels come from the extra index url in requirements.txt
ENV PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY app ./app

# Create storage dir (can be overridden by volume)
ENV STORAGE_DIR=/app/storage
RUN mkdir -p ${STORAGE_DIR}

# Expose API port
EXPOSE 8000

# Default command: run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
