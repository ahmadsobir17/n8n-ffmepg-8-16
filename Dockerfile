# Start from official Node.js Debian image  
FROM node:22-bookworm-slim

# Install system dependencies (including OpenCV and Whisper requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    xz-utils \
    ca-certificates \
    tini \
    libgl1 \
    libglib2.0-0 \
    git \
    procps \
    nodejs \
    fontconfig \
    fonts-liberation \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages: yt-dlp, OpenCV, MediaPipe (pinned version for mp.solutions compatibility)
RUN pip3 install --break-system-packages yt-dlp opencv-python-headless mediapipe==0.10.9

# Install PyTorch CPU-only (smaller, no CUDA) then Whisper
RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --break-system-packages openai-whisper

# Pre-download Whisper model (turbo = large-v3-turbo, higher accuracy)
RUN python3 -c "import whisper; whisper.load_model('turbo')"

# FFmpeg is now installed via apt-get for better reliability

# Install n8n globally
RUN npm install -g n8n

# Copy face detection script
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.py

# Create n8n directory
RUN mkdir -p /home/node/.n8n && chown -R node:node /home/node

# Default n8n port
EXPOSE 5678

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start n8n
CMD ["n8n", "start"]
