# n8n Short AI Clipper 🎬🤖

[![n8n](https://img.shields.io/badge/n8n-powered-orange.svg)](https://n8n.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

Automated video clipping workflow for creating vertical shorts/reels from horizontal videos (podcasts, interviews) using n8n and AI-powered face detection.

> [!NOTE]
> This repository is optimized for production environments with limited resources (e.g., 1GB RAM VPS).

## ✨ Features

- 🎭 **AI Face Detection**: Uses MediaPipe to identify speakers and automatically switch between close-up and wide shots.
- 📝 **Auto Subtitles**: Integrates OpenAI Whisper for word-timed vertical subtitles.
- ⚙️ **n8n Workflow**: Full automation engine to handle video downloads, processing, and output management.
- 🐳 **Dockerized**: One-command deployment with production-ready configuration (Postgres, Redis, n8n).
- 🔒 **Cloudflare Tunnel**: Secure access without opening ports.

## 📁 Repository Structure

- `workflows/`: Exported n8n workflows and credentials.
- `scripts/`: Python AI processing scripts (`face_crop.py`) and utility scripts.
- `docker-compose.yml`: Main orchestration for the stack.
- `Dockerfile`: Custom n8n image with FFmpeg and Python dependencies.

## 🚀 Quick Start

### 1. Requirements
- Docker & Docker Compose
- Cloudflare Tunnel Token (optional, for secure web access)

### 2. Setup
Clone the repo and configure your environment:
```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

### 3. Deploy
```bash
docker compose up -d
```

### 4. Import Workflows
Use the sync script to import all workflows into your n8n instance:
```bash
./scripts/sync-workflows.sh import-all
```

---

## 🛠 AI Engine Details

The core processing happens in `scripts/face_crop.py`, which:
1. Detects faces using **MediaPipe**.
2. Analyzes timestamps to determine whether a split-screen or single-shot layout is more appropriate.
3. Transcribes audio using **Whisper** (Tiny model optimized for speed).
4. Generates `.ass` subtitles and burns them into the final vertical MP4 using **FFmpeg**.

---

## 🇮🇩 Bahasa Indonesia (Ringkasan)

Repository ini berisi sistem otomasi pembuatan video Shorts/Reels dari video horizontal (seperti podcast). Menggunakan **n8n** sebagai otak otomasi dan **AI (MediaPipe & Whisper)** untuk deteksi wajah serta pembuatan subtitle otomatis. Sudah dioptimasi untuk berjalan di VPS RAM 1GB.

---
*Last Updated: 2026-01-30*
