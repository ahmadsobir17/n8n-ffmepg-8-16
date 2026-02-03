# n8n Short AI Clipper 🎬🤖

[![n8n](https://img.shields.io/badge/n8n-powered-orange.svg)](https://n8n.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

Automated video clipping workflow for creating vertical shorts/reels from horizontal videos (podcasts, interviews) using n8n and AI-powered face detection.

> [!IMPORTANT]
> This repository is optimized for **High-Performance VPS** (e.g., 8-Core CPU / 16GB RAM) utilizing multi-threading and advanced AI models.

## ✨ Features

- 🎭 **AI Face Detection**: Uses MediaPipe to identify speakers and automatically switch between close-up and wide shots. Optimized for multi-core CPUs.
- 📝 **Whisper Turbo**: Integrates OpenAI **Whisper Large-v3-Turbo** for ultra-accurate word-timed vertical subtitles.
- ⚙️ **n8n Workflow**: Full automation engine to handle video downloads, processing, and output management with support for high concurrency.
- 🐳 **Dockerized**: One-command deployment with production-ready configuration (Postgres, Redis, n8n with 4GB RAM limit).
- 🔒 **Cloudflare Tunnel**: Secure access without opening ports.

## 📁 Repository Structure

- `workflows/`: Exported n8n workflows and credentials.
- `scripts/`: Python AI processing scripts (`face_crop.py`) and utility scripts.
- `docker-compose.yml`: Main orchestration for the stack.
- `Dockerfile`: Custom n8n image with FFmpeg, Python dependencies, and pre-loaded AI models.

## 🚀 Quick Start

### 1. Requirements
- Docker & Docker Compose
- 8-Core CPU & 8GB+ RAM Recommended
- Cloudflare Tunnel Token (optional, for secure web access)

### 2. Setup
Clone the repo and configure your environment:
```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

### 3. Deploy
```bash
docker compose up -d --build
```

### 4. Import Workflows
Use the sync script to import all workflows into your n8n instance:
```bash
./scripts/sync-workflows.sh import-all
```

---

## 🛠 AI Engine Details

The core processing happens in `scripts/face_crop.py`, which:
1. Detects faces using **MediaPipe** (utilizing 4 parallel threads).
2. Analyzes timestamps to determine whether a split-screen or single-shot layout is more appropriate.
3. Transcribes audio using **Whisper Large-v3-Turbo** (high accuracy transcription).
4. Generates `.ass` subtitles and burns them into the final vertical MP4 using **FFmpeg** with optimized muxing.

---

## 🇮🇩 Bahasa Indonesia (Ringkasan)

Repository ini berisi sistem otomasi pembuatan video Shorts/Reels dari video horizontal (seperti podcast). Menggunakan **n8n** sebagai otak otomasi dan **AI (MediaPipe & Whisper Turbo)** untuk deteksi wajah serta pembuatan subtitle otomatis yang sangat akurat. Versi ini telah **dioptimasi khusus untuk VPS High-Spec** (8 Core CPU / 16GB RAM) dengan fitur multi-threading dan concurrency tinggi.

---
*Last Updated: 2026-02-03 (High-Spec Optimization)*
