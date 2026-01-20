#!/bin/bash

# =============================================
# n8n Auto Backup to GitHub
# =============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKFLOWS_DIR="$PROJECT_DIR/workflows"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Export Data menggunakan script yang sudah ada
cd "$PROJECT_DIR"
echo_info "Exporting n8n workflows and credentials..."
./scripts/sync-workflows.sh export-all

# 2. Setup Git jika belum ada
if [ ! -d ".git" ]; then
    echo_info "Initializing git repository..."
    git init
    git remote add origin https://github.com/ahmadsobir17/n8n-production.git
    git branch -M main
fi

# 3. Commit dan Push
echo_info "Pushing to GitHub..."
git add workflows/*.json .gitignore docker-compose.yml scripts/*.sh 2>/dev/null || true
git commit -m "auto-backup: $(date '+%Y-%m-%d %H:%M:%S')" || echo_info "No changes to commit"

# Push ke github (pastikan credentials git sudah tersimpan/menggunakan SSH/token)
git push origin main || echo_error "Failed to push to GitHub. Make sure git credentials are set."

echo_info "✅ Backup completed successfully!"
