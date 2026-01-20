#!/bin/bash

# =============================================
# n8n Workflow Sync Script
# =============================================
# Script untuk export/import workflows antara VPS dan lokal
# =============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKFLOWS_DIR="$PROJECT_DIR/workflows"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect container name
detect_container() {
    if docker ps --format '{{.Names}}' | grep -q "n8n-dev"; then
        echo "n8n-dev"
    elif docker ps --format '{{.Names}}' | grep -q "n8n-local"; then
        echo "n8n-local"
    elif docker ps --format '{{.Names}}' | grep -q "n8n-prod"; then
        echo "n8n-prod"
    else
        echo ""
    fi
}

export_workflows() {
    local container=$(detect_container)
    
    if [ -z "$container" ]; then
        echo_error "No n8n container found running!"
        exit 1
    fi
    
    echo_info "Exporting workflows from container: $container"
    
    # Create backup
    if [ -f "$WORKFLOWS_DIR/all-workflows.json" ]; then
        cp "$WORKFLOWS_DIR/all-workflows.json" "$WORKFLOWS_DIR/backup_workflows_$TIMESTAMP.json"
        echo_info "Backup created: backup_workflows_$TIMESTAMP.json"
    fi
    
    # Export workflows
    docker exec "$container" n8n export:workflow --all --output=/home/node/.n8n/all-workflows.json
    
    # Copy to workflows folder
    docker cp "$container:/home/node/.n8n/all-workflows.json" "$WORKFLOWS_DIR/all-workflows.json"
    
    echo_info "✅ Workflows exported successfully!"
    echo_info "File: $WORKFLOWS_DIR/all-workflows.json"
}

export_credentials() {
    local container=$(detect_container)
    
    if [ -z "$container" ]; then
        echo_error "No n8n container found running!"
        exit 1
    fi
    
    echo_info "Exporting credentials from container: $container"
    
    # Create backup
    if [ -f "$WORKFLOWS_DIR/all-credentials.json" ]; then
        cp "$WORKFLOWS_DIR/all-credentials.json" "$WORKFLOWS_DIR/backup_credentials_$TIMESTAMP.json"
        echo_info "Backup created: backup_credentials_$TIMESTAMP.json"
    fi
    
    # Export credentials
    docker exec "$container" n8n export:credentials --all --output=/home/node/.n8n/all-credentials.json
    
    # Copy to workflows folder
    docker cp "$container:/home/node/.n8n/all-credentials.json" "$WORKFLOWS_DIR/all-credentials.json"
    
    echo_warn "⚠️  Credentials exported! Make sure N8N_ENCRYPTION_KEY is the same on both environments."
    echo_info "File: $WORKFLOWS_DIR/all-credentials.json"
}

import_workflows() {
    local container=$(detect_container)
    
    if [ -z "$container" ]; then
        echo_error "No n8n container found running!"
        exit 1
    fi
    
    if [ ! -f "$WORKFLOWS_DIR/all-workflows.json" ]; then
        echo_error "Workflow file not found: $WORKFLOWS_DIR/all-workflows.json"
        exit 1
    fi
    
    echo_info "Importing workflows to container: $container"
    
    # Copy file to container
    docker cp "$WORKFLOWS_DIR/all-workflows.json" "$container:/home/node/.n8n/all-workflows.json"
    
    # Import workflows
    docker exec "$container" n8n import:workflow --input=/home/node/.n8n/all-workflows.json
    
    echo_info "✅ Workflows imported successfully!"
}

import_credentials() {
    local container=$(detect_container)
    
    if [ -z "$container" ]; then
        echo_error "No n8n container found running!"
        exit 1
    fi
    
    if [ ! -f "$WORKFLOWS_DIR/all-credentials.json" ]; then
        echo_error "Credentials file not found: $WORKFLOWS_DIR/all-credentials.json"
        exit 1
    fi
    
    echo_warn "⚠️  Make sure N8N_ENCRYPTION_KEY is the SAME as the source environment!"
    echo -n "Continue? (y/n): "
    read confirm
    
    if [ "$confirm" != "y" ]; then
        echo_info "Cancelled."
        exit 0
    fi
    
    echo_info "Importing credentials to container: $container"
    
    # Copy file to container
    docker cp "$WORKFLOWS_DIR/all-credentials.json" "$container:/home/node/.n8n/all-credentials.json"
    
    # Import credentials
    docker exec "$container" n8n import:credentials --input=/home/node/.n8n/all-credentials.json
    
    echo_info "✅ Credentials imported successfully!"
}

export_all() {
    export_workflows
    export_credentials
    echo ""
    echo_info "🎉 All data exported! You can now:"
    echo_info "1. Backup workflows folder manually"
    echo_info "2. Pull on your local machine"
    echo_info "3. Run './scripts/sync-workflows.sh import-all'"
}

import_all() {
    import_workflows
    import_credentials
    echo ""
    echo_info "🎉 All data imported!"
    echo_info "Restart n8n container to apply changes: docker restart $(detect_container)"
}

show_help() {
    echo "n8n Workflow Sync Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  export-workflows    Export all workflows to JSON"
    echo "  export-credentials  Export all credentials to JSON"
    echo "  export-all          Export both workflows and credentials"
    echo "  import-workflows    Import workflows from JSON"
    echo "  import-credentials  Import credentials from JSON"
    echo "  import-all          Import both workflows and credentials"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 export-all       # On VPS: export all data"
    echo "  $0 import-all       # On Local: import all data"
}

# Main
case "${1:-help}" in
    export-workflows)
        export_workflows
        ;;
    export-credentials)
        export_credentials
        ;;
    export-all)
        export_all
        ;;
    import-workflows)
        import_workflows
        ;;
    import-credentials)
        import_credentials
        ;;
    import-all)
        import_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
