#!/bin/bash
# =============================================================================
# Precision Agriculture RAG API Server
# =============================================================================
# 
# Starts the FastAPI server with Swagger documentation.
#
# Usage:
#   ./start_api.sh              # Start on default port 8000
#   ./start_api.sh 8080         # Start on custom port
#
# API Documentation:
#   Swagger UI: http://localhost:8000/docs
#   ReDoc:      http://localhost:8000/redoc
#
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Port (default: 8000)
PORT="${1:-8000}"

# Activate virtual environment
if [ -d "rag_env" ]; then
    echo "[*] Activating virtual environment..."
    source rag_env/bin/activate
else
    echo "[!] Virtual environment 'rag_env' not found!"
    echo "    Run: python3 -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check dependencies
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "[*] Installing FastAPI and Uvicorn..."
    pip install fastapi uvicorn --quiet
}

# Start server
echo ""
echo "=============================================="
echo "  Precision Agriculture RAG API"
echo "=============================================="
echo "  Swagger UI: http://localhost:$PORT/docs"
echo "  ReDoc:      http://localhost:$PORT/redoc"
echo "  API Base:   http://localhost:$PORT"
echo "=============================================="
echo ""
echo "[*] Starting server on port $PORT..."
echo "    Press Ctrl+C to stop"
echo ""

uvicorn src.adapters.api.server:app --reload --host 0.0.0.0 --port "$PORT"
