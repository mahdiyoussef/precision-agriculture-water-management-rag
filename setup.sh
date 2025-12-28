#!/bin/bash
# Setup script for Precision Agriculture RAG System

echo "======================================"
echo "Setting up RAG Environment"
echo "======================================"

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv rag_env

# Activate environment
echo "[2/5] Activating environment..."
source rag_env/bin/activate

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install -r requirements.txt

# Download SpaCy model
echo "[5/5] Downloading SpaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source rag_env/bin/activate"
echo "2. Verify Ollama: ollama list (qwen:1.8b should be listed)"
echo "3. Run ingestion: python src/ingest.py"
echo "4. Start chatting: python src/main.py"
echo ""
