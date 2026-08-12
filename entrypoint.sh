#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Downloading embedder model..."
python scripts/download.py

echo "Running database ingestion check..."
python src/ingest.py

echo "Starting Streamlit application..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
