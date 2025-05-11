#!/bin/bash
set -e

# Rebuild and run the container with GPU check
docker compose run --rm grande-gpu python3 /app/check_gpu.py

# If successful, inform user they can run Jupyter
echo -e "\n\nIf GPU detection is successful, you can run:"
echo "docker compose up grande-gpu"
echo "and access Jupyter at http://localhost:8888"
