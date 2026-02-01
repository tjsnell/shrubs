#!/bin/bash
# Nightly pipeline runner for cron

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run pipeline
python scripts/run_pipeline.py

echo "Pipeline completed at $(date)"
