#!/bin/bash
# HTCondor wrapper script for MT Identifier training
# This script runs inside the condor job

set -e

# Parse arguments
JSON_CONFIG=""
PLANE="X"

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_CONFIG="$2"
            shift 2
            ;;
        -p|--plane)
            PLANE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$JSON_CONFIG" ]]; then
    echo "Error: JSON config required (-j/--json)"
    exit 1
fi

echo "========================================="
echo "MT IDENTIFIER TRAINING - HTCondor Job"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Plane: $PLANE"
echo "JSON config: $JSON_CONFIG"
echo "========================================="

# Navigate to project directory first
PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/refactor_ml"
cd "$PROJECT_DIR"

# Setup environment using init.sh (sources LCG, sets PYTHONPATH)
echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""

# Check GPU availability (optional, fail gracefully)
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU available (CPU-only training)"
echo ""

# Navigate to project directory and run training
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run training
echo "Starting training..."
echo "Command: python3 mt_identifier/models/main_production.py -j $JSON_CONFIG --plane $PLANE"
echo ""

python3 mt_identifier/models/main_production.py -j "$JSON_CONFIG" --plane "$PLANE"

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
