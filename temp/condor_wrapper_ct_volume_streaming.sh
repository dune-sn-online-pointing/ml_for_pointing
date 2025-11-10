#!/bin/bash

# Wrapper script for CT volume training (streaming) on HTCondor

set -e

echo "=========================================="
echo "CT VOLUME STREAMING TRAINING"
echo "=========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

# Parse arguments
PLANE="X"
MAX_SAMPLES=50000
JSON_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--plane)
            PLANE="$2"
            shift 2
            ;;
        -m|--max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -j|--json)
            JSON_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Plane: $PLANE"
echo "Max samples: $MAX_SAMPLES"
echo "JSON config: $JSON_CONFIG"

# Activate LCG environment with GPU support
echo ""
echo "Activating LCG environment..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Verify environment
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo "TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'NO')"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi || echo "No GPU found"

# Navigate to project directory
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing

# Run training
echo ""
echo "Starting streaming training..."
python3 channel_tagging/train_ct_volume_streaming.py \
    --plane "$PLANE" \
    --max-samples "$MAX_SAMPLES" \
    --json "$JSON_CONFIG"

EXIT_CODE=$?

echo ""
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE
