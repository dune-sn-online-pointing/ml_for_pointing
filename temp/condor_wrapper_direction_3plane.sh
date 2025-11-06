#!/bin/bash
# HTCondor wrapper script for 3-Plane Direction Regressor training
# This script runs inside the condor job

set -e

JSON_CONFIG=${1:-/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/json/direction_3plane/production_training.json}
MAX_SAMPLES=${2:-}

echo "========================================="
echo "HTCondor GPU 3-Plane Direction Training"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "========================================="

# Setup environment - Use LCG stack with CUDA
LCG_RELEASE="LCG_106_cuda/x86_64-el9-gcc11-opt"
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/$LCG_RELEASE"

echo "Setting up LCG environment: $LCG_RELEASE"
source "$LCG_VIEW/setup.sh"

echo "Using LCG Python: $(which python3)"
python3 --version
echo ""

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

# Check Python and TensorFlow
echo ""
echo "Python version: $(python3 --version 2>&1)"
echo "Python path: $(which python3)"
echo "Checking TensorFlow..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]" 2>&1
echo ""

# Run training
PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
cd "$PROJECT_DIR"

ARGS="--json $JSON_CONFIG"
if [ -n "$MAX_SAMPLES" ]; then
    ARGS="$ARGS --max-samples $MAX_SAMPLES"
fi

echo "Running: ./scripts/train_direction_3plane.sh $ARGS"
echo "========================================="
./scripts/train_direction_3plane.sh $ARGS

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
