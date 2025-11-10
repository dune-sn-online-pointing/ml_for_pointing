#!/bin/bash
# HTCondor wrapper script for ML training
# This script runs inside the condor job

set -e

PLANE=${1:-X}
MAX_SAMPLES=${2:-}
JSON_CONFIG=${3:-json/mt_identifier/production_training_prod_main.json}

# Handle the case where HTCondor collapses empty positional arguments and the
# JSON config path arrives as the second parameter.
if [[ "$MAX_SAMPLES" == *.json ]]; then
    JSON_CONFIG="$MAX_SAMPLES"
    MAX_SAMPLES=""
fi

echo "========================================="
echo "HTCondor GPU Training Job"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Plane: $PLANE"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "JSON config: $JSON_CONFIG"
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

CMD=("./scripts/train_mt_identifier.sh" "-j" "$JSON_CONFIG" "--plane" "$PLANE")
if [[ -n "$MAX_SAMPLES" ]]; then
    if [[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
        CMD+=("--max-samples" "$MAX_SAMPLES")
    else
        echo "Warning: Ignoring unexpected max samples value: $MAX_SAMPLES"
    fi
fi

echo "Running: ${CMD[*]}"
echo "========================================="
"${CMD[@]}"

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
