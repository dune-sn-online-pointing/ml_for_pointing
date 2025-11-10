#!/bin/bash
# HTCondor wrapper script for three-plane electron direction training

set -e

TASK_TYPE=${1:-electron_direction}
TRAINING_SCRIPT=${2:-electron_direction/train_three_plane_matched.py}
JSON_CONFIG=${3:-json/electron_direction/three_plane_matched_v1.json}

echo "========================================="
echo "HTCondor GPU Training Job - Three Plane"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Task type: $TASK_TYPE"
echo "Training script: $TRAINING_SCRIPT"
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

CMD=("python3" "$TRAINING_SCRIPT" "--input_json" "$JSON_CONFIG")

echo "Running: ${CMD[*]}"
echo "========================================="
"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
