#!/bin/bash
set -e
PLANE=${1:-X}
MAX_SAMPLES=${2:-}
JSON_CONFIG=${3:-json/electron_direction/production_training_new.json}
[[ "$MAX_SAMPLES" == *.json ]] && JSON_CONFIG="$MAX_SAMPLES" && MAX_SAMPLES=""

echo "========================================="
echo "HTCondor Electron Direction Training"
echo "Job started: $(date) | Host: $(hostname)"
echo "Config: $JSON_CONFIG | Plane: $PLANE"
echo "========================================="

LCG_RELEASE="LCG_106_cuda/x86_64-el9-gcc11-opt"
source "/cvmfs/sft.cern.ch/lcg/views/$LCG_RELEASE/setup.sh"
nvidia-smi || echo "No GPU"

cd "/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
CMD=("./scripts/train_electron_direction.sh" "-j" "$JSON_CONFIG" "--plane" "$PLANE")
[[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" =~ ^[0-9]+$ ]] && CMD+=("--max-samples" "$MAX_SAMPLES")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Job completed: $(date)"
