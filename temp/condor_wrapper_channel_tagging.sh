#!/bin/bash
# HTCondor wrapper for Channel Tagging training
set -e

# Parse arguments with flags
PLANE="X"
MAX_SAMPLES=""
JSON_CONFIG="json/channel_tagging/production_training_100k_new.json"

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

echo "========================================="
echo "HTCondor Channel Tagging Training"
echo "========================================="
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Plane: $PLANE"
echo "Max samples: ${MAX_SAMPLES:-from config}"
echo "Config: $JSON_CONFIG"
echo "========================================="

# Setup LCG environment
LCG_RELEASE="LCG_106_cuda/x86_64-el9-gcc11-opt"
source "/cvmfs/sft.cern.ch/lcg/views/$LCG_RELEASE/setup.sh"

echo "LCG environment: $LCG_RELEASE"
echo "Python: $(which python3) ($(python3 --version 2>&1))"
nvidia-smi || echo "No GPU"

# Run training
PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
cd "$PROJECT_DIR"

CMD=("./scripts/train_channel_tagging.sh" "-j" "$JSON_CONFIG" "--plane" "$PLANE")
[[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" =~ ^[0-9]+$ ]] && CMD+=("--max-samples" "$MAX_SAMPLES")

echo "Running: ${CMD[*]}"
echo "========================================="
"${CMD[@]}"

echo "Job completed: $(date)"
