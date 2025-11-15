#!/bin/bash
#
# Training script for Electron Direction Regressor (Three Planes)
# Usage: train_electron_direction_3plane.sh [--local|--condor] [options]
#

# Default parameters
PLANE="all"  # Three planes: U, V, X
JSON_CONFIG="json/electron_direction_3plane/production_training.json"
MAX_SAMPLES_VALUE=""
DATA_DIR_PATH=""
VERBOSE_FLAG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --local)
      RUN_MODE="local"
      shift
      ;;
    --condor)
      RUN_MODE="condor"
      shift
      ;;
    --json)
      JSON_CONFIG="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES_VALUE="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR_PATH="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE_FLAG="--verbose"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_DIR"

# Convert config path to absolute if needed and ensure it exists
if [[ "$JSON_CONFIG" != /* ]]; then
  JSON_CONFIG="$PROJECT_DIR/$JSON_CONFIG"
fi

if [[ ! -f "$JSON_CONFIG" ]]; then
  echo "Error: JSON config not found at $JSON_CONFIG"
  exit 1
fi

echo "======================================================================"
echo "ELECTRON DIRECTION TRAINING (THREE PLANES)"
echo "======================================================================"
echo "Project directory: $PROJECT_DIR"
echo "JSON config: $JSON_CONFIG"
echo "Run mode: ${RUN_MODE:-not specified}"
echo "======================================================================"

# Source init script for environment setup
source "$SCRIPT_DIR/init.sh"

# Build python arguments using an array to preserve spacing
python_args=("--input_json" "$JSON_CONFIG")

if [[ -n "$MAX_SAMPLES_VALUE" ]]; then
  python_args+=("--max_samples" "$MAX_SAMPLES_VALUE")
fi

if [[ -n "$DATA_DIR_PATH" ]]; then
  python_args+=("--data_dir" "$DATA_DIR_PATH")
fi

if [[ -n "$VERBOSE_FLAG" ]]; then
  python_args+=("$VERBOSE_FLAG")
fi

# Run training
echo ""
echo "Starting training..."
echo "Command: python3 electron_direction_3plane/ed3p_training.py ${python_args[*]}"
echo ""

cd electron_direction_3plane
python3 ed3p_training.py "${python_args[@]}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Training completed successfully!"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Training failed with exit code: $exit_code"
    echo "======================================================================"
fi

exit $exit_code
