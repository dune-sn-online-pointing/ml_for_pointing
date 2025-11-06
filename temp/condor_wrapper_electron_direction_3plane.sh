#!/bin/bash
#
# HTCondor wrapper for Electron Direction training (Three Planes)
#

echo "======================================================================"
echo "HTCondor Job: Electron Direction (Three Planes) Training"
echo "======================================================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "======================================================================"

# Source LCG environment
echo "Setting up LCG_106_cuda environment..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

echo ""
echo "Environment setup complete"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi

echo ""
echo "TensorFlow GPU check:"
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPUs available: {len(tf.config.list_physical_devices(\"GPU\"))}')"

echo ""
echo "======================================================================"
echo "Starting training script..."
echo "======================================================================"

# Parse arguments passed from condor submission and forward to training script
declare -a CMD
CMD=("/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/scripts/train_electron_direction_3plane.sh" "--condor")

while [[ $# -gt 0 ]]; do
	case "$1" in
		--json)
			CMD+=("--json" "$2")
			shift 2
			;;
		--max_samples)
			CMD+=("--max_samples" "$2")
			shift 2
			;;
		--data_dir)
			CMD+=("--data_dir" "$2")
			shift 2
			;;
		--verbose)
			CMD+=("--verbose")
			shift
			;;
		*)
			echo "Warning: Unrecognized argument '$1'" >&2
			shift
			;;
	esac
done

echo "Executing: ${CMD[*]}"

"${CMD[@]}"

exit_code=$?

echo ""
echo "======================================================================"
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "======================================================================"

exit $exit_code
