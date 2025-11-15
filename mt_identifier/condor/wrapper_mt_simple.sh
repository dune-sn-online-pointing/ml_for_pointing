#!/bin/bash
set -e

JSON_CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json) JSON_CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1; ;;
    esac
done

[[ -z "$JSON_CONFIG" ]] && { echo "Error: JSON config required"; exit 1; }

PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/refactor_ml"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/init.sh"

nvidia-smi 2>/dev/null || echo "No GPU available"

echo "Starting MT training..."
python3 mt_identifier/models/mt_training.py --input_json "$JSON_CONFIG"
EXIT_CODE=$?

echo "Job finished: $(date) | Exit code: $EXIT_CODE"
exit $EXIT_CODE
