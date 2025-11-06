#!/bin/bash
set -e
export SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPTS_DIR/init.sh

json_file=""
plane=""
max_samples=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json) json_file="$2"; shift 2;;
    --plane) plane="$2"; shift 2;;
    --max-samples) max_samples="$2"; shift 2;;
    *) shift;;
  esac
done

CMD=("python3" "electron_direction/ed_training.py" "--input_json" "$json_file")
[[ -n "$plane" ]] && CMD+=("--plane" "$plane")
[[ -n "$max_samples" ]] && CMD+=("--max_samples" "$max_samples")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
