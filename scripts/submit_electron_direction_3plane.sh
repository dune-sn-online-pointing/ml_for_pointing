#!/bin/bash
#
# Submission script for Electron Direction Training (Three Planes)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_JSON="json/electron_direction_3plane/production_training.json"

USE_CONDOR=true
DRY_RUN=false
JSON_CONFIG="$DEFAULT_JSON"
MAX_SAMPLES=""
DATA_DIR=""
VERBOSE=false

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Submit Electron Direction (three-plane) regression training.

Options:
    --condor              Submit to HTCondor (default)
    --local               Run locally
    --json <path>         JSON config file (default: $DEFAULT_JSON)
    --max-samples <N>     Limit number of samples (testing only)
    --data-dir <path>     Override dataset directory
    --verbose             Enable verbose mode
    --dry-run             Print actions without executing
    -h, --help            Show this help text

Examples:
    $0 --condor
    $0 --condor --json json/electron_direction_3plane/production_training.json
    $0 --local --max-samples 2000 --verbose
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condor)
            USE_CONDOR=true
            shift
            ;;
        --local)
            USE_CONDOR=false
            shift
            ;;
        --json|-j)
            JSON_CONFIG="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Resolve JSON config to an absolute path
if [[ -f "$JSON_CONFIG" ]]; then
    JSON_CONFIG_ABS=$(readlink -f "$JSON_CONFIG")
elif [[ -f "$PROJECT_DIR/$JSON_CONFIG" ]]; then
    JSON_CONFIG_ABS=$(readlink -f "$PROJECT_DIR/$JSON_CONFIG")
else
    echo "Error: JSON configuration file not found: $JSON_CONFIG" >&2
    exit 1
fi

echo "========================================"
echo "Electron Direction 3-Plane Submission"
echo "========================================"
echo "Mode: $([[ "$USE_CONDOR" == true ]] && echo "HTCondor" || echo "Local")"
echo "Project: $PROJECT_DIR"
echo "Config: $JSON_CONFIG_ABS"
[[ -n "$MAX_SAMPLES" ]] && echo "Max Samples: $MAX_SAMPLES"
[[ -n "$DATA_DIR" ]] && echo "Data Dir: $DATA_DIR"
[[ "$VERBOSE" == true ]] && echo "Verbose: enabled"
echo "========================================"

ARGS=("--json" "$JSON_CONFIG_ABS")
[[ -n "$MAX_SAMPLES" ]] && ARGS+=("--max_samples" "$MAX_SAMPLES")
[[ -n "$DATA_DIR" ]] && ARGS+=("--data_dir" "$DATA_DIR")
[[ "$VERBOSE" == true ]] && ARGS+=("--verbose")

if [[ "$USE_CONDOR" == true ]]; then
    CONDOR_SUB="$PROJECT_DIR/submit_condor_electron_direction_3plane.sub"
    if [[ ! -f "$CONDOR_SUB" ]]; then
        echo "Error: HTCondor submission file not found: $CONDOR_SUB" >&2
        exit 1
    fi

    mkdir -p "$PROJECT_DIR/logs"

    ARG_STRING="${ARGS[*]}"
    TEMP_SUB=$(mktemp)
    sed "s|^arguments.*|arguments               = $ARG_STRING|" "$CONDOR_SUB" > "$TEMP_SUB"

    echo "HTCondor submission file: $CONDOR_SUB"
    echo "Arguments: $ARG_STRING"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] condor_submit $TEMP_SUB"
        cat "$TEMP_SUB"
        rm "$TEMP_SUB"
        exit 0
    fi

    if condor_submit "$TEMP_SUB"; then
        echo ""
        echo "========================================"
        echo "✓ Job submitted successfully"
        echo "========================================"
        echo "Monitor with: condor_q"
        echo "Logs: $PROJECT_DIR/logs"
    else
        rm "$TEMP_SUB"
        echo ""
        echo "========================================"
        echo "✗ Submission failed"
        echo "========================================"
        exit 1
    fi
    rm "$TEMP_SUB"
else
    CMD=("$PROJECT_DIR/scripts/train_electron_direction_3plane.sh" "--local" "${ARGS[@]}")

    echo "Local command: ${CMD[*]}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] ${CMD[*]}"
        exit 0
    fi

    "${CMD[@]}"
fi
