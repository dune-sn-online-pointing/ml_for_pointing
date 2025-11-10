#!/bin/bash
#
# Submit Channel Tagging Training to GPU
# Supports both HTCondor and direct SSH submission
#

set -e

# Configuration
GPU_NODE="lxplus-gpu.cern.ch"
PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
JSON_CONFIG="json/channel_tagging/production_training.json"

# Parse arguments
PLANE="X"
MAX_SAMPLES=""
OUTPUT_DIR=""
DRY_RUN=false
USE_CONDOR=true

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit channel tagging classifier training to GPU.

Options:
    --plane <U|V|X>         Detector plane (default: X)
    --max-samples <N>       Limit number of samples (for testing)
    --output <dir>          Override output directory
    --condor                Use HTCondor batch system (default)
    --ssh                   Use direct SSH to lxplus-gpu
    --dry-run               Print command without executing
    -h, --help              Show this help message

Examples:
    # Submit via HTCondor (recommended for long jobs)
    $0 --condor

    # Direct SSH execution (for quick tests)
    $0 --ssh --max-samples 5000

    # Use plane V
    $0 --plane V

    # Dry run to see the command
    $0 --dry-run

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plane)
            PLANE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --json|-j)
            JSON_CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --condor)
            USE_CONDOR=true
            shift
            ;;
        --ssh)
            USE_CONDOR=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            ;;
    esac
done

# Display configuration
echo "========================================"
echo "Channel Tagging GPU Training Submission"
echo "========================================"
echo "Method: $([ "$USE_CONDOR" = true ] && echo "HTCondor" || echo "SSH to $GPU_NODE")"
echo "Project: $PROJECT_DIR"
echo "Config: $JSON_CONFIG"
echo "Plane: $PLANE"
[[ -n "$MAX_SAMPLES" ]] && echo "Max Samples: $MAX_SAMPLES"
[[ -n "$OUTPUT_DIR" ]] && echo "Output Dir: $OUTPUT_DIR"
echo "========================================"
echo ""

if [[ ! -f "$JSON_CONFIG" ]]; then
    echo "Error: JSON configuration file not found: $JSON_CONFIG"
    exit 1
fi

if [ "$USE_CONDOR" = true ]; then
    # HTCondor submission
    CONDOR_SUB="$PROJECT_DIR/scripts/submit_condor_channel_tagging.sub"
    
    if [ ! -f "$CONDOR_SUB" ]; then
        echo "Error: HTCondor submission file not found: $CONDOR_SUB"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Create temporary submission file with parameters
    TEMP_SUB=$(mktemp)
    cat "$CONDOR_SUB" \
        | sed "s/^plane.*=.*/plane = $PLANE/" \
        | sed "s/^max_samples.*=.*/max_samples = $MAX_SAMPLES/" \
        | sed "s|^json_config.*=.*|json_config = $JSON_CONFIG|" > "$TEMP_SUB"
    
    echo "HTCondor submission file:"
    echo "  $CONDOR_SUB"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "condor_submit $TEMP_SUB"
        cat "$TEMP_SUB"
        rm "$TEMP_SUB"
        exit 0
    fi
    
    echo "Submitting to HTCondor..."
    condor_submit "$TEMP_SUB"
    EXIT_CODE=$?
    rm "$TEMP_SUB"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "✓ Job submitted successfully!"
        echo "========================================"
        echo ""
        echo "Monitor your job with:"
        echo "  condor_q"
        echo ""
        echo "Check logs in:"
        echo "  $PROJECT_DIR/logs/"
        echo ""
        echo "Hold/release job:"
        echo "  condor_hold <job_id>"
        echo "  condor_release <job_id>"
        echo ""
        echo "Remove job:"
        echo "  condor_rm <job_id>"
    else
        echo ""
        echo "========================================"
        echo "✗ Submission failed with exit code $EXIT_CODE"
        echo "========================================"
        exit $EXIT_CODE
    fi
    
else
    # SSH submission (original behavior)
    REMOTE_CMD="cd $PROJECT_DIR && ./scripts/train_channel_tagging.sh -j \"$JSON_CONFIG\" --plane $PLANE"
    
    if [[ -n "$MAX_SAMPLES" ]]; then
        REMOTE_CMD="$REMOTE_CMD --max-samples $MAX_SAMPLES"
    fi
    
    if [[ -n "$OUTPUT_DIR" ]]; then
    REMOTE_CMD="$REMOTE_CMD -o \"$OUTPUT_DIR\""
    fi
    
    echo "Remote Command:"
    echo "  $REMOTE_CMD"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "ssh -Y $GPU_NODE \"$REMOTE_CMD\""
        exit 0
    fi
    
    # Confirm before proceeding
    read -p "Submit training to GPU via SSH? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    echo ""
    echo "Connecting to $GPU_NODE and starting training..."
    echo "This will keep the terminal attached. Use Ctrl+C to interrupt (training will stop)."
    echo ""
    
    # Execute on GPU node
    ssh -Y $GPU_NODE "$REMOTE_CMD"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "✓ Training completed successfully!"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "✗ Training failed with exit code $EXIT_CODE"
        echo "========================================"
        exit $EXIT_CODE
    fi
fi
