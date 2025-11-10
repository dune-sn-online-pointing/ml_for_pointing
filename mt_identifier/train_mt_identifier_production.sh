#!/bin/bash
#
# Train Main Track Identifier (Production)
# Wrapper script for GPU cluster training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source initialization script
source "$PROJECT_ROOT/scripts/init.sh"

# Default values
JSON_FILE=""
OUTPUT_DIR=""
PLANE=""

# Print usage
print_usage() {
    echo "Usage: $0 -j <json_config> [-o <output_dir>] [--plane <U|V|X>]"
    echo ""
    echo "Options:"
    echo "  -j, --json <file>       JSON configuration file (required)"
    echo "  -o, --output <dir>      Override output directory"
    echo "  --plane <U|V|X>         Override plane selection"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -j json/mt_identifier/production_training.json"
    echo "  $0 -j json/mt_identifier/production_training.json --plane X"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--json)
            JSON_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plane)
            PLANE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$JSON_FILE" ]]; then
    echo "Error: JSON configuration file required (-j)"
    print_usage
fi

if [[ ! -f "$JSON_FILE" ]]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

# Build command
CMD="python3 $PROJECT_ROOT/mt_identifier/main_production.py --json $JSON_FILE"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

if [[ -n "$PLANE" ]]; then
    CMD="$CMD --plane $PLANE"
fi

echo "="
echo "Starting Main Track Identifier Training"
echo "========================================"
echo "JSON Config: $JSON_FILE"
echo "Project Root: $PROJECT_ROOT"
echo ""
echo "Command: $CMD"
echo "========================================"
echo ""

# Run training
exec $CMD
