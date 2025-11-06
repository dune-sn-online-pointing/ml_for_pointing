#!/bin/bash
# 3-Plane Direction Regressor Training Script
# Wrapper for ed3p_training.py with environment setup

set -e

# Source initialization
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/init.sh"

# Default configuration
JSON_CONFIG="json/direction_3plane/production_training.json"
MAX_SAMPLES=""
OUTPUT_DIR=""

print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Train the 3-plane electron direction regressor.

Options:
    -j, --json <file>           JSON configuration file (default: $JSON_CONFIG)
    --max-samples <N>           Limit training samples (for testing)
    -o, --output <dir>          Override output directory
    -h, --help                  Show this help message

Examples:
    # Full training with default config
    $0

    # Use specific config
    $0 -j json/direction_3plane/production_training.json

    # Quick test with limited samples
    $0 --max-samples 5000

