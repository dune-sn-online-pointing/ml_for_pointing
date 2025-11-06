#!/bin/bash

set -e
export SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPTS_DIR/init.sh

print_help(){
  echo "Usage: $0 -j <config.json> [options]"
  echo "Train the Main Track Identifier neural network"
  echo ""
  echo "Required arguments:"
  echo "  -j|--json <file>            JSON configuration file"
  echo ""
  echo "Optional arguments:"
  echo "  -o|--output <dir>           Override output directory from JSON"
  echo "  -d|--data <dir>             Override data directory from JSON"
  echo "  --plane <U|V|X>             Override plane selection (default: X)"
  echo "  --max-samples <N>           Limit number of samples for testing"
  echo "  -v|--verbose                Enable verbose output"
  echo "  -h|--help                   Print this help message"
  echo ""
  echo "Examples:"
  echo "  # Train with configuration file"
  echo "  $0 -j json/mt_identifier/basic_training.json"
  echo ""
  echo "  # Quick test with limited samples"
  echo "  $0 -j json/mt_identifier/basic_training.json --max-samples 1000"
  echo ""
  echo "  # Override output directory"
  echo "  $0 -j json/mt_identifier/basic_training.json -o /path/to/output"
  exit 0
}

# Default values
json_file=""
output_dir=""
data_dir=""
plane=""
max_samples=""
verbose=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json) json_file="$2"; shift 2;;
    -o|--output) output_dir="$2"; shift 2;;
    -d|--data) data_dir="$2"; shift 2;;
    --plane) plane="$2"; shift 2;;
    --max-samples) max_samples="$2"; shift 2;;
    -v|--verbose) verbose=true; shift;;
    -h|--help) print_help;;
    *) echo "Unknown option: $1"; print_help;;
  esac
done

# Validate required arguments
if [ -z "$json_file" ]; then
  print_error "JSON configuration file is required (-j|--json)"
  echo ""
  print_help
fi

# Check if JSON file exists
check_file "$json_file"

# Get absolute path to JSON file
json_file=$(realpath "$json_file")

print_info "Starting Main Track Identifier Training"
print_info "Configuration: $json_file"

# Build Python command arguments
python_args="--input_json $json_file"

if [ -n "$output_dir" ]; then
  ensure_dir "$output_dir"
  python_args="$python_args --output_folder $output_dir"
  print_info "Output directory: $output_dir"
fi

if [ -n "$data_dir" ]; then
  check_dir "$data_dir"
  python_args="$python_args --data_dir $data_dir"
  print_info "Data directory: $data_dir"
fi

if [ -n "$plane" ]; then
  python_args="$python_args --plane $plane"
  print_info "Plane: $plane"
fi

if [ -n "$max_samples" ]; then
  python_args="$python_args --max_samples $max_samples"
  print_info "Max samples: $max_samples"
fi

if [ "$verbose" = true ]; then
  python_args="$python_args --verbose"
fi

# Change to mt_identifier directory and run the training
cd "$REPO_DIR/mt_identifier"

print_info "Running training script..."
print_info "Command: python3 mt_training.py $python_args"
echo ""

# Run the Python training script
python3 mt_training.py $python_args

exit_code=$?

if [ $exit_code -eq 0 ]; then
  print_success "Training completed successfully!"
else
  print_error "Training failed with exit code $exit_code"
  exit $exit_code
fi

# Return to original directory
cd "$SCRIPTS_DIR"
