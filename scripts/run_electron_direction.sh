
#!bin/bash
# This script is used to run the pipeline
# INPUT_JSON=/afs/cern.ch/work/d/dapullia/public/dune/machine_learning/json/regression/hp_classification.json
OUTPUT_FOLDER=/eos/user/d/dapullia/dune/ML/directional_regression/v3-d3_simple_cnn/
INPUT_JSON=/afs/cern.ch/work/d/dapullia/public/dune/machine_learning/json/regression/basic-v3-d3_simple_cnn.json
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_json)
            INPUT_JSON="$2"
            shift 2
            ;;
        -o|--output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            shift
            ;;
    esac
done

REPO_HOME=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:$REPO_HOME/custom_python_libs/lib/python3.9/site-packages

cd ../es_tracks_dir_regressor/
python main.py --input_json $INPUT_JSON --output_folder $OUTPUT_FOLDER
cd ../scripts
