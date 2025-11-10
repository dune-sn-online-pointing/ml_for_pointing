#!/bin/bash
# Wrapper script for single-plane X training

cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing

# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Run training
python3 electron_direction/train_single_plane_x.py \
    --input_json json/electron_direction/single_plane_x_filtered_v1.json
