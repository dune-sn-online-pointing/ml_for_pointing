#!/bin/bash
# Wrapper script for three-plane training

cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing

# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Run training
python3 electron_direction/train_three_plane_matched.py \
    --input_json json/electron_direction/three_plane_matched_v1.json
