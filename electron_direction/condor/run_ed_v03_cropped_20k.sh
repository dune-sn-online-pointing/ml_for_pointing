#!/bin/bash

# Navigate to project root
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml

# Run ED training with cropped 3-plane volumes
python3 electron_direction/models/train_ed_cropped_3plane.py \
    -j electron_direction/json/ed_v03_cropped_20k.json
