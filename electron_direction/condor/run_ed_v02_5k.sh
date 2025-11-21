#!/bin/bash

# Navigate to project root
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml

# Run ED volume batch reload training with all three planes
python3 electron_direction/models/train_ed_volume_batch_reload.py \
    -j electron_direction/json/ed_volumes_v02_5k.json \
    --plane all \
    --reload-epochs 5
