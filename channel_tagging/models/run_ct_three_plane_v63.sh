#!/bin/bash
source /afs/cern.ch/work/e/evilla/private/dune/refactor_ml/scripts/init.sh --quiet
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml/channel_tagging/models
python train_ct_three_plane_batch_reload.py --json ../json/volume_v63_three_plane_batch_reload.json
