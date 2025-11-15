#!/bin/bash
source /afs/cern.ch/work/e/evilla/private/dune/refactor_ml/scripts/init.sh
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml
python3 channel_tagging/models/train_ct_volume_hyperopt.py "$@"
