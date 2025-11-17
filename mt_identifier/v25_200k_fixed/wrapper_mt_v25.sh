#!/bin/bash
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml/mt_identifier
source ../scripts/init.sh
python3 models/train_mt_incremental.py v25_200k_fixed/config_mt_v25.json
