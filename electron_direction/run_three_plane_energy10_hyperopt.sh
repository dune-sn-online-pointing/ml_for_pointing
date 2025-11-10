#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing
python electron_direction/train_three_plane_hyperopt.py --input_json json/electron_direction/three_plane_matched_v4_energy10_hyperopt.json
