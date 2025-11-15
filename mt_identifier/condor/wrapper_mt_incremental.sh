#!/bin/bash
# Wrapper script for MT incremental training

# Parse arguments
while getopts "j:" opt; do
  case $opt in
    j) JSON_CONFIG="$OPTARG";;
  esac
done

# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Set working directory
cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml

# Set PYTHONPATH
export PYTHONPATH=/afs/cern.ch/work/e/evilla/private/dune/refactor_ml/python:$PYTHONPATH

# Run training
python3 mt_identifier/models/train_mt_incremental.py -j $JSON_CONFIG
