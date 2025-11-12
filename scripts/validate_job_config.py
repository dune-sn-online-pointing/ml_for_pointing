#!/usr/bin/env python3
"""
Validate job configuration before HTCondor submission.
Usage: python3 scripts/validate_job_config.py <config.json>
"""
import json
import sys
import os
from pathlib import Path

def validate_config(config_path):
    """Validate a job configuration file."""
    print(f"\n🔍 Validating: {config_path}")
    print("=" * 60)
    
    # Check file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Load and parse JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Valid JSON format")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return False
    
    # Check required sections
    required_sections = []
    if 'channel_tagging' in config_path:
        required_sections = ['model', 'data', 'training']
    elif 'mt_identifier' in config_path:
        required_sections = ['model_name', 'data_directories', 'dataset_parameters', 'model_parameters']
    elif 'electron_direction' in config_path:
        required_sections = ['model', 'data', 'training', 'output']
    
    missing = [s for s in required_sections if s not in config]
    if missing:
        print(f"❌ Missing required sections: {missing}")
        return False
    print(f"✅ All required sections present: {required_sections}")
    
    # Check data directories/paths
    data_dirs = []
    if 'data' in config and 'es_directory' in config['data']:
        data_dirs = [config['data']['es_directory'], config['data'].get('cc_directory')]
    elif 'data' in config and 'data_directories' in config['data']:
        data_dirs = config['data']['data_directories']
    elif 'data_directories' in config:
        data_dirs = config['data_directories']
    
    all_exist = True
    for d in data_dirs:
        if d and not os.path.exists(d):
            print(f"⚠️  Data directory not found: {d}")
            all_exist = False
    
    if all_exist and data_dirs:
        print(f"✅ All data directories exist ({len(data_dirs)} dirs)")
    
    # Summary
    print("=" * 60)
    if not missing and all_exist:
        print("✅ VALIDATION PASSED - Safe to submit")
        return True
    else:
        print("❌ VALIDATION FAILED - Fix issues before submission")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/validate_job_config.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = validate_config(config_path)
    sys.exit(0 if success else 1)
