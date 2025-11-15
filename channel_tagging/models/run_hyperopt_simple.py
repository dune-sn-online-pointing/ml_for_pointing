#!/usr/bin/env python3
"""
Simple hyperparameter optimization wrapper for CT training.
Runs multiple training iterations with different hyperparameters and tracks results.
"""

import sys
import os
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

def run_trial(config_template, trial_params, trial_idx, base_output_dir):
    """Run a single training trial with given hyperparameters."""
    
    # Create modified config for this trial
    trial_config = config_template.copy()
    
    # Update with trial parameters
    if 'learning_rate' in trial_params:
        trial_config['training']['learning_rate'] = trial_params['learning_rate']
    if 'dropout_rate' in trial_params:
        trial_config['model']['dropout_rate'] = trial_params['dropout_rate']
    if 'n_filters' in trial_params:
        trial_config['model']['n_filters'] = trial_params['n_filters']
    if 'dense_units' in trial_params:
        trial_config['model']['dense_units'] = trial_params['dense_units']
    
    # Update version for this trial
    orig_version = trial_config['version']
    trial_config['version'] = f"{orig_version}_trial{trial_idx}"
    trial_config['description'] = f"Trial {trial_idx}: {trial_params}"
    
    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(trial_config, f, indent=2)
        temp_config = f.name
    
    try:
        # Run training - use absolute paths
        script_dir = Path(__file__).parent.parent.parent.absolute()
        training_script = script_dir / 'channel_tagging' / 'models' / 'train_ct_volume_simple.py'
        
        cmd = [
            'python3',
            str(training_script),
            '-j', temp_config
        ]
        
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_idx}")
        print(f"Parameters: {trial_params}")
        print(f"{'='*80}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script_dir))
        
        # Parse output for test accuracy
        test_acc = None
        for line in result.stdout.split('\n'):
            if 'Accuracy:' in line and 'Test Results' in result.stdout[:result.stdout.index(line)]:
                try:
                    test_acc = float(line.split(':')[1].strip())
                    break
                except:
                    pass
        
        return {
            'trial': trial_idx,
            'params': trial_params,
            'test_accuracy': test_acc,
            'stdout': result.stdout[-1000:],  # Last 1000 chars
            'success': result.returncode == 0
        }
        
    finally:
        # Clean up temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_hyperopt_simple.py <config.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load base config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    hyperopt_config = config.get('hyperopt', {})
    if not hyperopt_config.get('enabled', False):
        print("ERROR: hyperopt.enabled must be true in config")
        sys.exit(1)
    
    # Define search space from config
    search_space = hyperopt_config.get('search_space', {})
    max_trials = hyperopt_config.get('max_trials', 5)
    
    # Generate trial parameter sets
    trials_params = []
    
    # Get parameter ranges
    lr_values = search_space.get('learning_rate', {}).get('values', [0.0001, 0.0003, 0.0005, 0.001])
    if 'min' in search_space.get('learning_rate', {}):
        import numpy as np
        lr_min = search_space['learning_rate']['min']
        lr_max = search_space['learning_rate']['max']
        lr_values = list(np.logspace(np.log10(lr_min), np.log10(lr_max), num=3))
    
    dropout_values = search_space.get('dropout_rate', {}).get('values', [0.2, 0.3, 0.4, 0.5])
    filters_values = search_space.get('n_filters', {}).get('values', [32, 64, 128])
    dense_values = search_space.get('dense_units', {}).get('values', [128, 256, 512])
    
    # Create trial combinations (random sample if needed)
    import itertools
    import random
    
    all_combinations = list(itertools.product(lr_values, dropout_values, filters_values, dense_values))
    
    if len(all_combinations) > max_trials:
        random.seed(42)
        selected_combinations = random.sample(all_combinations, max_trials)
    else:
        selected_combinations = all_combinations[:max_trials]
    
    for i, (lr, dropout, filters, dense) in enumerate(selected_combinations):
        trials_params.append({
            'learning_rate': lr,
            'dropout_rate': dropout,
            'n_filters': filters,
            'dense_units': dense
        })
    
    # Create output directory for hyperopt results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = config.get('version', 'unknown')
    base_output_dir = Path(config.get('output', {}).get('base_dir', 'training_output/channel_tagging'))
    hyperopt_output_dir = base_output_dir / f"{version}_hyperopt" / timestamp
    hyperopt_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Running {len(trials_params)} trials")
    print(f"Results will be saved to: {hyperopt_output_dir}")
    print(f"{'='*80}\n")
    
    # Run all trials
    results = []
    best_acc = 0.0
    best_trial = None
    
    for idx, trial_params in enumerate(trials_params):
        result = run_trial(config, trial_params, idx, hyperopt_output_dir)
        results.append(result)
        
        if result['success'] and result['test_accuracy'] is not None:
            if result['test_accuracy'] > best_acc:
                best_acc = result['test_accuracy']
                best_trial = idx
                print(f"\n🎯 New best accuracy: {best_acc:.4f}")
    
    # Save results
    results_file = hyperopt_output_dir / 'hyperopt_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_trial': best_trial,
            'best_accuracy': best_acc,
            'best_params': trials_params[best_trial] if best_trial is not None else None,
            'all_trials': results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Best trial: {best_trial}")
    print(f"Best test accuracy: {best_acc:.4f}")
    if best_trial is not None:
        print(f"Best parameters: {trials_params[best_trial]}")
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
