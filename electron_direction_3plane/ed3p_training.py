"""
Electron Direction Regressor Training Script (Three-Plane Version)

Trains a multi-input CNN using three detector planes (U, V, X) to predict electron direction
"""
# Standard library imports
import argparse
import json
import os
import sys
import time
import inspect

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Optional third-party (keep hp=None if not installed)
try:
    import hyperopt as hp
except Exception:
    hp = None

# Ensure local python package directory is on the path, then import project libs
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import general_purpose_libs as gpl
import regression_libs as rl

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Electron Direction Regressor with Three Planes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to JSON configuration file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Output folder (overrides JSON config)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory containing NPZ files for all three planes'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to load (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def load_configuration(json_file):
    """Load and validate JSON configuration"""
    print(f"\nLoading configuration from: {json_file}")
    
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'dataset_parameters', 'model_parameters']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in JSON: {field}")
    
    print("Configuration loaded successfully")
    return config


def setup_output_directory(config, args):
    """Setup output directory for saving models and results"""
    
    # Start with base output directory
    if args.output_folder:
        base_output = args.output_folder
    else:
        base_output = config.get('output_folder', '/eos/user/e/evilla/dune/sn-tps/neural_networks')
    
    # Add model name subdirectory
    model_name = config['model_name']
    output_folder = os.path.join(base_output, 'electron_direction_3plane', model_name)
    
    # Add augmentation coefficient to path if > 1
    aug_coeff = config['dataset_parameters'].get('aug_coefficient', 1)
    if aug_coeff > 1:
        output_folder = os.path.join(output_folder, f"aug_coeff_{aug_coeff}")
    
    # Add timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, timestamp)
    
    # Create directory
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nOutput directory: {output_folder}")
    
    return output_folder


def select_model(model_name):
    """Import and return the selected model module"""
    
    if model_name == 'd3_views_hp_simple_cnn':
        import models.d3_views_hp_simple_cnn as selected_model
    elif model_name == 'd3_views_simple_cnn':
        import models.d3_views_simple_cnn as selected_model
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'd3_views_hp_simple_cnn' or 'd3_views_simple_cnn'")
    
    print(f"Selected model: {model_name}")
    return selected_model


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("ELECTRON DIRECTION TRAINING (THREE PLANES)")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPU is available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("\n⚠ GPU is NOT available - training will use CPU")
    
    # Load configuration
    config = load_configuration(args.input_json)
    
    # Setup output directory
    output_folder = setup_output_directory(config, args)
    
    # Save configuration to output folder
    config_copy_path = os.path.join(output_folder, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_copy_path}")
    
    # Save command line arguments
    args_dict = vars(args)
    args_path = os.path.join(output_folder, 'args.json')
    with open(args_path, 'w') as f:
        args_serializable = {k: str(v) if v is not None else None for k, v in args_dict.items()}
        json.dump(args_serializable, f, indent=2)
    print(f"Arguments saved to: {args_path}")
    
    # Extract configuration
    model_name = config['model_name']
    model_parameters = config['model_parameters']
    dataset_parameters = config['dataset_parameters']
    
    # Override dataset parameters with command line arguments
    if args.max_samples:
        dataset_parameters['max_samples'] = args.max_samples
    
    # Select model
    selected_model = select_model(model_name)
    
    # Prepare data from NPZ files (three planes)
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION (THREE PLANES)")
    print("="*70)
    
    # Use data directory from args or config
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dirs = config.get('data_directories', [])
        if not data_dirs:
            raise ValueError("No data_directories specified in config")
        data_dir = data_dirs[0] if len(data_dirs) == 1 else data_dirs
    
    train, validation, test = rl.prepare_data_from_npz_3planes(
        data_dir=data_dir,
        dataset_parameters=dataset_parameters,
        output_folder=output_folder
    )
    
    # Create and train model
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    n_outputs = model_parameters.get('output_dim', 3)  # 3D direction vector
    
    model, history = selected_model.create_and_train_model(
        n_outputs=n_outputs,
        model_parameters=model_parameters,
        train=train,
        validation=validation,
        output_folder=output_folder,
        model_name=model_name
    )
    
    print("\n✓ Model training completed")
    
    # Save the model
    print("\n" + "="*70)
    print("STEP 3: SAVING MODEL")
    print("="*70)
    
    model_path = os.path.join(output_folder, f'{model_name}.h5')
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save the training history
    gpl.save_history(history, output_folder)
    print(f"✓ Training history saved")
    
    # Test the model
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    rl.test_regression_model(
        model,
        test,
        output_folder
    )
    
    print("\n✓ Model evaluation completed")
    
    # Create final report
    print("\n" + "="*70)
    print("STEP 5: GENERATING REPORT")
    print("="*70)
    
    try:
        model_source = inspect.getsource(selected_model.create_and_train_model)
    except:
        model_source = "Source code not available"
    
    gpl.create_report(
        output_folder,
        model_name,
        config,
        model_source
    )
    
    print(f"✓ Report generated in: {output_folder}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_folder}")
    print(f"Model file: {model_path}")
    print("\nFiles generated:")
    print(f"  - {model_name}.h5 (trained model)")
    print(f"  - training_history.json (training metrics)")
    print(f"  - config.json (configuration used)")
    print(f"  - args.json (command line arguments)")
    print(f"  - Various evaluation plots and metrics")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
