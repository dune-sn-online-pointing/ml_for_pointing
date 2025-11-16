"""
Main Track Identifier Training Script

Trains a CNN to distinguish main electron tracks from background
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
# Also add project root for mt_identifier imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import general_purpose_libs as gpl
import classification_libs as cl

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Main Track Identifier neural network',
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
        default='/eos/home-e/evilla/dune/sn-tps/images_test',
        help='Data directory containing NPZ files'
    )
    
    parser.add_argument(
        '--plane',
        type=str,
        choices=['U', 'V', 'X'],
        default='X',
        help='Detector plane to use (U, V, or X)'
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
    output_folder = os.path.join(base_output, 'mt_identifier', model_name)
    
    # Add augmentation coefficient to path if > 1
    aug_coeff = config['dataset_parameters'].get('aug_coefficient', 1)
    if aug_coeff > 1:
        output_folder = os.path.join(output_folder, f"aug_coeff_{aug_coeff}")
    
    # Add plane information
    output_folder = os.path.join(output_folder, f"plane_{args.plane}")
    
    # Add timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, timestamp)
    
    # Create directory
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nOutput directory: {output_folder}")
    
    return output_folder


def select_model(model_name):
    """Import and return the selected model module"""
    
    if model_name == 'simple_cnn':
        import mt_identifier.models.simple_cnn as selected_model
    elif model_name == 'hyperopt_simple_cnn':
        import mt_identifier.models.hyperopt_simple_cnn as selected_model
    elif model_name == 'hyperopt_simple_cnn_multiclass':
        import mt_identifier.models.hyperopt_simple_cnn_multiclass as selected_model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Selected model: {model_name}")
    return selected_model


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("MAIN TRACK IDENTIFIER TRAINING")
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
    
    # Use data_directories from JSON config if available, otherwise use command line arg
    if "data_directories" in config and config["data_directories"]:
        data_dirs = config["data_directories"] if isinstance(config["data_directories"], list) else [config["data_directories"]]
        print(f"Using {len(data_dirs)} data director{'ies' if len(data_dirs) > 1 else 'y'} from JSON config:")
        for data_dir in data_dirs:
            print(f"  - {data_dir}")
    else:
        data_dirs = [args.data_dir]
        print(f"Using data directory from command line: {args.data_dir}")
    
    # Prepare data from NPZ files
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    # Load from all directories and concatenate
    all_train = []
    all_validation = []
    all_test = []
    
    for i, data_dir in enumerate(data_dirs):
        print(f"\nLoading from directory {i+1}/{len(data_dirs)}: {data_dir}")
        train, validation, test = cl.prepare_data_from_npz(
            data_dir=data_dir,
            plane=args.plane,
            dataset_parameters=dataset_parameters,
            output_folder=output_folder
        )
        all_train.append(train)
        all_validation.append(validation)
        all_test.append(test)
    
    # Concatenate data from all directories
    if len(all_train) > 1:
        print(f"\nConcatenating data from {len(data_dirs)} directories...")
        import numpy as np
        train = (
            np.concatenate([t[0] for t in all_train], axis=0),
            np.concatenate([t[1] for t in all_train], axis=0)
        )
        validation = (
            np.concatenate([v[0] for v in all_validation], axis=0),
            np.concatenate([v[1] for v in all_validation], axis=0)
        )
        test = (
            np.concatenate([te[0] for te in all_test], axis=0),
            np.concatenate([te[1] for te in all_test], axis=0)
        )
        print(f"Total training samples: {train[0].shape[0]}")
        print(f"Total validation samples: {validation[0].shape[0]}")
        print(f"Total test samples: {test[0].shape[0]}")
    else:
        train, validation, test = all_train[0], all_validation[0], all_test[0]

    
    # Ensure batch_size and epochs are in model_parameters
    if "batch_size" not in model_parameters and "batch_size" in dataset_parameters:
        model_parameters["batch_size"] = dataset_parameters["batch_size"]
    if "epochs" not in model_parameters:
        model_parameters["epochs"] = model_parameters.get("epochs", 50)

    # Create and train model
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    model, history = selected_model.create_and_train_model(
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
    
    evaluation = cl.test_model(
        model,
        test,
        output_folder,
        label_names=["Background", "Main Track"]
    )
    
    print("\n✓ Model evaluation completed")

    history_dict = gpl.history_to_serializable(history)
    results_payload = {
        "config": {
            "model": {
                "name": model_name
            },
            "task_label": config.get("task_label", "mt_identifier"),
            "dataset": {
                "plane": args.plane,
                "train_fraction": dataset_parameters.get("train_fraction"),
                "val_fraction": dataset_parameters.get("val_fraction"),
                "test_fraction": dataset_parameters.get("test_fraction"),
                "balance_data": dataset_parameters.get("balance_data", False),
                "max_samples": dataset_parameters.get("max_samples"),
                "data_directories": data_dirs
            },
            "training": {
                "epochs": model_parameters.get("epochs"),
                "batch_size": model_parameters.get("batch_size"),
                "learning_rate": model_parameters.get("learning_rate"),
                "decay_rate": model_parameters.get("decay_rate")
            }
        },
        "metrics": evaluation.get("metrics", {}),
        "history": history_dict,
        "artifacts": evaluation.get("artifacts", {})
    }
    gpl.write_results_json(output_folder, results_payload)
    
    # Create final report
    print("\n" + "="*70)
    print("STEP 5: GENERATING REPORT")
    print("="*70)
    
    try:
        model_source = inspect.getsource(selected_model.create_and_train_model)
    except:
        model_source = "Source code not available"
    
    #     gpl.create_report(
    #         output_folder,
    #         model_name,
    #         config,
    #         model_source
    #     )
    
    print(f"✓ Training completed. Results saved in: {output_folder}")
    
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
    print(f"  - samples/ (sample images)")
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
