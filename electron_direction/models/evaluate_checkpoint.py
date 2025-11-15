#!/usr/bin/env python3
"""
Evaluate a saved checkpoint and generate predictions for plotting.
This is useful for runs that completed without saving predictions.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

from data_loader import load_three_plane_matched

# Import custom losses
sys.path.insert(0, str(project_root / 'electron_direction' / 'models'))
from direction_losses import angular_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint and save predictions')
    parser.add_argument('-c', '--checkpoint', required=True, help='Path to checkpoint .keras file')
    parser.add_argument('-j', '--json', required=True, help='Path to original config JSON')
    parser.add_argument('-o', '--output', help='Output directory (defaults to checkpoint dir parent)')
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.json)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_path.parent.parent
    
    print("=" * 70)
    print("CHECKPOINT EVALUATION - GENERATE PREDICTIONS")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_config = config['data']
    
    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    print(f"Loading three-plane matched data...")
    data_dir = data_config['data_directories'][0]  # Use first directory
    images_u, images_v, images_x, metadata = load_three_plane_matched(
        data_dir=data_dir,
        max_samples=data_config.get('max_samples', None),
        shuffle=False,  # Don't shuffle for evaluation
        verbose=True
    )
    
    print(f"✓ Loaded {len(images_u)} samples")
    
    # Get directions from metadata (fields 7-9)
    directions = metadata[:, 7:10].astype(np.float32)
    
    # Normalize directions
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / (dir_norms + 1e-8)
    
    # Split data
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    
    n_samples = len(images_u)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Use validation set for evaluation
    val_u = images_u[n_train:n_train+n_val]
    val_v = images_v[n_train:n_train+n_val]
    val_x = images_x[n_train:n_train+n_val]
    val_y = directions[n_train:n_train+n_val]
    val_metadata = metadata[n_train:n_train+n_val]
    
    print(f"Validation set: {len(val_u)} samples")
    print()
    
    # Load model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Loading checkpoint: {checkpoint_path.name}")
    
    # Load with custom objects
    model = load_model(checkpoint_path, custom_objects={'angular_loss': angular_loss})
    print(f"✓ Model loaded successfully")
    print()
    
    # Make predictions
    print("=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    
    predictions = model.predict([val_u, val_v, val_x], batch_size=32, verbose=1)
    
    # Normalize predictions
    pred_norms = np.linalg.norm(predictions, axis=1, keepdims=True)
    predictions = predictions / (pred_norms + 1e-8)
    
    # Calculate angular errors
    dot_products = np.sum(predictions * val_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {np.mean(angular_errors):.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {np.std(angular_errors):.2f}°")
    print(f"  25th:   {np.percentile(angular_errors, 25):.2f}°")
    print(f"  68th:   {np.percentile(angular_errors, 68):.2f}°")
    print(f"  75th:   {np.percentile(angular_errors, 75):.2f}°")
    print()
    
    # Extract energy from validation metadata for likelihood building
    # Column 10 contains true_particle_energy in MeV
    offset = 1 if val_metadata.shape[1] == 12 else 0
    val_energies = val_metadata[:, 10 + offset].astype(np.float32)
    
    # Save predictions
    predictions_file = output_dir / "val_predictions.npz"
    np.savez(predictions_file,
             predictions=predictions,
             true_directions=val_y,
             angular_errors=angular_errors,
             energies=val_energies,  # Energy in MeV for likelihood building
             metadata=val_metadata)  # Full metadata for additional analysis
    
    print(f"✓ Predictions saved to: {predictions_file}")
    print(f"  Included: predictions, true_directions, angular_errors, energies, metadata")
    print(f"✓ Ready for plotting with analyze_results_flexible.py")
    print()

if __name__ == '__main__':
    main()
