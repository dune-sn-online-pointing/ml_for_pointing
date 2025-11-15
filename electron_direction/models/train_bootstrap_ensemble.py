#!/usr/bin/env python3
"""
Bootstrap Ensemble Training for Three-Plane CNN
Trains multiple models on bootstrap samples and averages predictions.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import tensorflow as tf
from tensorflow import keras

# Import data loader and model
import python.matched_three_plane_data_loader as data_loader
from electron_direction.models.hyperopt_three_plane_attention import create_model


def create_bootstrap_sample(images_u, images_v, images_x, directions, random_seed=None):
    """Create a bootstrap sample by sampling with replacement"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(images_u)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    return (
        images_u[indices],
        images_v[indices],
        images_x[indices],
        directions[indices]
    )


def main():
    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python train_bootstrap_ensemble.py <config.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    n_models = config.get('n_ensemble_models', 5)
    
    print("="*70)
    print("BOOTSTRAP ENSEMBLE TRAINING")
    print("="*70)
    print(f"Config: {config_file}")
    print(f"Number of models: {n_models}")
    print(f"Epochs per model: {config.get('epochs', 100)}")
    
    # Load data
    data_file = config.get('data_file') or config.get('matched_data_file')
    (train_u, train_v, train_x, train_y), \
    (val_u, val_v, val_x, val_y), \
    (test_u, test_v, test_x, test_y) = data_loader.load_matched_three_plane_data(
        matched_file_path=data_file,
        train_fraction=config.get('train_fraction', 0.7),
        val_fraction=config.get('val_fraction', 0.15),
        shuffle=True,
        random_seed=42,
        min_energy=config.get('min_energy', None)
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('model_name', 'bootstrap_ensemble')
    output_dir = os.path.join(
        config.get('output_dir', 'training_output/electron_direction'),
        model_name,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model parameters
    params = config.get('model_params', {
        'n_conv_layers': 3,
        'n_filters': 32,
        'kernel_size': 5,
        'n_dense_layers': 2,
        'dense_units': 512,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    })
    
    # Train ensemble
    models = []
    val_losses = []
    
    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL {i+1}/{n_models}")
        print(f"{'='*70}")
        
        # Create bootstrap sample from training data
        boot_u, boot_v, boot_x, boot_y = create_bootstrap_sample(
            train_u, train_v, train_x, train_y,
            random_seed=42 + i
        )
        
        print(f"Bootstrap sample: {len(boot_u)} samples")
        
        # Create model
        model = create_model(
            params,
            input_shape=(128, 16, 1),
            output_dim=3,
            loss_function=config.get('loss_function', 'cosine')
        )
        
        # Callbacks
        model_dir = os.path.join(output_dir, f'model_{i+1}')
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 15),
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            [boot_u, boot_v, boot_x],
            boot_y,
            validation_data=([val_u, val_v, val_x], val_y),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=2
        )
        
        best_val_loss = min(history.history['val_loss'])
        val_losses.append(best_val_loss)
        
        print(f"✓ Model {i+1} trained - Best val_loss: {best_val_loss:.4f}")
        
        # Save model
        model.save(os.path.join(model_dir, 'final_model.keras'))
        models.append(model)
    
    # Ensemble prediction on test set
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION ON TEST SET")
    print("="*70)
    
    all_predictions = []
    
    for i, model in enumerate(models):
        pred = model.predict([test_u, test_v, test_x], verbose=0)
        all_predictions.append(pred)
        
        # Individual model performance
        dot_products = np.sum(pred * test_y, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angular_errors = np.arccos(dot_products) * 180.0 / np.pi
        
        print(f"Model {i+1}: {angular_errors.mean():.2f}° mean, {np.median(angular_errors):.2f}° median")
    
    # Average predictions (ensemble)
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Normalize ensemble predictions
    pred_norm = ensemble_predictions / (np.linalg.norm(ensemble_predictions, axis=1, keepdims=True) + 1e-8)
    
    dot_products = np.sum(pred_norm * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE (Average of {n_models} models):")
    print(f"{'='*70}")
    print(f"  Mean:   {angular_errors.mean():.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {angular_errors.std():.2f}°")
    print(f"  25th:   {np.percentile(angular_errors, 25):.2f}°")
    print(f"  75th:   {np.percentile(angular_errors, 75):.2f}°")
    
    # Save results
    results = {
        'n_models': n_models,
        'individual_val_losses': [float(v) for v in val_losses],
        'mean_val_loss': float(np.mean(val_losses)),
        'ensemble_test_angular_errors': {
            'mean': float(angular_errors.mean()),
            'median': float(np.median(angular_errors)),
            'std': float(angular_errors.std()),
            'q25': float(np.percentile(angular_errors, 25)),
            'q75': float(np.percentile(angular_errors, 75))
        },
        'config': config,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    np.savez(
        os.path.join(output_dir, 'ensemble_predictions.npz'),
        ensemble_predictions=pred_norm,
        individual_predictions=np.array(all_predictions),
        true_directions=test_y,
        angular_errors=angular_errors
    )
    
    print(f"\n✓ Ensemble results saved to: {output_dir}")
    print(f"✓ Ensemble angular error: {angular_errors.mean():.2f}° (mean), {np.median(angular_errors):.2f}° (median)")


if __name__ == '__main__':
    main()
