"""
Enhanced 3-plane electron direction training with advanced features:
- Deeper architectures with spatial dropout
- Cosine annealing LR schedule
- Multi-task learning (direction + energy)
- Save checkpoints every epoch
- Better augmentation support
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Import custom modules
import data_loader
from three_plane_cnn_enhanced import (
    build_three_plane_cnn_enhanced,
    get_callbacks
)

def main():
    parser = argparse.ArgumentParser(description='Train 3-plane electron direction CNN (enhanced)')
    parser.add_argument('-j', '--json', type=str, required=True,
                       help='JSON configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.json, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("THREE-PLANE ELECTRON DIRECTION TRAINING (ENHANCED)")
    print("=" * 70)
    print(f"Config: {args.json}")
    print(f"Version: {config['model'].get('name', 'unknown')}")
    print()
    
    # Extract config parameters
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    output_config = config['output']
    
    # Data directory
    data_dir = data_config['data_directories'][0]
    max_samples = data_config.get('max_samples', None)
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    shuffle = data_config.get('shuffle', True)
    
    # Augmentation settings
    augmentation_config = data_config.get('augmentation', {})
    use_augmentation = augmentation_config.get('enabled', False)
    
    # Model parameters
    input_shape = tuple(model_config['input_shape'])
    output_dim = model_config['output_dim']
    n_conv_layers = model_config['n_conv_layers']
    n_filters = model_config['n_filters']
    kernel_size = model_config.get('kernel_size', 3)
    n_dense_layers = model_config['n_dense_layers']
    n_dense_units = model_config['n_dense_units']
    use_batch_norm = model_config.get('use_batch_norm', True)
    spatial_dropout = model_config.get('spatial_dropout', 0.0)
    dense_dropout = model_config.get('dense_dropout', 0.3)
    
    # Multi-task learning
    multitask = model_config.get('multitask', False)
    energy_output = model_config.get('auxiliary_tasks', [])
    has_energy_output = 'energy' in energy_output if isinstance(energy_output, list) else multitask
    
    # Training parameters
    loss = training_config.get('loss', 'angular_loss')
    learning_rate = training_config['learning_rate']
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']
    early_stopping_patience = training_config.get('early_stopping_patience', 30)
    
    # Learning rate schedule
    use_cosine_annealing = training_config.get('lr_schedule', '') == 'cosine_annealing'
    cosine_t0 = training_config.get('cosine_annealing_t0', 50)
    cosine_t_mult = training_config.get('cosine_annealing_tmult', 2)
    reduce_lr_patience = training_config.get('reduce_lr_patience', 10)
    reduce_lr_factor = training_config.get('reduce_lr_factor', 0.5)
    reduce_lr_min_lr = training_config.get('reduce_lr_min_lr', 1e-6)
    
    # Checkpoint saving
    save_every_epoch = training_config.get('save_every_epoch', True)
    save_predictions = training_config.get('save_predictions', True)
    
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Max samples: {max_samples}")
    if use_augmentation:
        print(f"Augmentation: ENABLED")
        print(f"  Flip X prob: {augmentation_config.get('prob_flip_x', 0.5)}")
        print(f"  Flip Y prob: {augmentation_config.get('prob_flip_y', 0.5)}")
        print(f"  Rotation: ±{augmentation_config.get('rotation_range', 0)}°")
        print(f"  Zoom: ±{augmentation_config.get('zoom_range', 0)*100}%")
    print()
    
    # Load 3-plane matched data
    images_u, images_v, images_x, metadata = data_loader.load_three_plane_matched(
        data_dir,
        max_samples=max_samples
    )
    


    # Extract direction labels from metadata
    directions = data_loader.extract_direction_labels(metadata)
    if shuffle:
        print("Shuffling dataset...")
        indices = np.random.permutation(len(images_u))
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        directions = directions[indices]
        metadata = metadata[indices]
    
    print(f"\nData shapes:")
    print(f"  U images: {images_u.shape}")
    print(f"  V images: {images_v.shape}")
    print(f"  X images: {images_x.shape}")
    print(f"  Directions: {directions.shape}")
    
    # Add channel dimension if needed
    if len(images_u.shape) == 3:
        images_u = np.expand_dims(images_u, axis=-1)
        images_v = np.expand_dims(images_v, axis=-1)
        images_x = np.expand_dims(images_x, axis=-1)
        print(f"\nAdded channel dimension:")
        print(f"  New shapes: {images_u.shape}, {images_v.shape}, {images_x.shape}")
    
    # Split into train/val/test
    n_total = len(images_u)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_u = images_u[:n_train]
    train_v = images_v[:n_train]
    train_x = images_x[:n_train]
    train_y = directions[:n_train]
    train_metadata = metadata[:n_train]
    
    val_u = images_u[n_train:n_train+n_val]
    val_v = images_v[n_train:n_train+n_val]
    val_x = images_x[n_train:n_train+n_val]
    val_y = directions[n_train:n_train+n_val]
    val_metadata = metadata[n_train:n_train+n_val]
    
    test_u = images_u[n_train+n_val:]
    test_v = images_v[n_train+n_val:]
    test_x = images_x[n_train+n_val:]
    test_y = directions[n_train+n_val:]
    test_metadata = metadata[n_train+n_val:]
    
    print(f"\nTrain/Val/Test split:")
    print(f"  Train: {len(train_u)} samples ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_u)} samples ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_u)} samples ({test_split*100:.0f}%)")
    
    # Extract energies for multi-task learning if needed
    train_energies = None
    val_energies = None
    test_energies = None
    
    if has_energy_output:
        # Column 10 contains true_particle_energy in MeV
        offset = 1 if train_metadata.shape[1] == 12 else 0
        train_energies = train_metadata[:, 10 + offset].astype(np.float32)
        val_energies = val_metadata[:, 10 + offset].astype(np.float32)
        test_energies = test_metadata[:, 10 + offset].astype(np.float32)
        
        # Normalize energies (helps with training stability)
        energy_mean = np.mean(train_energies)
        energy_std = np.std(train_energies)
        train_energies = (train_energies - energy_mean) / energy_std
        val_energies = (val_energies - energy_mean) / energy_std
        test_energies = (test_energies - energy_mean) / energy_std
        
        print(f"\nEnergy statistics (training set):")
        print(f"  Mean: {energy_mean:.2f} MeV")
        print(f"  Std:  {energy_std:.2f} MeV")
        print(f"  Range: [{np.min(train_metadata[:, 10 + offset]):.1f}, {np.max(train_metadata[:, 10 + offset]):.1f}] MeV")
    
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING MODEL")
    print("=" * 70)
    print(f"Architecture: Enhanced 3-plane CNN")
    print(f"  Conv layers per branch: {n_conv_layers}")
    print(f"  Filters (first layer): {n_filters}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Dense layers: {n_dense_layers}")
    print(f"  Dense units: {n_dense_units}")
    print(f"  Batch norm: {use_batch_norm}")
    print(f"  Spatial dropout: {spatial_dropout}")
    print(f"  Dense dropout: {dense_dropout}")
    if has_energy_output:
        print(f"  Multi-task: Energy prediction enabled")
    print(f"  Loss: {loss}")
    print(f"  Learning rate: {learning_rate}")
    if use_cosine_annealing:
        print(f"  LR schedule: Cosine annealing (T0={cosine_t0}, Tmult={cosine_t_mult})")
    else:
        print(f"  LR schedule: ReduceLROnPlateau (patience={reduce_lr_patience}, factor={reduce_lr_factor})")
    print()
    
    # Build model
    model = build_three_plane_cnn_enhanced(
        input_shape=input_shape,
        output_dim=output_dim,
        n_conv_layers=n_conv_layers,
        n_filters=n_filters,
        kernel_size=kernel_size,
        n_dense_layers=n_dense_layers,
        n_dense_units=n_dense_units,
        spatial_dropout=spatial_dropout,
        dense_dropout=dense_dropout,
        use_batch_norm=use_batch_norm,
        learning_rate=learning_rate,
        multitask=has_energy_output
    )
    
    # Setup loss functions
    import tensorflow as tf
    from tensorflow import keras
    
    if loss == 'cosine_similarity':
        loss_fn = keras.losses.CosineSimilarity(axis=1)
    elif loss == 'angular_loss':
        sys.path.insert(0, os.path.dirname(__file__))
        from direction_losses import angular_loss
        loss_fn = angular_loss
    elif loss == 'focal_angular_loss':
        from direction_losses import focal_angular_loss
        loss_fn = focal_angular_loss
    elif loss == 'hybrid_loss':
        from direction_losses import hybrid_loss
        loss_fn = hybrid_loss
    else:
        raise ValueError(f"Unknown loss: {loss}")
    
    # Compile model
    if has_energy_output:
        # Multi-task: direction + energy
        energy_loss_weight = model_config.get('energy_loss_weight', 0.2)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
            loss={'direction_output': loss_fn, 'energy_output': 'mse'},
            loss_weights={'direction_output': 1.0, 'energy_output': energy_loss_weight},
            metrics={'direction_output': ['mae'], 'energy_output': ['mae']}
        )
        print(f"✓ Multi-task model compiled (energy loss weight: {energy_loss_weight})")
    else:
        # Single-task: direction only
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
            loss=loss_fn,
            metrics=['mae']
        )
        print(f"✓ Model compiled with {loss}")
    
    print(f"\nModel summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING")
    print("=" * 70)
    
    # Create output directory
    version = config['model'].get('name', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_config['base_dir']) / f"three_plane_{version}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_file}")
    
    # Get callbacks
    callbacks = get_callbacks(
        output_dir=str(output_dir),
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_min_lr=reduce_lr_min_lr,
        use_cosine_annealing=use_cosine_annealing,
        cosine_t0=cosine_t0,
        cosine_t_mult=cosine_t_mult,
        initial_lr=learning_rate,
        save_every_epoch=save_every_epoch
    )
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Saving checkpoints: {'every epoch' if save_every_epoch else 'best only'}")
    print()
    
    # Prepare training data
    train_inputs = [train_u, train_v, train_x]
    val_inputs = [val_u, val_v, val_x]
    
    if has_energy_output:
        train_outputs = {'direction_output': train_y, 'energy_output': train_energies}
        val_outputs = {'direction_output': val_y, 'energy_output': val_energies}
    else:
        train_outputs = train_y
        val_outputs = val_y
    
    # Train
    history = model.fit(
        train_inputs,
        train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATION")
    print("=" * 70)
    
    # Evaluate on validation set
    val_predictions = model.predict([val_u, val_v, val_x])
    
    # Extract direction predictions (for multi-task, first output is direction)
    if has_energy_output:
        val_pred_directions = val_predictions[0]
        val_pred_energies = val_predictions[1]
    else:
        val_pred_directions = val_predictions
        val_pred_energies = None
    
    # Normalize direction predictions
    pred_norms = np.linalg.norm(val_pred_directions, axis=1, keepdims=True)
    val_pred_directions = val_pred_directions / (pred_norms + 1e-8)
    
    # Calculate angular errors
    dot_products = np.sum(val_pred_directions * val_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics (Validation):")
    print(f"  Mean:   {np.mean(angular_errors):.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {np.std(angular_errors):.2f}°")
    print(f"  25th:   {np.percentile(angular_errors, 25):.2f}°")
    print(f"  75th:   {np.percentile(angular_errors, 75):.2f}°")
    
    if has_energy_output:
        # Denormalize energy predictions
        val_pred_energies_denorm = val_pred_energies * energy_std + energy_mean
        val_true_energies_denorm = val_metadata[:, 10 + offset].astype(np.float32)
        
        energy_errors = np.abs(val_pred_energies_denorm.flatten() - val_true_energies_denorm)
        print(f"\nEnergy Prediction Statistics (Validation):")
        print(f"  MAE:    {np.mean(energy_errors):.2f} MeV")
        print(f"  Median: {np.median(energy_errors):.2f} MeV")
        print(f"  Std:    {np.std(energy_errors):.2f} MeV")
    
    # Extract energy from validation metadata
    offset = 1 if val_metadata.shape[1] == 12 else 0
    val_true_energies = val_metadata[:, 10 + offset].astype(np.float32)
    
    # Save predictions
    if save_predictions:
        predictions_file = output_dir / "val_predictions.npz"
        save_dict = {
            'predictions': val_pred_directions,
            'true_directions': val_y,
            'angular_errors': angular_errors,
            'energies': val_true_energies,
            'metadata': val_metadata
        }
        
        if has_energy_output:
            save_dict['predicted_energies'] = val_pred_energies_denorm
        
        np.savez(predictions_file, **save_dict)
        print(f"\n✓ Predictions saved to: {predictions_file}")
    
    # Save results
    results = {
        'config': config,
        'angular_error_mean': float(np.mean(angular_errors)),
        'angular_error_median': float(np.median(angular_errors)),
        'angular_error_std': float(np.std(angular_errors)),
        'angular_error_25th': float(np.percentile(angular_errors, 25)),
        'angular_error_75th': float(np.percentile(angular_errors, 75)),
        'history': {k: [float(v) for v in history.history[k]] for k in history.history.keys()}
    }
    
    if has_energy_output:
        results['energy_mae'] = float(np.mean(energy_errors))
        results['energy_median_error'] = float(np.median(energy_errors))
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Best angular error: {np.mean(angular_errors):.2f}° (mean), {np.median(angular_errors):.2f}° (median)")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE! ✓")
    print("=" * 70)

if __name__ == '__main__':
    main()
