"""
Enhanced Three-Plane CNN for Electron Direction Prediction

Improvements over base version:
- Configurable dropout rates (spatial and dense)
- Support for deeper architectures
- Cosine annealing learning rate schedule
- Multi-task learning support (direction + energy)
- Better callback management
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from direction_losses import cosine_similarity_loss


def build_three_plane_cnn_enhanced(
    input_shape=(128, 32, 1),
    output_dim=3,
    n_conv_layers=4,
    n_filters=64,
    kernel_size=3,
    n_dense_layers=2,
    n_dense_units=256,
    spatial_dropout=0.0,
    dense_dropout=0.3,
    use_batch_norm=True,
    learning_rate=0.001,
    multitask=False,
    energy_output=False
):
    """
    Build an enhanced three-plane CNN model for electron direction prediction.
    
    Args:
        input_shape: Shape of each plane input (height, width, channels)
        output_dim: Output dimension (3 for x, y, z)
        n_conv_layers: Number of convolutional layers per branch
        n_filters: Number of filters in first conv layer (doubles each layer)
        kernel_size: Size of convolutional kernels
        n_dense_layers: Number of dense layers after concatenation
        n_dense_units: Number of units in dense layers
        spatial_dropout: Dropout rate after conv layers (0 to disable)
        dense_dropout: Dropout rate after dense layers
        use_batch_norm: Whether to use batch normalization
        learning_rate: Learning rate for optimizer
        multitask: Whether to include energy prediction as auxiliary task
        energy_output: Whether to add energy output head
        
    Returns:
        Compiled Keras model
    """
    
    def create_cnn_branch(name_prefix):
        """Create a single CNN branch for one plane"""
        branch_input = keras.Input(shape=input_shape, name=f'{name_prefix}_input')
        x = branch_input
        
        # Convolutional layers
        for i in range(n_conv_layers):
            # Calculate number of filters (doubles each layer)
            filters = n_filters * (2 ** i)
            
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'{name_prefix}_conv_{i+1}'
            )(x)
            
            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'{name_prefix}_pool_{i+1}'
            )(x)
            
            if use_batch_norm:
                x = layers.BatchNormalization(
                    name=f'{name_prefix}_bn_{i+1}'
                )(x)
            
            # Add spatial dropout after batch norm (if enabled)
            if spatial_dropout > 0:
                x = layers.SpatialDropout2D(
                    spatial_dropout,
                    name=f'{name_prefix}_spatial_dropout_{i+1}'
                )(x)
        
        # Flatten
        x = layers.Flatten(name=f'{name_prefix}_flatten')(x)
        
        return branch_input, x
    
    # Create three branches
    input_u, features_u = create_cnn_branch('plane_u')
    input_v, features_v = create_cnn_branch('plane_v')
    input_x, features_x = create_cnn_branch('plane_x')
    
    # Concatenate all features
    concatenated = layers.Concatenate(name='concatenate')([features_u, features_v, features_x])
    
    # Dense layers (shared representation)
    x = concatenated
    for i in range(n_dense_layers):
        x = layers.Dense(
            n_dense_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        
        if dense_dropout > 0:
            x = layers.Dropout(dense_dropout, name=f'dropout_{i+1}')(x)
    
    # Output layer for direction
    direction_output = layers.Dense(
        output_dim, 
        activation='linear', 
        name='direction_output'
    )(x)
    
    # Optional: Add energy prediction head for multi-task learning
    outputs = [direction_output]
    output_names = ['direction_output']
    
    if multitask or energy_output:
        # Separate head for energy prediction
        energy_head = layers.Dense(64, activation='relu', name='energy_dense')(x)
        energy_head = layers.Dropout(0.2, name='energy_dropout')(energy_head)
        energy_output = layers.Dense(1, activation='linear', name='energy_output')(energy_head)
        outputs.append(energy_output)
        output_names.append('energy_output')
    
    # Create model
    model = keras.Model(
        inputs=[input_u, input_v, input_x],
        outputs=outputs if len(outputs) > 1 else direction_output,
        name='three_plane_cnn_enhanced'
    )
    
    # Don't compile here - will be done in training script with proper loss
    
    return model


class CosineAnnealingSchedule(keras.callbacks.Callback):
    """
    Cosine annealing learning rate schedule with warm restarts.
    
    Args:
        initial_lr: Initial learning rate
        t0: Number of epochs for the first restart
        t_mult: Factor to increase t0 after each restart
        eta_min: Minimum learning rate
    """
    def __init__(self, initial_lr=0.001, t0=50, t_mult=2, eta_min=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.t0 = t0
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.t_cur = 0
        self.t_i = t0
        
    def on_epoch_begin(self, epoch, logs=None):
        # Calculate current learning rate
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.t_cur / self.t_i)
        ) / 2
        
        # Update learning rate
        self.model.optimizer.learning_rate.assign(lr)
        
        # Update counters
        self.t_cur += 1
        if self.t_cur >= self.t_i:
            # Restart
            self.t_cur = 0
            self.t_i *= self.t_i
            print(f"\n🔄 Cosine annealing restart at epoch {epoch+1}, next restart in {self.t_i} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = float(self.model.optimizer.learning_rate.numpy())


def get_callbacks(
    output_dir,
    early_stopping_patience=30,
    reduce_lr_patience=10,
    reduce_lr_factor=0.5,
    reduce_lr_min_lr=1e-6,
    use_cosine_annealing=False,
    cosine_t0=50,
    cosine_t_mult=2,
    initial_lr=0.001,
    save_every_epoch=False
):
    """
    Get training callbacks with enhanced options.
    
    Args:
        output_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for ReduceLROnPlateau
        reduce_lr_factor: Factor to reduce LR
        reduce_lr_min_lr: Minimum LR for ReduceLROnPlateau
        use_cosine_annealing: Use cosine annealing instead of ReduceLROnPlateau
        cosine_t0: Initial period for cosine annealing
        cosine_t_mult: Multiplier for cosine annealing period
        initial_lr: Initial learning rate (for cosine annealing)
        save_every_epoch: Save checkpoint every epoch (not just best)
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Learning rate schedule
    if use_cosine_annealing:
        callbacks.append(
            CosineAnnealingSchedule(
                initial_lr=initial_lr,
                t0=cosine_t0,
                t_mult=cosine_t_mult,
                eta_min=reduce_lr_min_lr
            )
        )
    else:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=reduce_lr_min_lr,
                verbose=1
            )
        )
    
    # Model checkpointing
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save best model
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )
    
    # Optionally save every epoch
    if save_every_epoch:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir,
                    'checkpoint_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.keras'
                ),
                monitor='val_loss',
                save_best_only=False,
                save_freq='epoch',
                verbose=0  # Less verbose to avoid cluttering output
            )
        )
    
    # CSV logger for training history
    callbacks.append(
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_history.csv'),
            append=False
        )
    )
    
    return callbacks


if __name__ == '__main__':
    # Test model creation
    print("Testing enhanced three-plane CNN model...")
    print("\n" + "="*70)
    print("TEST 1: Basic configuration")
    print("="*70)
    
    model = build_three_plane_cnn_enhanced(
        input_shape=(128, 32, 1),
        output_dim=3,
        n_conv_layers=4,
        n_filters=64,
        kernel_size=3,
        n_dense_layers=2,
        n_dense_units=256,
        spatial_dropout=0.0,
        dense_dropout=0.3,
        use_batch_norm=True
    )
    
    print(f"\n✓ Basic model created")
    print(f"  Total parameters: {model.count_params():,}")
    
    print("\n" + "="*70)
    print("TEST 2: Deeper configuration with spatial dropout")
    print("="*70)
    
    model_deep = build_three_plane_cnn_enhanced(
        input_shape=(128, 32, 1),
        output_dim=3,
        n_conv_layers=5,
        n_filters=96,
        kernel_size=3,
        n_dense_layers=3,
        n_dense_units=384,
        spatial_dropout=0.25,
        dense_dropout=0.3,
        use_batch_norm=True
    )
    
    print(f"\n✓ Deep model created")
    print(f"  Total parameters: {model_deep.count_params():,}")
    
    print("\n" + "="*70)
    print("TEST 3: Multi-task configuration")
    print("="*70)
    
    model_multitask = build_three_plane_cnn_enhanced(
        input_shape=(128, 32, 1),
        output_dim=3,
        n_conv_layers=4,
        n_filters=64,
        kernel_size=3,
        n_dense_layers=2,
        n_dense_units=256,
        spatial_dropout=0.0,
        dense_dropout=0.3,
        use_batch_norm=True,
        multitask=True
    )
    
    print(f"\n✓ Multi-task model created")
    print(f"  Total parameters: {model_multitask.count_params():,}")
    print(f"  Outputs: {[out.name for out in model_multitask.outputs]}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
