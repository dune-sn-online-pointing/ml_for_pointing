"""
Three-Plane CNN for Electron Direction Prediction

This model takes three separate 2D projections (U, V, X planes) as input,
processes each with its own CNN branch, then concatenates the features
to predict the 3D direction (x, y, z).

Architecture:
    Input U (128, 16, 1) → CNN_U → Flatten → Features_U
    Input V (128, 16, 1) → CNN_V → Flatten → Features_V  
    Input X (128, 16, 1) → CNN_X → Flatten → Features_X
    Concatenate(Features_U, Features_V, Features_X) → Dense layers → (x, y, z)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from direction_losses import cosine_similarity_loss
from direction_losses import cosine_similarity_loss


def build_three_plane_cnn(
    input_shape=(128, 16, 1),
    output_dim=3,
    n_conv_layers=2,
    n_filters=64,
    kernel_size=3,
    n_dense_layers=2,
    n_dense_units=128,
    learning_rate=0.001,
    decay_rate=0.95
):
    """
    Build a three-plane CNN model for electron direction prediction.
    
    Args:
        input_shape: Shape of each plane input (height, width, channels)
        output_dim: Output dimension (3 for x, y, z)
        n_conv_layers: Number of convolutional layers per branch
        n_filters: Number of filters in conv layers
        kernel_size: Size of convolutional kernels
        n_dense_layers: Number of dense layers after concatenation
        n_dense_units: Number of units in dense layers
        learning_rate: Learning rate for optimizer
        decay_rate: Learning rate decay rate
        
    Returns:
        Compiled Keras model
    """
    
    def create_cnn_branch(name_prefix):
        """Create a single CNN branch for one plane"""
        branch_input = keras.Input(shape=input_shape, name=f'{name_prefix}_input')
        x = branch_input
        
        # Convolutional layers
        for i in range(n_conv_layers):
            x = layers.Conv2D(
                filters=n_filters * (2 ** i),
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'{name_prefix}_conv_{i+1}'
            )(x)
            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'{name_prefix}_pool_{i+1}'
            )(x)
            x = layers.BatchNormalization(
                name=f'{name_prefix}_bn_{i+1}'
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
    
    # Dense layers
    x = concatenated
    for i in range(n_dense_layers):
        x = layers.Dense(
            n_dense_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(0.3, name=f'dropout_{i+1}')(x)
    
    # Output layer
    output = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    # Create model
    model = keras.Model(
        inputs=[input_u, input_v, input_x],
        outputs=output,
        name='three_plane_cnn'
    )
    
    # Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        decay=decay_rate
    )
    
    model.compile(
        optimizer=optimizer,
        loss=cosine_similarity_loss,
        metrics=['mae']
    )
    
    return model


def train_three_plane_model(
    model,
    train_data,
    val_data,
    epochs=50,
    batch_size=32,
    output_folder='.',
    early_stopping_patience=10
):
    """
    Train the three-plane model.
    
    Args:
        model: Compiled Keras model
        train_data: Training dataset (must provide 3 inputs + 1 output)
        val_data: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        output_folder: Where to save checkpoints
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Training history
    """
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Add ModelCheckpoint if output folder exists
    if os.path.exists(output_folder):
        checkpoint_dir = os.path.join(output_folder, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir,
                    'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'
                ),
                monitor='val_loss',
                save_best_only=False,
                save_freq='epoch',
                verbose=1
            )
        )
    
    # Unpack train and validation data
    train_x, train_y = train_data
    val_x, val_y = val_data
    
    history = model.fit(
        train_x,  # List of [U, V, X] plane images
        train_y,  # Direction labels
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


if __name__ == '__main__':
    # Test model creation
    print("Testing three-plane CNN model...")
    
    model = build_three_plane_cnn(
        input_shape=(128, 16, 1),
        output_dim=3,
        n_conv_layers=2,
        n_filters=64,
        kernel_size=3,
        n_dense_layers=2,
        n_dense_units=128
    )
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel inputs:")
    for i, inp in enumerate(model.inputs):
        print(f"  Input {i}: {inp.name} - shape {inp.shape}")
    
    print("\nModel output:")
    print(f"  {model.output.name} - shape {model.output.shape}")
    
    print("\n✓ Model created successfully!")
