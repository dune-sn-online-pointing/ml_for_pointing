"""
Single-Plane CNN with Normalized Output for Hyperopt
Supports multiple loss functions with output normalization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.direction_losses import cosine_similarity_loss, angular_loss, focal_angular_loss, hybrid_angular_mse_loss


def create_model(params, input_shape=(128, 16, 1), output_dim=3, loss_function='cosine'):
    """
    Create single-plane CNN with normalized output.
    
    Args:
        params: Dictionary with hyperparameters
        input_shape: Shape of input plane
        output_dim: Output dimension (3 for direction)
        loss_function: Loss to use ('cosine', 'angular', 'focal', 'hybrid')
    
    Returns:
        Compiled model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # Get params with defaults
    n_conv_layers = params.get('n_conv_layers', 3)
    n_filters = params.get('n_filters', 32)
    kernel_size = params.get('kernel_size', 5)
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters * (2 ** i)
        
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv_{i+1}'
        )(x)
        
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        
        # Add pooling except on last layer if too small
        if i < n_conv_layers - 1:
            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'pool_{i+1}'
            )(x)
    
    # Flatten
    x = layers.Flatten(name='flatten')(x)
    
    # Dense layers
    n_dense_layers = params.get('n_dense_layers', 2)
    dense_units = params.get('dense_units', 512)
    dropout_rate = params.get('dropout_rate', 0.3)
    
    for i in range(n_dense_layers):
        x = layers.Dense(
            dense_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Output layer - predict unnormalized direction
    output = layers.Dense(output_dim, activation='linear', name='output_unnormalized')(x)
    
    # Add normalization layer to ensure unit vector output
    output_normalized = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=-1),
        name='output_normalized'
    )(output)
    
    # Create model
    model = keras.Model(
        inputs=inputs,
        outputs=output_normalized,
        name='single_plane_normalized_cnn'
    )
    
    # Select loss function
    loss_functions = {
        'cosine': cosine_similarity_loss,
        'angular': angular_loss,
        'focal': focal_angular_loss,
        'hybrid': hybrid_angular_mse_loss
    }
    loss_fn = loss_functions.get(loss_function, cosine_similarity_loss)
    
    # Compile
    learning_rate = params.get('learning_rate', 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae']
    )
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing single-plane normalized CNN...")
    
    params = {
        'n_conv_layers': 3,
        'n_filters': 32,
        'kernel_size': 5,
        'n_dense_layers': 2,
        'dense_units': 512,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
    
    for loss_name in ['cosine', 'angular', 'focal', 'hybrid']:
        print(f"\n{loss_name.upper()} Loss:")
        model = create_model(params, loss_function=loss_name)
        print(f"  ✓ Model created with {loss_name} loss")
    
    print("\n✓ All loss variants working!")
