"""
Three-Plane CNN with Attention for Hyperopt
Includes attention mechanism and diverse architecture options.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.direction_losses import cosine_similarity_loss, angular_loss, focal_angular_loss, hybrid_angular_mse_loss


def attention_block(x, name_prefix='attention'):
    """
    Spatial attention mechanism to focus on important regions.
    
    Args:
        x: Input tensor (batch, height, width, channels)
        name_prefix: Prefix for layer names
    
    Returns:
        Attention-weighted tensor
    """
    # Channel-wise attention
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True, name=f'{name_prefix}_avg_pool')(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True, name=f'{name_prefix}_max_pool')(x)
    
    # Combine pooled features
    concat = layers.Concatenate(name=f'{name_prefix}_concat')([avg_pool, max_pool])
    
    # Generate attention weights
    attention = layers.Conv2D(
        filters=x.shape[-1],
        kernel_size=1,
        activation='sigmoid',
        padding='same',
        name=f'{name_prefix}_weights'
    )(concat)
    
    # Apply attention
    attended = layers.Multiply(name=f'{name_prefix}_multiply')([x, attention])
    
    return attended


def create_model(params, input_shape=(128, 16, 1), output_dim=3, loss_function='cosine'):
    """
    Create three-plane CNN with attention mechanism.
    
    Architecture diversity:
    - Different conv layer depths per plane
    - Varying filter sizes
    - Attention after convolutions
    - Diverse dense layer configurations
    
    Args:
        params: Dictionary with hyperparameters
        input_shape: Shape of each plane
        output_dim: Output dimension (3 for direction)
        loss_function: Loss to use ('cosine', 'angular', 'focal', 'hybrid')
    
    Returns:
        Compiled model
    """
    
    def create_cnn_branch(name_prefix, n_layers_offset=0):
        """
        Create a single CNN branch with diverse architecture.
        
        Args:
            name_prefix: Prefix for layer names
            n_layers_offset: Offset to diversify layer depth between planes
        """
        branch_input = keras.Input(shape=input_shape, name=f'{name_prefix}_input')
        x = branch_input
        
        # Get params with defaults
        base_layers = params.get('n_conv_layers', 3)
        n_layers = base_layers + n_layers_offset  # Diversify depth
        n_filters = params.get('n_filters', 32)
        kernel_size = params.get('kernel_size', 5)
        
        # Convolutional layers
        for i in range(n_layers):
            # Diversify filter growth rate
            if name_prefix == 'plane_u':
                filters = n_filters * (2 ** i)  # Exponential growth
            elif name_prefix == 'plane_v':
                filters = n_filters * (i + 1)  # Linear growth
            else:  # plane_x
                filters = n_filters  # Constant
            
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'{name_prefix}_conv_{i+1}'
            )(x)
            
            x = layers.BatchNormalization(name=f'{name_prefix}_bn_{i+1}')(x)
            
            # Add pooling only if spatial dimensions are large enough
            # Skip pooling on last 2 layers to avoid dimension issues with 128x16 input
            if i < n_layers - 2:
                x = layers.MaxPooling2D(
                    pool_size=(2, 2),
                    name=f'{name_prefix}_pool_{i+1}'
                )(x)
        
        # Apply attention mechanism
        x = attention_block(x, name_prefix=f'{name_prefix}_attention')
        
        # Flatten
        x = layers.Flatten(name=f'{name_prefix}_flatten')(x)
        
        return branch_input, x
    
    # Create three branches with diverse depths
    # U plane: base layers - 1
    # V plane: base layers
    # X plane: base layers + 1
    input_u, features_u = create_cnn_branch('plane_u', n_layers_offset=-1)
    input_v, features_v = create_cnn_branch('plane_v', n_layers_offset=0)
    input_x, features_x = create_cnn_branch('plane_x', n_layers_offset=1)
    
    # Concatenate all features
    concatenated = layers.Concatenate(name='concatenate')([features_u, features_v, features_x])
    
    # Dense layers with diverse configurations
    x = concatenated
    n_dense_layers = params.get('n_dense_layers', 2)
    dense_units = params.get('dense_units', 512)
    dropout_rate = params.get('dropout_rate', 0.3)
    
    for i in range(n_dense_layers):
        # Diversify dense layer sizes: decreasing pattern
        units = dense_units // (2 ** i)
        units = max(units, 64)  # Minimum 64 units
        
        x = layers.Dense(
            units,
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
        inputs=[input_u, input_v, input_x],
        outputs=output_normalized,
        name='three_plane_attention_cnn'
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
    print("Testing three-plane attention CNN...")
    
    params = {
        'n_conv_layers': 3,
        'n_filters': 32,
        'kernel_size': 5,
        'n_dense_layers': 2,
        'dense_units': 512,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
    
    model = create_model(params)
    
    print("\nModel Summary:")
    model.summary()
    
    print("\n✓ Model with attention created successfully!")
