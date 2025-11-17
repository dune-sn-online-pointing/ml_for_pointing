"""
Three-Plane Channel Tagging CNN

Builds a CNN that processes U, V, X planes separately then concatenates for classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_three_plane_ct_cnn(
    input_shape=(128, 32, 1),
    n_classes=3,
    n_conv_layers=3,
    n_filters=64,
    kernel_size=3,
    n_dense_layers=2,
    n_dense_units=256,
    dropout_rate=0.3,
    learning_rate=0.001
):
    """
    Build a three-plane CNN model for channel tagging classification.
    
    Args:
        input_shape: Shape of each plane input (height, width, channels)
        n_classes: Number of output classes (ES, CC, NC)
        n_conv_layers: Number of convolutional layers per branch
        n_filters: Base number of filters in conv layers
        kernel_size: Size of convolutional kernels
        n_dense_layers: Number of dense layers after concatenation
        n_dense_units: Number of units in dense layers
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        
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
        
        # Flatten the output
        x = layers.Flatten(name=f'{name_prefix}_flatten')(x)
        
        return branch_input, x
    
    # Create three branches
    u_input, u_features = create_cnn_branch('u_plane')
    v_input, v_features = create_cnn_branch('v_plane')
    x_input, x_features = create_cnn_branch('x_plane')
    
    # Concatenate features from all planes
    concatenated = layers.Concatenate(name='concatenate')([u_features, v_features, x_features])
    
    # Dense layers
    x = concatenated
    for i in range(n_dense_layers):
        x = layers.Dense(
            units=n_dense_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Output layer
    output = layers.Dense(
        units=n_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Create model
    model = keras.Model(
        inputs=[u_input, v_input, x_input],
        outputs=output,
        name='three_plane_ct_cnn'
    )
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_three_plane_ct_model(
    model,
    train_data,
    val_data,
    epochs=50,
    batch_size=32,
    output_folder=None,
    early_stopping_patience=20,
    reduce_lr_patience=10
):
    """
    Train the three-plane CT model.
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of ((u, v, x), labels)
        val_data: Tuple of ((u, v, x), labels)
        epochs: Number of training epochs
        batch_size: Batch size
        output_folder: Folder to save checkpoints
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        
    Returns:
        Training history
    """
    
    (train_u, train_v, train_x), train_labels = train_data
    (val_u, val_v, val_x), val_labels = val_data
    
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
    
    # Reduce learning rate on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            verbose=1,
            min_lr=1e-6
        )
    )
    
    # Model checkpoint
    if output_folder:
        import os
        os.makedirs(output_folder, exist_ok=True)
        checkpoint_path = os.path.join(output_folder, 'best_model.keras')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Train model
    history = model.fit(
        [train_u, train_v, train_x],
        train_labels,
        validation_data=([val_u, val_v, val_x], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
