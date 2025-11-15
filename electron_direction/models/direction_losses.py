"""
Custom loss functions for electron direction regression.

MSE is inappropriate for unit vector regression because it doesn't 
account for the directional nature of the predictions and can lead
to collapse to mean vector predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="DirectionLosses")
def cosine_similarity_loss(y_true, y_pred):
    """
    Cosine similarity loss for 3D direction vectors.
    
    Loss = 1 - cos(theta) = 1 - (y_true · y_pred) / (||y_true|| ||y_pred||)
    
    This loss ranges from 0 (perfect alignment) to 2 (opposite directions).
    Better than MSE for unit vectors because it directly optimizes angular difference.
    
    Args:
        y_true: True direction vectors, shape (batch_size, 3)
        y_pred: Predicted direction vectors, shape (batch_size, 3)
    
    Returns:
        Scalar loss value
    """
    # Normalize both vectors to unit length
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Compute cosine similarity
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Loss is 1 - cosine similarity
    # Clip to avoid numerical issues with acos later
    cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
    loss = 1.0 - cosine_sim
    
    return tf.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable(package="DirectionLosses")
def angular_loss(y_true, y_pred):
    """
    Angular loss for 3D direction vectors.
    
    Loss = arccos(cos(theta)) = angular separation in radians
    
    This directly optimizes the angular error between predicted and true directions.
    
    Args:
        y_true: True direction vectors, shape (batch_size, 3)
        y_pred: Predicted direction vectors, shape (batch_size, 3)
    
    Returns:
        Scalar loss value (mean angular error in radians)
    """
    # Normalize both vectors to unit length
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Compute cosine similarity
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Clip to valid range for acos
    cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
    
    # Angular error in radians
    angular_error = tf.acos(cosine_sim)
    
    return tf.reduce_mean(angular_error)


@tf.keras.utils.register_keras_serializable(package="DirectionLosses")
def hybrid_angular_mse_loss(y_true, y_pred, alpha=0.7):
    """
    Hybrid loss combining angular loss and MSE.
    
    Loss = alpha * angular_loss + (1-alpha) * mse_loss
    
    The angular component ensures directional optimization while
    MSE helps with gradient stability in early training.
    
    Args:
        y_true: True direction vectors, shape (batch_size, 3)
        y_pred: Predicted direction vectors, shape (batch_size, 3)
        alpha: Weight for angular loss (0 to 1)
    
    Returns:
        Scalar loss value
    """
    ang_loss = angular_loss(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    return alpha * ang_loss + (1.0 - alpha) * mse


@tf.keras.utils.register_keras_serializable(package="DirectionLosses")
def focal_angular_loss(y_true, y_pred, gamma=2.0):
    """
    Focal angular loss for 3D direction vectors.
    
    Applies focal loss weighting to angular errors, focusing training on
    hard examples (large angular errors). Based on focal loss from:
    Lin et al., "Focal Loss for Dense Object Detection"
    
    Loss = (angular_error)^gamma * angular_error
    
    This downweights easy examples (small errors) and focuses on hard cases.
    gamma=2.0 is standard, higher values increase focus on hard examples.
    
    Args:
        y_true: True direction vectors, shape (batch_size, 3)
        y_pred: Predicted direction vectors, shape (batch_size, 3)
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Scalar loss value
    """
    # Normalize both vectors to unit length
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Compute cosine similarity
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Clip to valid range for acos
    cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
    
    # Angular error in radians
    angular_error = tf.acos(cosine_sim)
    
    # Apply focal weighting: (error)^gamma * error
    # Normalize by pi to keep errors in [0, 1] range for weighting
    normalized_error = angular_error / np.pi
    focal_weight = tf.pow(normalized_error, gamma)
    focal_loss = focal_weight * angular_error
    
    return tf.reduce_mean(focal_loss)
