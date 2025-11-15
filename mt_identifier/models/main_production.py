#!/usr/bin/env python3
"""
Main Track Identifier Training Script (Production)

Trains a CNN to distinguish main electron tracks from background.
Supports multiple input directories and saves comprehensive performance metrics.
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import pickle

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import general_purpose_libs as gpl
import classification_libs as cl
import data_loader as dl

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Main Track Identifier neural network',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-j', '--json',
        type=str,
        required=True,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Override output folder from JSON'
    )
    
    parser.add_argument(
        '--plane',
        type=str,
        default=None,
        choices=['U', 'V', 'X'],
        help='Override plane selection from JSON'
    )
    
    return parser.parse_args()


def load_configuration(json_path):
    """Load configuration from JSON file"""
    print(f"\nLoading configuration from: {json_path}")
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'output_folder', 'data_directories']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in JSON: {field}")
    
    return config


def setup_output_directory(config, override_output=None):
    """Create output directory structure"""
    base_output = override_output if override_output else config['output_folder']
    model_name = config['model_name']
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(base_output, f"mt_identifier_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nOutput directory: {output_dir}")
    
    return output_dir


def select_model(config):
    """Select and import the appropriate model"""
    model_name = config['model_name']
    
    if model_name == 'simple_cnn':
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
        import simple_cnn as model_module
        return model_module
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_performance_metrics(history, model, test_data, test_labels, output_dir, config):
    """Save comprehensive performance metrics and plots"""
    print("\n" + "="*60)
    print("SAVING PERFORMANCE METRICS")
    print("="*60)
    
    save_config = config.get('save_outputs', {})
    
    # 1. Save training history
    if save_config.get('save_history', True):
        history_path = os.path.join(output_dir, 'metrics', 'training_history.json')
        history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"✓ Saved training history to {history_path}")
        
        # Plot training curves
        plot_training_history(history, output_dir)
    
    # 2. Get predictions on test set
    print("\nGenerating predictions on test set...")
    test_predictions = model.predict(test_data, verbose=0)
    test_pred_classes = (test_predictions > 0.5).astype(int).flatten()
    
    # 3. Save predictions
    if save_config.get('save_predictions', True):
        pred_path = os.path.join(output_dir, 'predictions', 'test_predictions.npz')
        np.savez(pred_path,
                 predictions=test_predictions,
                 pred_classes=test_pred_classes,
                 true_labels=test_labels)
        print(f"✓ Saved predictions to {pred_path}")
    
    # 4. Calculate and save metrics
    if save_config.get('save_metrics', True):
        metrics = calculate_metrics(test_labels, test_predictions, test_pred_classes)
        metrics_path = os.path.join(output_dir, 'metrics', 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")
        
        # Print summary
        print("\nTest Set Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # 5. Save confusion matrix
    if save_config.get('save_confusion_matrix', True):
        plot_confusion_matrix(test_labels, test_pred_classes, output_dir)
    
    # 6. Save ROC curve
    if save_config.get('save_roc_curve', True):
        plot_roc_curve(test_labels, test_predictions, output_dir)
    
    # 7. Save model
    if save_config.get('save_model', True):
        model_path = os.path.join(output_dir, 'models', 'final_model.keras')
        model.save(model_path)
        print(f"✓ Saved model to {model_path}")
    
    print("\n" + "="*60)


def calculate_metrics(true_labels, predictions, pred_classes):
    """Calculate comprehensive classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': float(accuracy_score(true_labels, pred_classes)),
        'precision': float(precision_score(true_labels, pred_classes, zero_division=0)),
        'recall': float(recall_score(true_labels, pred_classes, zero_division=0)),
        'f1_score': float(f1_score(true_labels, pred_classes, zero_division=0)),
        'auc_roc': float(roc_auc_score(true_labels, predictions))
    }
    
    return metrics


def plot_training_history(history, output_dir):
    """Plot and save training history curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curves to {plot_path}")


def plot_confusion_matrix(true_labels, pred_classes, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(true_labels, pred_classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Background', 'Main Track'],
           yticklabels=['Background', 'Main Track'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as numpy array
    cm_path = os.path.join(output_dir, 'metrics', 'confusion_matrix.npy')
    np.save(cm_path, cm)
    
    print(f"✓ Saved confusion matrix to {plot_path}")


def plot_roc_curve(true_labels, predictions, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'plots', 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save ROC data
    roc_data_path = os.path.join(output_dir, 'metrics', 'roc_data.npz')
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
    
    print(f"✓ Saved ROC curve to {plot_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("MAIN TRACK IDENTIFIER TRAINING")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_configuration(args.json)
    
    # Override config with command-line arguments
    if args.plane:
        config['plane'] = args.plane
    
    plane = config.get('plane', 'X')
    
    # Setup output directory
    output_dir = setup_output_directory(config, args.output)
    
    # Select model
    print("\nSelecting model...")
    model_module = select_model(config)
    
    # Prepare data from multiple directories
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    train, val, test, history_dict = cl.prepare_data_from_multiple_npz(
        data_dirs=config['data_directories'],
        plane=plane,
        dataset_parameters=config['dataset_parameters'],
        output_folder=output_dir
    )
    
    train_img, train_label = train
    val_img, val_label = val
    test_img, test_label, test_metadata = test
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    # Get input shape from train data
    input_shape = train_img.shape[1:]
    
    # Get epochs and batch_size from config
    epochs = config['model_parameters'].get('epochs', 200)
    batch_size = config['model_parameters'].get('batch_size', 32)
    
    model, history = model_module.build_model(
        config['model_parameters'],
        train,
        val,
        output_dir,
        input_shape,
        epochs=epochs,
        batch_size=batch_size
    )
    model.summary()
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    callbacks = []
    
    # Early stopping
    if 'early_stopping' in config['model_parameters']:
        es_config = config['model_parameters']['early_stopping']
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            restore_best_weights=es_config.get('restore_best_weights', True),
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, 'models', 'best_model.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Train
    history = model.fit(
        train_img, train_label,
        validation_data=(val_img, val_label),
        epochs=config['model_parameters']['epochs'],
        batch_size=config['model_parameters']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(test_img, test_label, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save performance metrics
    save_performance_metrics(history, model, test_img, test_label, output_dir, config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")
    

if __name__ == '__main__':
    main()
