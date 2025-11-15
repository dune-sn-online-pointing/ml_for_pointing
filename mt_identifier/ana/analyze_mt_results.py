#!/usr/bin/env python3
"""
Analyze MT (Main Track) Identifier Binary Classification Results
Generates confusion matrix, ROC curve, prediction distributions, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import tensorflow as tf

def load_model_and_data(run_dir):
    """Load trained model and extract predictions on test data"""
    run_path = Path(run_dir)
    
    # Load config
    config_file = run_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded: {config.get('model_name', 'unknown')}")
    
    # Find model file
    model_files = list(run_path.glob("*.h5")) + list(run_path.glob("*.keras"))
    if not model_files:
        # Check models subdirectory
        models_dir = run_path / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    
    if not model_files:
        raise FileNotFoundError(f"No model file found in {run_path}")
    
    model_file = model_files[0]
    print(f"Loading model from: {model_file}")
    
    try:
        model = tf.keras.models.load_model(model_file)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
    
    return config, model

def load_test_data(config, max_samples=None):
    """Load test data based on config"""
    data_dirs = config.get('data_directories', [])
    plane = config.get('plane', 'X')
    
    print(f"\nLoading test data from {len(data_dirs)} directories...")
    print(f"Plane: {plane}")
    
    all_images = []
    all_labels = []
    
    for i, data_dir in enumerate(data_dirs):
        data_path = Path(data_dir)
        
        # Find NPZ files
        npz_files = list(data_path.glob(f"plane_{plane}/*.npz"))
        if not npz_files:
            npz_files = list(data_path.glob("*.npz"))
        
        print(f"  Dir {i}: {data_path.name} - {len(npz_files)} files")
        
        label = i  # 0 for first dir (CC), 1 for second dir (ES)
        
        for npz_file in npz_files[:max_samples] if max_samples else npz_files:
            try:
                data = np.load(npz_file)
                image = data['image']
                all_images.append(image)
                all_labels.append(label)
            except Exception as e:
                print(f"    Warning: Failed to load {npz_file.name}: {e}")
    
    X = np.array(all_images)
    y = np.array(all_labels)
    
    # Normalize images
    X = X / np.max(X) if np.max(X) > 0 else X
    
    # Reshape if needed
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    
    print(f"\n✅ Loaded {len(X)} samples")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: {np.bincount(y)}")
    print(f"   Label 0 (CC): {np.sum(y==0)} samples")
    print(f"   Label 1 (ES): {np.sum(y==1)} samples")
    
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate model and get predictions"""
    print("\nEvaluating model...")
    
    # Get predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_prob = y_prob.flatten()
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    
    print(f"✅ Predictions generated")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Predicted 0: {np.sum(y_pred==0)}")
    print(f"   Predicted 1: {np.sum(y_pred==1)}")
    
    return y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot confusion matrix for binary classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['CC (0)', 'ES (1)'],
                yticklabels=['CC (0)', 'ES (1)'])
    
    # Add percentages as text
    for i in range(2):
        for j in range(2):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)',
                          ha="center", va="center", color="gray", fontsize=10)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add metrics
    accuracy = np.trace(cm) / np.sum(cm) * 100
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
             transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix: {output_path}")

def plot_roc_curve(y_true, y_prob, output_path, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curve: {output_path}")

def plot_prediction_distribution(y_true, y_prob, output_path, title="Prediction Distribution"):
    """Plot histogram of prediction probabilities for each class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each true class
    for label, name in [(0, 'CC'), (1, 'ES')]:
        mask = y_true == label
        ax.hist(y_prob[mask], bins=50, alpha=0.6, label=f'True {name}', 
                range=(0, 1), density=True)
    
    ax.set_xlabel('Predicted Probability (ES)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved prediction distribution: {output_path}")

def plot_sample_predictions(X_test, y_true, y_pred, y_prob, output_path, n_samples=20):
    """Plot sample predictions with their confidence"""
    # Select diverse samples
    indices = []
    # True positives (high confidence)
    tp_high = np.where((y_true == 1) & (y_pred == 1) & (y_prob > 0.9))[0]
    if len(tp_high) > 0:
        indices.extend(np.random.choice(tp_high, min(5, len(tp_high)), replace=False))
    
    # True negatives (high confidence)
    tn_high = np.where((y_true == 0) & (y_pred == 0) & (y_prob < 0.1))[0]
    if len(tn_high) > 0:
        indices.extend(np.random.choice(tn_high, min(5, len(tn_high)), replace=False))
    
    # False positives
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    if len(fp) > 0:
        indices.extend(np.random.choice(fp, min(5, len(fp)), replace=False))
    
    # False negatives
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    if len(fn) > 0:
        indices.extend(np.random.choice(fn, min(5, len(fn)), replace=False))
    
    if len(indices) == 0:
        print("⚠️  No samples to plot")
        return
    
    indices = indices[:n_samples]
    
    n_cols = 5
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, i in enumerate(indices):
        ax = axes[idx]
        img = X_test[i].squeeze()
        ax.imshow(img, cmap='viridis', aspect='auto')
        
        true_label = 'ES' if y_true[i] == 1 else 'CC'
        pred_label = 'ES' if y_pred[i] == 1 else 'CC'
        confidence = y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
        
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f}',
                    fontsize=9, color=color)
        ax.axis('off')
    
    # Hide remaining axes
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved sample predictions: {output_path}")

def print_metrics_summary(y_true, y_pred, y_prob):
    """Print comprehensive metrics summary"""
    print("\n" + "="*60)
    print("MT IDENTIFIER CLASSIFICATION METRICS")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, 
                               target_names=['CC (0)', 'ES (1)'],
                               digits=4))
    
    # Confusion matrix details
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:")
    print(f"  True CC (0): {cm[0,0]} correct, {cm[0,1]} misclassified as ES")
    print(f"  True ES (1): {cm[1,1]} correct, {cm[1,0]} misclassified as CC")
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Additional metrics
    accuracy = np.mean(y_true == y_pred)
    precision_es = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall_es = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    precision_cc = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    recall_cc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"  Overall Accuracy: {accuracy*100:.2f}%")
    print(f"  ES Precision: {precision_es*100:.2f}%")
    print(f"  ES Recall: {recall_es*100:.2f}%")
    print(f"  CC Precision: {precision_cc*100:.2f}%")
    print(f"  CC Recall: {recall_cc*100:.2f}%")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_mt_results.py <run_directory> [--max-samples N]")
        print("\nExample:")
        print("  python analyze_mt_results.py /eos/.../mt_identifier/v8_simple_10k/mt_identifier_simple_cnn_20251111_101424/")
        print("  python analyze_mt_results.py /eos/.../mt_identifier/hyperopt_simple_cnn/aug_coeff_2/plane_X/20251106_170758/ --max-samples 1000")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    max_samples = None
    
    # Parse max_samples if provided
    if '--max-samples' in sys.argv:
        idx = sys.argv.index('--max-samples')
        if idx + 1 < len(sys.argv):
            max_samples = int(sys.argv[idx + 1])
    
    output_dir = Path(run_dir) / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("MT IDENTIFIER ANALYSIS TOOL")
    print("="*60)
    print(f"\nRun directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    if max_samples:
        print(f"Max samples per class: {max_samples}")
    
    # Load model and config
    config, model = load_model_and_data(run_dir)
    
    if model is None:
        print("\n❌ Cannot proceed without model")
        sys.exit(1)
    
    # Load test data
    X_test, y_test = load_test_data(config, max_samples=max_samples)
    
    # Evaluate model
    y_pred, y_prob = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print_metrics_summary(y_test, y_pred, y_prob)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    
    plot_confusion_matrix(
        y_test, y_pred,
        output_dir / "confusion_matrix.png",
        title="MT Identifier Confusion Matrix"
    )
    
    plot_roc_curve(
        y_test, y_prob,
        output_dir / "roc_curve.png",
        title="MT Identifier ROC Curve"
    )
    
    plot_prediction_distribution(
        y_test, y_prob,
        output_dir / "pred_distribution.png",
        title="MT Identifier Prediction Distribution"
    )
    
    plot_sample_predictions(
        X_test, y_test, y_pred, y_prob,
        output_dir / "sample_predictions.png",
        n_samples=20
    )
    
    # Save predictions
    np.savez(
        output_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob
    )
    print(f"✅ Saved predictions: {output_dir / 'predictions.npz'}")
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
