#!/usr/bin/env python3
"""
Create training plots from MT Identifier v5 log output
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Training metrics extracted from log
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [0.2875, 0.2442, 0.2401, 0.2376, 0.2355, 0.2336, 0.2320, 0.2304, 0.2292, 0.2276]
val_loss = [0.2419, 0.2377, 0.2337, 0.2329, 0.2321, 0.2318, 0.2324, 0.2324, 0.2332, 0.2331]
train_acc = [0.9045, 0.9162, 0.9175, 0.9180, 0.9186, 0.9191, 0.9196, 0.9201, 0.9202, 0.9205]
val_acc = [0.9158, 0.9182, 0.9193, 0.9198, 0.9203, 0.9198, 0.9202, 0.9207, 0.9198, 0.9201]

# Final test metrics from log
test_accuracy = 0.9203
test_precision = 0.9218
test_recall = 0.9203
test_f1 = 0.9202

# Confusion matrix from log
confusion_matrix = np.array([[0.89, 0.11], [0.05, 0.95]])

# Output directory
output_dir = "/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/simple_cnn/plane_X/20251105_192215"

# Create training history plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=6)
ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.22, 0.30])

# Accuracy plot
ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
ax2.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.90, 0.93])

plt.tight_layout()
plt.savefig(f'{output_dir}/training_history.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/training_history.png")
plt.close()

# Create confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Proportion', rotation=270, labelpad=20, fontsize=12)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=16, fontweight='bold')

# Labels and title
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Background', 'Main Track'], fontsize=11)
ax.set_yticklabels(['Background', 'Main Track'], fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix\nAccuracy: {test_accuracy:.2%}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/confusion_matrix.png")
plt.close()

# Create metrics summary plot
fig, ax = plt.subplots(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_accuracy, test_precision, test_recall, test_f1]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2%}',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('MT Identifier v5 - Test Set Performance', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/test_metrics.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/test_metrics.png")
plt.close()

# Save training history as JSON
history_data = {
    'epochs': epochs,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1
    },
    'confusion_matrix': confusion_matrix.tolist()
}

with open(f'{output_dir}/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)
print(f"Saved: {output_dir}/training_history.json")

print("\n✓ All plots created successfully!")
print(f"\nFiles saved to: {output_dir}/")
print("  - training_history.png (loss and accuracy curves)")
print("  - confusion_matrix.png (classification performance)")
print("  - test_metrics.png (summary bar chart)")
print("  - training_history.json (raw data)")
