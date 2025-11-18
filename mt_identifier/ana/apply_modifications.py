#!/usr/bin/env python3
"""Apply modifications to the comprehensive report script"""

# Read the original script
with open('comprehensive_report.py', 'r') as f:
    content = f.read()

# Modification 1: Confusion matrix - show only percentages
old_cm = '''    # Confusion matrix
    ax = plt.subplot(2, 2, 2)
    cm = confusion_matrix(labels, pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-MT', 'Main Track'], yticklabels=['Non-MT', 'Main Track'])
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.7, f'({cm_norm[i,j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')'''

new_cm = '''    # Confusion matrix
    ax = plt.subplot(2, 2, 2)
    cm = confusion_matrix(labels, pred_binary)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=['Non-MT', 'Main Track'], yticklabels=['Non-MT', 'Main Track'],
                cbar_kws={'label': 'Percentage (%)'})
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix (Percentages)')'''

content = content.replace(old_cm, new_cm)

# Modification 2: Add log scale prediction distribution and change layout
old_pred_dist = '''    # Prediction distribution
    ax = plt.subplot(2, 2, 4)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)'''

new_pred_dist = '''    # Prediction distribution (linear scale)
    ax = plt.subplot(2, 3, 4)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Prediction distribution (log scale)
    ax = plt.subplot(2, 3, 5)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Prediction Distribution (Log Scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)'''

content = content.replace(old_pred_dist, new_pred_dist)

# Update figure layout for summary page to accommodate new plot
old_fig_def = '''def plot_summary_page(pdf, results, run_path):
    """Page 1: Summary statistics and basic info"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('MT Identifier Comprehensive Analysis Report', fontsize=16, fontweight='bold')'''

new_fig_def = '''def plot_summary_page(pdf, results, run_path):
    """Page 1: Summary statistics and basic info"""
    fig = plt.figure(figsize=(14, 8.5))
    fig.suptitle('MT Identifier Comprehensive Analysis Report', fontsize=16, fontweight='bold')'''

content = content.replace(old_fig_def, new_fig_def)

# Update subplot layouts
old_summary_text = '''    # Summary text box
    ax = plt.subplot(2, 2, 1)'''
new_summary_text = '''    # Summary text box
    ax = plt.subplot(2, 3, 1)'''
content = content.replace(old_summary_text, new_summary_text)

old_cm_subplot = '''    # Confusion matrix
    ax = plt.subplot(2, 2, 2)'''
new_cm_subplot = '''    # Confusion matrix
    ax = plt.subplot(2, 3, 2)'''
content = content.replace(old_cm_subplot, new_cm_subplot)

old_roc = '''    # ROC Curve
    ax = plt.subplot(2, 2, 3)'''
new_roc = '''    # ROC Curve
    ax = plt.subplot(2, 3, 3)'''
content = content.replace(old_roc, new_roc)

# Modification 3: Change colors in "Track Predictions by Energy" plots
old_nonmt_pred = '''    # Non-MT predictions by energy
    ax = axes[0, 0]
    nonmt_correct = nonmt_mask & correct_mask
    nonmt_incorrect = nonmt_mask & ~correct_mask
    ax.hist(energies[nonmt_correct], bins=50, alpha=0.6, 
            label=f'Non-MT Correct (n={nonmt_correct.sum():,})', color='blue')
    ax.hist(energies[nonmt_incorrect], bins=50, alpha=0.6, 
            label=f'Non-MT Misclassified (n={nonmt_incorrect.sum():,})', color='lightblue')'''

new_nonmt_pred = '''    # Non-MT predictions by energy
    ax = axes[0, 0]
    nonmt_correct = nonmt_mask & correct_mask
    nonmt_incorrect = nonmt_mask & ~correct_mask
    ax.hist(energies[nonmt_correct], bins=50, alpha=0.7, 
            label=f'Non-MT Correct (n={nonmt_correct.sum():,})', color='#2E86AB')  # Dark blue
    ax.hist(energies[nonmt_incorrect], bins=50, alpha=0.7, 
            label=f'Non-MT Misclassified (n={nonmt_incorrect.sum():,})', color='#A23B72')'''  # Purple

content = content.replace(old_nonmt_pred, new_nonmt_pred)

old_mt_pred = '''    # Main Track predictions by energy
    ax = axes[1, 0]
    mt_correct = mt_mask & correct_mask
    mt_incorrect = mt_mask & ~correct_mask
    ax.hist(energies[mt_correct], bins=50, alpha=0.6, 
            label=f'Main Track Correct (n={mt_correct.sum():,})', color='red')
    ax.hist(energies[mt_incorrect], bins=50, alpha=0.6, 
            label=f'Main Track Misclassified (n={mt_incorrect.sum():,})', color='lightcoral')'''

new_mt_pred = '''    # Main Track predictions by energy
    ax = axes[1, 0]
    mt_correct = mt_mask & correct_mask
    mt_incorrect = mt_mask & ~correct_mask
    ax.hist(energies[mt_correct], bins=50, alpha=0.7, 
            label=f'Main Track Correct (n={mt_correct.sum():,})', color='#F18F01')  # Orange
    ax.hist(energies[mt_incorrect], bins=50, alpha=0.7, 
            label=f'Main Track Misclassified (n={mt_incorrect.sum():,})', color='#C73E1D')'''  # Dark red

content = content.replace(old_mt_pred, new_mt_pred)

# Modification 4: Combine prediction vs energy plots
old_pred_analysis = '''def plot_prediction_analysis(pdf, results):
    """Page 5: Detailed prediction analysis"""
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    
    # Handle both structured array and dict array formats
    if metadata.dtype.names:
        energies = metadata['true_energy_sum']
    else:
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    pred_binary = (predictions > 0.5).astype(int)
    nonmt_mask = (labels == 0)
    mt_mask = (labels == 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Predictions vs energy scatter (Non-MT)
    ax = axes[0, 0]
    scatter = ax.scatter(energies[nonmt_mask], predictions[nonmt_mask], 
                        c=pred_binary[nonmt_mask], cmap='RdYlGn', 
                        alpha=0.3, s=10, vmin=0, vmax=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Predicted Probability (Main Track)')
    ax.set_title('Non-MT: Predictions vs Energy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Predicted Class')
    
    # Predictions vs energy scatter (Main Track)
    ax = axes[0, 1]
    scatter = ax.scatter(energies[mt_mask], predictions[mt_mask], 
                        c=pred_binary[mt_mask], cmap='RdYlGn', 
                        alpha=0.3, s=10, vmin=0, vmax=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Predicted Probability (Main Track)')
    ax.set_title('Main Track: Predictions vs Energy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Predicted Class')'''

new_pred_analysis = '''def plot_prediction_analysis(pdf, results):
    """Page 5: Detailed prediction analysis"""
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    
    # Handle both structured array and dict array formats
    if metadata.dtype.names:
        energies = metadata['true_energy_sum']
    else:
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    pred_binary = (predictions > 0.5).astype(int)
    nonmt_mask = (labels == 0)
    mt_mask = (labels == 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Combined predictions vs energy scatter
    ax = axes[0, 0]
    ax.scatter(energies[nonmt_mask], predictions[nonmt_mask], 
              color='#2E86AB', alpha=0.3, s=10, label='Non-MT')
    ax.scatter(energies[mt_mask], predictions[mt_mask], 
              color='#F18F01', alpha=0.3, s=10, label='Main Track')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Predicted Probability (Main Track)')
    ax.set_title('Predictions vs Energy (Both Classes)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy distribution showing negative values if present
    ax = axes[0, 1]
    negative_mask = energies < 0
    if negative_mask.sum() > 0:
        ax.hist(energies, bins=100, alpha=0.7, color='gray', label=f'All (n={len(energies):,})')
        ax.hist(energies[negative_mask], bins=50, alpha=0.7, color='red', 
                label=f'Negative (n={negative_mask.sum():,})')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution (showing negative values)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
    else:
        ax.hist(energies[nonmt_mask], bins=50, alpha=0.6, label='Non-MT', color='#2E86AB')
        ax.hist(energies[mt_mask], bins=50, alpha=0.6, label='Main Track', color='#F18F01')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)'''

content = content.replace(old_pred_analysis, new_pred_analysis)

# Write the modified script
with open('comprehensive_report_modified.py', 'w') as f:
    f.write(content)

print("✅ Modifications applied successfully to comprehensive_report_modified.py")
