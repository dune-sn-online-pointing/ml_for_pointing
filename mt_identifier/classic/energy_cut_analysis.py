#!/usr/bin/env python3
"""
Energy Cut vs ML Classifier Analysis for Main Track Identification

This script compares the performance of a simple energy threshold cut
against the ML classifier (MT v27) to determine if the ML approach
provides sufficient added value.

Questions to answer:
1. What energy cut removes 95% of background (CC) clusters?
2. How many true MT (ES) clusters would we lose with that cut?
3. How does this compare to MT v27's performance?
4. Is the ML classifier worth the complexity?
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Paths
MODEL_DIR = Path("/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/v27_200k/mt_fixed_20251117_222505")
OUTPUT_DIR = Path("/afs/cern.ch/work/e/evilla/private/dune/refactor_ml/mt_identifier/classic")

def load_mt_data():
    """Load MT v27 test data."""
    print("=" * 80)
    print("LOADING MT v27 TEST DATA")
    print("=" * 80)
    
    # Load test data
    test_labels = np.load(MODEL_DIR / "test_labels.npy")
    test_predictions = np.load(MODEL_DIR / "test_predictions.npy")
    test_predictions_binary = np.load(MODEL_DIR / "test_predictions_binary.npy")
    test_metadata = np.load(MODEL_DIR / "test_metadata.npy", allow_pickle=True)
    
    # Load metrics
    with open(MODEL_DIR / "metrics" / "test_metrics.json", 'r') as f:
        metrics = json.load(f)
    
    print(f"✓ Loaded {len(test_labels)} test samples")
    print(f"✓ True ES (label=1): {np.sum(test_labels == 1)}")
    print(f"✓ True CC (label=0): {np.sum(test_labels == 0)}")
    print(f"\nML Classifier Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    print(f"  AUC-ROC:   {metrics['auc']:.1%}")
    print()
    
    # Extract energy from metadata (stored as dictionary with 'true_energy_sum' key)
    energies = np.array([meta['true_energy_sum'] for meta in test_metadata])
    
    return {
        'labels': test_labels,
        'predictions': test_predictions,
        'predictions_binary': test_predictions_binary,
        'energies': energies,
        'metadata': test_metadata,
        'ml_metrics': metrics
    }

def analyze_energy_distributions(data):
    """Analyze energy distributions for ES and CC."""
    print("=" * 80)
    print("ENERGY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    es_mask = data['labels'] == 1
    cc_mask = data['labels'] == 0
    
    es_energies = data['energies'][es_mask]
    cc_energies = data['energies'][cc_mask]
    
    print(f"\nES (Main Track) Energy Statistics:")
    print(f"  Count:  {len(es_energies)}")
    print(f"  Mean:   {np.mean(es_energies):.2f} MeV")
    print(f"  Median: {np.median(es_energies):.2f} MeV")
    print(f"  Std:    {np.std(es_energies):.2f} MeV")
    print(f"  Min:    {np.min(es_energies):.2f} MeV")
    print(f"  Max:    {np.max(es_energies):.2f} MeV")
    
    print(f"\nCC (Background) Energy Statistics:")
    print(f"  Count:  {len(cc_energies)}")
    print(f"  Mean:   {np.mean(cc_energies):.2f} MeV")
    print(f"  Median: {np.median(cc_energies):.2f} MeV")
    print(f"  Std:    {np.std(cc_energies):.2f} MeV")
    print(f"  Min:    {np.min(cc_energies):.2f} MeV")
    print(f"  Max:    {np.max(cc_energies):.2f} MeV")
    print()
    
    return es_energies, cc_energies

def find_energy_cut_for_background_rejection(data, threshold=3.0):
    """Analyze performance at specified energy threshold."""
    print("=" * 80)
    print(f"ANALYZING ENERGY CUT AT {threshold} MeV")
    print("=" * 80)
    
    cc_mask = data['labels'] == 0
    cc_energies = data['energies'][cc_mask]
    
    print(f"\nEnergy threshold: {threshold:.2f} MeV")
    
    # Calculate actual performance at this threshold
    es_mask = data['labels'] == 1
    es_energies = data['energies'][es_mask]
    
    # How many ES would we keep?
    es_kept = np.sum(es_energies >= threshold)
    es_total = len(es_energies)
    es_efficiency = es_kept / es_total
    
    # How many CC would we keep?
    cc_kept = np.sum(cc_energies >= threshold)
    cc_total = len(cc_energies)
    cc_rejection = 1 - (cc_kept / cc_total)
    
    # Overall accuracy if we classify all above threshold as ES
    correct = es_kept + (cc_total - cc_kept)
    total = es_total + cc_total
    accuracy = correct / total
    
    # Precision and recall
    true_positives = es_kept
    false_positives = cc_kept
    false_negatives = es_total - es_kept
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance with E > {threshold:.2f} MeV cut:")
    print(f"  ES kept:           {es_kept}/{es_total} ({es_efficiency:.1%})")
    print(f"  CC rejected:       {cc_total - cc_kept}/{cc_total} ({cc_rejection:.1%})")
    print(f"  Overall accuracy:  {accuracy:.1%}")
    print(f"  Precision:         {precision:.1%}")
    print(f"  Recall:            {recall:.1%}")
    print(f"  F1 Score:          {f1:.3f}")
    print()
    
    return {
        'threshold': threshold,
        'es_efficiency': es_efficiency,
        'cc_rejection': cc_rejection,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'es_kept': es_kept,
        'es_total': es_total,
        'cc_kept': cc_kept,
        'cc_total': cc_total
    }

def compare_methods(data, energy_cut_results):
    """Compare energy cut vs ML classifier."""
    print("=" * 80)
    print("COMPARISON: ENERGY CUT vs ML CLASSIFIER")
    print("=" * 80)
    
    ml_metrics = data['ml_metrics']
    
    print(f"\n{'Metric':<20} {'Energy Cut':<15} {'ML Classifier':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('Accuracy', energy_cut_results['accuracy'], ml_metrics['accuracy']),
        ('Precision', energy_cut_results['precision'], ml_metrics['precision']),
        ('Recall', energy_cut_results['recall'], ml_metrics['recall']),
        ('F1 Score', energy_cut_results['f1_score'], ml_metrics['f1_score']),
    ]
    
    for name, energy_val, ml_val in metrics_to_compare:
        diff = ml_val - energy_val
        diff_str = f"{diff:+.1%}" if abs(diff) > 0.001 else "~0"
        print(f"{name:<20} {energy_val:<15.1%} {ml_val:<15.1%} {diff_str:<15}")
    
    print()
    print("Key Findings:")
    
    if ml_metrics['accuracy'] > energy_cut_results['accuracy']:
        improvement = ml_metrics['accuracy'] - energy_cut_results['accuracy']
        print(f"  ✓ ML classifier provides {improvement:.1%} accuracy improvement")
    else:
        print(f"  ✗ Energy cut achieves similar/better accuracy")
    
    if ml_metrics['recall'] > energy_cut_results['recall']:
        improvement = ml_metrics['recall'] - energy_cut_results['recall']
        saved_mt = improvement * energy_cut_results['es_total']
        print(f"  ✓ ML classifier saves {saved_mt:.0f} more MT clusters ({improvement:.1%})")
    else:
        print(f"  ✗ Energy cut has similar/better recall")
    
    print()
    
    return metrics_to_compare

def plot_energy_distributions(data, energy_cut_results, es_energies, cc_energies):
    """Create comprehensive energy distribution plots."""
    print("Generating energy distribution plots...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Overlapping histograms
    ax1 = plt.subplot(2, 3, 1)
    bins = np.linspace(0, max(np.max(es_energies), np.max(cc_energies)), 50)
    ax1.hist(cc_energies, bins=bins, alpha=0.6, label='Background', color='red', edgecolor='black')
    ax1.hist(es_energies, bins=bins, alpha=0.6, label='Main Tracks', color='blue', edgecolor='black')
    ax1.axvline(energy_cut_results['threshold'], color='green', linestyle='--', linewidth=2, 
                label=f'Cut: {energy_cut_results["threshold"]:.1f} MeV')
    ax1.set_xlabel('Cluster Energy (MeV)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Energy Distributions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distributions
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(es_energies, bins=100, cumulative=True, density=True, alpha=0.7, 
             label='Main Tracks', color='blue', histtype='step', linewidth=2)
    ax2.hist(cc_energies, bins=100, cumulative=True, density=True, alpha=0.7,
             label='Background', color='red', histtype='step', linewidth=2)
    ax2.axvline(energy_cut_results['threshold'], color='green', linestyle='--', linewidth=2)
    ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='95% rejection target')
    ax2.set_xlabel('Cluster Energy (MeV)', fontsize=12)
    ax2.set_ylabel('Cumulative Fraction', fontsize=12)
    ax2.set_title('Cumulative Energy Distributions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Log scale histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(cc_energies, bins=bins, alpha=0.6, label='Background', color='red', edgecolor='black')
    ax3.hist(es_energies, bins=bins, alpha=0.6, label='Main Tracks', color='blue', edgecolor='black')
    ax3.axvline(energy_cut_results['threshold'], color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Cluster Energy (MeV)', fontsize=12)
    ax3.set_ylabel('Count (log scale)', fontsize=12)
    ax3.set_yscale('log')
    ax3.set_title('Energy Distributions (Log Scale)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Performance comparison bar chart
    ax4 = plt.subplot(2, 3, 4)
    ml_metrics = data['ml_metrics']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    energy_vals = [energy_cut_results['accuracy'], energy_cut_results['precision'], 
                   energy_cut_results['recall'], energy_cut_results['f1_score']]
    ml_vals = [ml_metrics['accuracy'], ml_metrics['precision'], 
               ml_metrics['recall'], ml_metrics['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, energy_vals, width, label='Energy Cut', color='orange', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ml_vals, width, label='ML Classifier', color='purple', alpha=0.8)
    
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 5. ES efficiency vs CC rejection trade-off
    ax5 = plt.subplot(2, 3, 5)
    thresholds = np.percentile(cc_energies, np.linspace(80, 99, 50))
    
    es_effs = []
    cc_rejs = []
    
    for thresh in thresholds:
        es_eff = np.sum(es_energies >= thresh) / len(es_energies)
        cc_rej = 1 - (np.sum(cc_energies >= thresh) / len(cc_energies))
        es_effs.append(es_eff)
        cc_rejs.append(cc_rej)
    
    ax5.plot(cc_rejs, es_effs, 'b-', linewidth=2, label='Energy cut performance')
    ax5.plot(1 - ml_metrics['precision'], ml_metrics['recall'], 'r*', markersize=15, 
             label='ML Classifier', zorder=10)
    ax5.plot(energy_cut_results['cc_rejection'], energy_cut_results['recall'], 'go', 
             markersize=12, label=f'Cut @ {energy_cut_results["threshold"]:.1f} MeV', zorder=10)
    ax5.set_xlabel('CC Rejection Rate', fontsize=12)
    ax5.set_ylabel('ES Efficiency (Recall)', fontsize=12)
    ax5.set_title('ES Efficiency vs CC Rejection', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.8, 1.0)
    ax5.set_ylim(0.6, 1.0)
    
    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    SUMMARY: Energy Cut vs ML Classifier
    ═══════════════════════════════════════
    
    Energy Cut Strategy:
    • Threshold: {energy_cut_results['threshold']:.1f} MeV
    • ES Efficiency: {energy_cut_results['recall']:.1%}
    • CC Rejection: {energy_cut_results['cc_rejection']:.1%}
    • Accuracy: {energy_cut_results['accuracy']:.1%}
    
    ML Classifier (v27):
    • ES Efficiency: {ml_metrics['recall']:.1%}
    • CC Rejection: {1-ml_metrics['precision']:.1%}
    • Accuracy: {ml_metrics['accuracy']:.1%}
    • AUC-ROC: {ml_metrics['auc']:.1%}
    
    Comparison:
    • Accuracy gain: {(ml_metrics['accuracy'] - energy_cut_results['accuracy'])*100:+.1f}%
    • ES saved: {(ml_metrics['recall'] - energy_cut_results['recall']) * energy_cut_results['es_total']:.0f} clusters
    • Trade-off: {energy_cut_results['es_total'] - energy_cut_results['es_kept']:.0f} ES lost with simple cut
    
    Conclusion:
    """
    
    if ml_metrics['accuracy'] - energy_cut_results['accuracy'] > 0.05:
        conclusion = "✓ ML classifier provides SIGNIFICANT improvement\n    Worth the complexity!"
    elif ml_metrics['accuracy'] - energy_cut_results['accuracy'] > 0.02:
        conclusion = "? ML classifier provides MODEST improvement\n    Marginal benefit"
    else:
        conclusion = "✗ ML classifier provides MINIMAL improvement\n    Simple cut may suffice"
    
    summary_text += conclusion
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "energy_cut_vs_ml_analysis_corrected_labels.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()

def save_analysis_report(data, energy_cut_results, es_energies, cc_energies):
    """Save detailed text report."""
    print("Generating analysis report...")
    
    report_path = OUTPUT_DIR / "energy_cut_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MAIN TRACK IDENTIFICATION: Energy Cut vs ML Classifier Analysis\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {np.datetime64('today')}\n")
        f.write(f"Model: MT v27 (200k balanced samples)\n")
        f.write(f"Test samples: {len(data['labels'])}\n")
        f.write("\n")
        
        f.write("QUESTION:\n")
        f.write("-" * 80 + "\n")
        f.write("Is the ML classifier worth the complexity, or could we achieve similar\n")
        f.write("performance with a simple energy threshold cut?\n")
        f.write("\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Analyze energy distributions for ES (main track) vs CC (background)\n")
        f.write("2. Apply 3.0 MeV energy threshold cut\n")
        f.write("3. Measure ES retention and CC rejection rates\n")
        f.write("4. Compare performance metrics with ML classifier (v27)\n")
        f.write("\n")
        
        f.write("ENERGY STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"ES (Main Track) clusters: {len(es_energies)}\n")
        f.write(f"  Mean:   {np.mean(es_energies):8.2f} MeV\n")
        f.write(f"  Median: {np.median(es_energies):8.2f} MeV\n")
        f.write(f"  Std:    {np.std(es_energies):8.2f} MeV\n")
        f.write(f"  Range:  {np.min(es_energies):8.2f} - {np.max(es_energies):.2f} MeV\n")
        f.write("\n")
        f.write(f"CC (Background) clusters: {len(cc_energies)}\n")
        f.write(f"  Mean:   {np.mean(cc_energies):8.2f} MeV\n")
        f.write(f"  Median: {np.median(cc_energies):8.2f} MeV\n")
        f.write(f"  Std:    {np.std(cc_energies):8.2f} MeV\n")
        f.write(f"  Range:  {np.min(cc_energies):8.2f} - {np.max(cc_energies):.2f} MeV\n")
        f.write("\n")
        
        f.write("ENERGY CUT ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Energy threshold: {energy_cut_results['threshold']:.2f} MeV\n")
        f.write("\n")
        f.write("Performance with simple cut (E > 3.0 MeV = ES):\n")
        f.write(f"  ES efficiency (recall):    {energy_cut_results['recall']:6.1%}\n")
        f.write(f"  CC rejection:              {energy_cut_results['cc_rejection']:6.1%}\n")
        f.write(f"  Accuracy:                  {energy_cut_results['accuracy']:6.1%}\n")
        f.write(f"  Precision:                 {energy_cut_results['precision']:6.1%}\n")
        f.write(f"  F1 Score:                  {energy_cut_results['f1_score']:6.3f}\n")
        f.write("\n")
        f.write(f"  ES clusters kept:  {energy_cut_results['es_kept']}/{energy_cut_results['es_total']}\n")
        f.write(f"  ES clusters lost:  {energy_cut_results['es_total'] - energy_cut_results['es_kept']}\n")
        f.write(f"  CC clusters kept:  {energy_cut_results['cc_kept']}/{energy_cut_results['cc_total']}\n")
        f.write("\n")
        
        ml_metrics = data['ml_metrics']
        f.write("ML CLASSIFIER PERFORMANCE (v27):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Accuracy:                  {ml_metrics['accuracy']:6.1%}\n")
        f.write(f"  Precision:                 {ml_metrics['precision']:6.1%}\n")
        f.write(f"  Recall:                    {ml_metrics['recall']:6.1%}\n")
        f.write(f"  F1 Score:                  {ml_metrics['f1_score']:6.3f}\n")
        f.write(f"  AUC-ROC:                   {ml_metrics['auc']:6.1%}\n")
        f.write("\n")
        
        f.write("COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<20} {'Energy Cut':>12} {'ML Classifier':>15} {'Difference':>12}\n")
        f.write("-" * 80 + "\n")
        
        metrics_pairs = [
            ('Accuracy', energy_cut_results['accuracy'], ml_metrics['accuracy']),
            ('Precision', energy_cut_results['precision'], ml_metrics['precision']),
            ('Recall', energy_cut_results['recall'], ml_metrics['recall']),
            ('F1 Score', energy_cut_results['f1_score'], ml_metrics['f1_score']),
        ]
        
        for name, energy_val, ml_val in metrics_pairs:
            diff = ml_val - energy_val
            diff_str = f"{diff:+.1%}" if abs(diff) > 0.001 else "~0"
            f.write(f"{name:<20} {energy_val:>11.1%} {ml_val:>14.1%} {diff_str:>12}\n")
        
        f.write("\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n")
        
        acc_diff = ml_metrics['accuracy'] - energy_cut_results['accuracy']
        recall_diff = ml_metrics['recall'] - energy_cut_results['recall']
        saved_mt = recall_diff * energy_cut_results['es_total']
        
        f.write(f"1. Accuracy improvement: {acc_diff:+.1%}\n")
        f.write(f"   ML classifier is {'' if acc_diff > 0 else 'NOT '}more accurate\n")
        f.write("\n")
        f.write(f"2. Main Track preservation: {saved_mt:+.0f} clusters ({recall_diff:+.1%})\n")
        f.write(f"   ML saves {saved_mt:.0f} more ES clusters compared to energy cut\n")
        f.write("\n")
        f.write(f"3. Lost Main Tracks with energy cut: {energy_cut_results['es_total'] - energy_cut_results['es_kept']:.0f}\n")
        f.write(f"   This is {(1 - energy_cut_results['recall']):.1%} of all ES clusters\n")
        f.write("\n")
        
        f.write("CONCLUSION:\n")
        f.write("-" * 80 + "\n")
        
        if acc_diff > 0.05:
            f.write("✓ ML classifier provides SIGNIFICANT improvement (>5% accuracy gain)\n")
            f.write("  RECOMMENDATION: Use ML classifier - worth the complexity\n")
        elif acc_diff > 0.02:
            f.write("? ML classifier provides MODEST improvement (2-5% accuracy gain)\n")
            f.write("  RECOMMENDATION: Marginal benefit - consider operational complexity\n")
        else:
            f.write("✗ ML classifier provides MINIMAL improvement (<2% accuracy gain)\n")
            f.write("  RECOMMENDATION: Simple energy cut may suffice for most use cases\n")
        
        f.write("\n")
        f.write("ADDITIONAL CONSIDERATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("• Energy cut is simpler, faster, and more interpretable\n")
        f.write("• ML classifier can capture more complex patterns beyond energy\n")
        f.write("• Energy cut has no training overhead or model maintenance\n")
        f.write("• ML classifier may be more robust to variations in data\n")
        f.write("\n")
        
    print(f"✓ Saved report: {report_path}")

def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("MAIN TRACK IDENTIFICATION: ENERGY CUT VS ML CLASSIFIER ANALYSIS")
    print("=" * 80 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_mt_data()
    
    # Analyze energy distributions
    es_energies, cc_energies = analyze_energy_distributions(data)
    
    # Find optimal energy cut
    energy_cut_results = find_energy_cut_for_background_rejection(data, threshold=3.0)
    
    # Compare methods
    compare_methods(data, energy_cut_results)
    
    # Generate plots
    plot_energy_distributions(data, energy_cut_results, es_energies, cc_energies)
    
    # Save report
    save_analysis_report(data, energy_cut_results, es_energies, cc_energies)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  • energy_cut_vs_ml_analysis.png")
    print("  • energy_cut_analysis_report.txt")
    print()

if __name__ == "__main__":
    main()
