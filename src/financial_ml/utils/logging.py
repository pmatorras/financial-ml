"""Utilities for saving training logs and summaries"""
import numpy as np
from datetime import datetime
from pathlib import Path


from financial_ml.utils.paths import get_dir
from financial_ml.utils.config import SEPARATOR_WIDTH
def save_training_summary(fold_results, input_keys, args, output_dir=None):
    """Write fold results dictionary to clean text file
    
    Args:
        fold_results: Dict of {model_name: [fold_info_dicts]}
        input_keys: List of feature names
        args: Command-line arguments namespace
        output_dir: Path object for output directory
    
    Returns:
        Path to saved summary file
    """
    if output_dir is None: output_dir = get_dir(args, 'model')

    summary_file = output_dir / f"training_summary.txt"
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * SEPARATOR_WIDTH + "\n")
        f.write(f"TRAINING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * SEPARATOR_WIDTH + "\n\n")
        
        # Features
        f.write(f"Features ({len(input_keys)}): {', '.join(input_keys)}\n\n")
        
        # Results per model
        for model_name, folds in fold_results.items():
            f.write(f"{model_name.upper()}\n")
            f.write("-" * SEPARATOR_WIDTH + "\n")
            
            # Per-fold results
            for fold in folds:
                f.write(f"  Fold {fold['fold']}: "
                       f"{fold['train_start']} -> {fold['train_end']} | "
                       f"{fold['test_start']} -> {fold['test_end']}\n")
                f.write(f"    Train AUC: {fold['train_auc']:.3f}  "
                       f"Test AUC: {fold['test_auc']:.3f}  "
                       f"Test samples: {fold['test_samples']:,}\n")
            
            # Summary stats
            test_aucs = [f['test_auc'] for f in folds]
            train_aucs = [f['train_auc'] for f in folds]
            f.write(f"  Mean: Train {np.mean(train_aucs):.3f}  "
                   f"Test {np.mean(test_aucs):.3f}  "
                   f"Std {np.std(test_aucs):.3f}\n\n")
        
        f.write("=" * SEPARATOR_WIDTH + "\n")
    
    print(f"Summary saved to {summary_file}")
    return summary_file
