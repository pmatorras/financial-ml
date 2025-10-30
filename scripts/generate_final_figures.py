"""
Generate publication-ready figures for documentation

Usage:
    python scripts/generate_final_figures.py --model rf --do-sentiment
    python scripts/generate_final_figures.py --model xgb --market-only
"""

import argparse
import shutil
from pathlib import Path
from financial_ml.utils.paths import get_dir, get_fig_name

def copy_figure(src_dir, dest_dir, filename, new_name=None):
    """Copy figure from source to destination with optional rename"""
    src_path = src_dir / filename
    dest_path = dest_dir / (new_name or filename)
    print(src_dir, dest_dir, filename)
    if src_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"✅ Copied: {filename} -> {dest_path.name}")
        return True
    else:
        print(f"⚠️  Not found: {src_path}")
        return False

def generate_final_figures(args):
    """
    Copy model-specific figures to docs/images/ for documentation
    
    Args:
        model: Model name (rf, xgb, lgbm, etc.)
        do_sentiment: Whether sentiment features were used
    """
    # Get directories
    fig_dir = get_dir(args, 'figure')

    dest_dir = Path('docs/images')
    model =args.model
    print(f"\n{'='*60}")
    print(f"Generating Final Figures: {model}")
    print(f"Source: {fig_dir}")
    print(f"Destination: {dest_dir}")
    print(f"{'='*60}\n")
    
    # Essential figures to copy
    figures = {
        # Portfolio performance
        f'portfolio_performance.png': get_fig_name(fig_type='performance', model_name=model, p_type='100long', per_top=10),
        f'sector_drift.png': get_fig_name(fig_type='concentration', model_name=model),
        
        # Model analysis
        f'feature_importance.png': get_fig_name(fig_type='importance', model_name=model),
        'correlation_matrix.png': get_fig_name(fig_type='correlation', model_name=model),
        
    }
    
    copied = 0
    for dest_name, src_name in figures.items():
        if copy_figure(fig_dir, dest_dir, src_name, dest_name):
            copied += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Copied {copied}/{len(figures)} figures to {dest_dir}")
    print(f"{'='*60}\n")
    
    # Show what's in docs/images/
    if dest_dir.exists():
        print("\nFinal docs/images/ contents:")
        for img in sorted(dest_dir.glob('*.png')):
            print(f"  - {img.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures for documentation'
    )
    parser.add_argument('--model', type=str, default='rf',
                       help='Model name (rf, xgb, lgbm, gb, etc.)')
    
    # Sentiment flags (mutually exclusive)
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument('--do-sentiment', action='store_true',
                                help='Use sentiment features')
    sentiment_group.add_argument('--market-only', action='store_true',
                                help='Market features only (no sentiment)')
    
    args = parser.parse_args()
    
    # Default to sentiment if neither specified
    
    generate_final_figures(args)

if __name__ == '__main__':
    main()
