# src/financial_ml/evaluation/analyze.py
"""
Load trained models and run feature importance analysis.
"""

import joblib
from pathlib import Path
from financial_ml.evaluation.feature_analysis import analyze_feature_importance
from financial_ml.models.definitions import get_models
from financial_ml.utils.paths import get_model_file
from financial_ml.utils.config import MODELS_DIR

def analyze_models(args):
    """
    Load trained models from disk and analyze feature importance.
    
    Args:
        args: Namespace with debug/test flags
        
    Returns:
        None (prints analysis and saves plots)
    """
    # Determine which model directory to load from
        
    # Load all saved models
    trained_models = {}
    model_keys = get_models().keys()
    
    for model_name in model_keys:
        model_path = get_model_file(args, model_name)
        if model_path.exists():
            trained_models[model_name] = joblib.load(model_path)
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ⚠️  {model_name} not found")
    
    if not trained_models:
        print("\n❌ No models could be loaded!")
        return 1
    
    # Load feature names
    feature_path = MODELS_DIR / "feature_names.txt"
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            input_keys = [line.strip() for line in f]
        print(f"  ✓ Loaded {len(input_keys)} feature names")
    else:
        print("\n⚠️  Warning: feature_names.txt not found, using defaults")
        from financial_ml.utils.config import MARKET_KEYS, FUNDA_KEYS
        input_keys = MARKET_KEYS + FUNDA_KEYS
    
    # Run analysis
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    analyze_feature_importance(
        models_dict=trained_models,
        feature_names=input_keys
    )
    
    print(f"\n✓ Analysis complete! Charts saved to figures/")
    return 0
