"""
Feature importance analysis for trained models.
Extracts and visualizes coefficients and feature importances.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from financial_ml.utils.config import FIGURE_DIR


def plot_rf_feature_importance(model, feature_names, save_path=FIGURE_DIR / "feature_importance_rf.png"):
    """
    Plot feature importance from trained Random Forest
    Args:
        model: Trained RandomForest model
        feature_names: List of feature names
        save_path: Path to save figure (optional)
        
    Returns:
        DataFrame with feature importances
    """
    
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': std
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['feature'], importance_df['importance'], 
            xerr=importance_df['std'], capsize=3)
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        print("Saving figure to:", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return importance_df

def plot_logistic_coefficients(pipeline, feature_names, save_path=FIGURE_DIR / "feature_importance_logistic.png"):
    """
    Plot coefficients from logistic regression
    Args:
        pipeline: Trained sklearn Pipeline with StandardScaler + LogisticRegression
        feature_names: List of feature names
        save_path: Path to save figure (optional)
        
    Returns:
        DataFrame with coefficients
    """
    
    # Extract the scaler and classifier from pipeline
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scale', None)
        clf = pipeline.named_steps.get('clf', None)
    else:
        # If not a pipeline, assume no scaling
        scaler = None
        clf = pipeline
    
    # Get coefficients
    coef_scaled = clf.coef_[0]
    
    # Unscale if scaler exists
    if scaler is not None and hasattr(scaler, 'scale_'):
        coef = coef_scaled / scaler.scale_  # Convert to original units
    else:
        coef = coef_scaled
        
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    }).sort_values('coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if c < 0 else 'green' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Logistic Regression Coefficients\n(Green = Predicts Outperformance, Red = Predicts Underperformance)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        print(f"Saving figure to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return coef_df

def analyze_feature_importance(models_dict, feature_names, fig_dir=FIGURE_DIR):
    """
    Analyze feature importance across all trained models
    
    Parameters:
    - models_dict: dict with keys like 'RandomForest', 'LogisticL2'
    - feature_names: list of feature names
    """
    for model_name, pipeline in models_dict.items():
        # Pipelines have steps, access the final step (the classifier)
        if hasattr(pipeline, 'named_steps'):
            # Get the last step name (usually 'model' or 'classifier')
            step_names = list(pipeline.named_steps.keys())
            final_step_name = step_names[-1]  # Last step is the model
            model = pipeline.named_steps[final_step_name]
        elif hasattr(pipeline, 'steps'):
            # Alternative: access via steps attribute
            model = pipeline.steps[-1][1]  # Last tuple is ('name', estimator)
        else:
            # If it's not a pipeline, use directly
            model = pipeline
        
        # If it's a CalibratedClassifierCV, unwrap to get the base estimator
        if hasattr(model, 'calibrated_classifiers_'):
            # CalibratedClassifierCV stores calibrated classifiers
            # Get the base estimator from the first calibrated classifier
            base_estimator = model.calibrated_classifiers_[0].estimator
            print(f"\n{model_name}: Detected CalibratedClassifierCV, accessing base estimator")
            model = base_estimator
            
        # Now extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Random Forest or tree-based model
            print(f"\n{'='*60}")
            print(f"Feature Importance: {model_name}")
            print(f"{'='*60}")
            importance_df = plot_rf_feature_importance(
                model, feature_names, 
                save_path=fig_dir/ f"importance_{model_name.lower()}.png"
            )
            print(importance_df.sort_values('importance', ascending=False).to_string(index=False))
            
        elif hasattr(model, 'coef_'):
            # Logistic Regression or linear model
            print(f"\n{'='*60}")
            print(f"Coefficients: {model_name}")
            print(f"{'='*60}")
            coef_df = plot_logistic_coefficients(
                pipeline, feature_names,
                save_path=fig_dir /f"coefficients_{model_name.lower()}.png"
            )
            print(coef_df.sort_values('abs_coefficient', ascending=False).to_string(index=False))
        else:
            print(f"\n{model_name}: Cannot extract feature importance (no coef_ or feature_importances_)")
