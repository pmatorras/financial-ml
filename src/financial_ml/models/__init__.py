"""
Machine learning model training and definitions.

Main functions:
- train(): Train models with cross-validation
- get_models(): Get configured model pipelines
- get_model_name(): Get properly format model names

Usage:
    from financial_ml.models import train, get_models
    
    models = get_models()
    predictions, trained_models = train(args)
"""

# Main training function
from financial_ml.models.training import train

# Model definitions
from financial_ml.models.definitions import get_models, get_model_name

__all__ = [
    'train',           # Main training function
    'get_models',      # Get model pipelines
    'get_model_name',  # Format model names
]

__version__ = '0.1.0'
