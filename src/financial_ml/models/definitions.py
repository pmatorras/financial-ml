import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

MODEL_METADATA = {
    "logreg_l1": {
        "name": "Logistic Regression (L1)",
        "short_name": "L1 Logreg",
        "description": "L1-regularized logistic regression with feature selection",
        "color": "#1f77b4",  # For future implemmentation
        "marker": "o"
    },
    "logreg_l2": {
        "name": "Logistic Regression (L2)",
        "short_name": "L2 Logreg",
        "description": "L2-regularized logistic regression (ridge)",
        "color": "#ff7f0e",
        "marker": "s"
    },
    "rf": {
        "name": "Random Forest",
        "short_name": "RF",
        "description": "Random forest classifier with balanced class weights",
        "color": "#2ca02c",
        "marker": "^"
    }
}

def get_model_name (model_key, short_name=False):
    """
    Get the display name for a model
    Args:
        model_key: Model identifier (has to be one of MODEL_METADATA keys)
        short_name: Display the short name or not 
    Returns:
        Display name/short name for the model
    """
    key = "short_name" if short_name else "name"
    return MODEL_METADATA.get(model_key, {}).get(key, model_key)

# Define function at module level (not inside another function)
def _sanitize_infinities(X):
    """Replace ±inf with NaN for sklearn imputation."""
    return np.where(np.isfinite(X), X, np.nan)


def build_sanitize():
    """Create transformer to replace infinite values with NaN."""
    return FunctionTransformer(_sanitize_infinities, validate=False)

def get_models(args):
    sanitize = build_sanitize()
    models = {
        "logreg_l1": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                    C=0.5, max_iter=5000, class_weight="balanced"))
        ])
        ,
        "logreg_l2": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        ]),

        "rf": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scaler", "passthrough"),  # trees don"t need scaling
            ("clf", RandomForestClassifier(n_estimators=args.tree_nestimators, 
                                           max_depth=args.tree_depth,
                                           min_samples_split=0.02,
                                           min_samples_leaf=0.01,
                                           max_samples=args.tree_max_samples,
                                           max_features=args.tree_max_features,
                                           random_state=42,
                                           n_jobs=-1, 
                                           class_weight="balanced"

                                        ))
        ]),
        "rf_cal": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scaler", "passthrough"),  # trees don"t need scaling
            ("clf", CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    min_samples_split=0.02,
                    min_samples_leaf=0.01,
                    max_features='log2',
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                ),
                method='isotonic',
                cv=3
            ))
        ])
    }
    return models

    '''
    'gb': Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.01,
            subsample=0.5,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42
        ))
    ])
    '''