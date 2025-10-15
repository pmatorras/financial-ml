import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_sanitize():
    return FunctionTransformer(lambda X: np.where(np.isfinite(X), X, np.nan), validate=False)

def get_models():
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
            ("clf", RandomForestClassifier(n_estimators=50, 
                                           max_depth=4,
                                           min_samples_split=0.01,
                                           min_samples_leaf=0.005,
                                           max_features='log2',#'sqrt',
                                           random_state=42,
                                           n_jobs=-1, 
                                           class_weight="balanced"

                                        ))
        ])
    }
    return models