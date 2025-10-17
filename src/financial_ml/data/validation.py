'''
Some data validation utilities
'''

import pandas as pd

def require_non_empty(df: pd.DataFrame, name: str, min_rows: int = 1, min_cols: int = 1):
    '''
    Validate that the input dataframe is non empty, and has a minimum shape. Raises a ValueError the number of rows or columns in the df is too small.
    '''
    # .empty is True if any axis has length 0; still False when all-NaN rows exist
    if df is None or df.empty:
        raise ValueError(f"{name} is empty (no rows or no columns).")  # fail fast
    n_rows, n_cols = df.shape
    if n_rows < min_rows or n_cols < min_cols:
        raise ValueError(f"{name} too small: shape={df.shape}, expected at least ({min_rows}, {min_cols}).")
    


def validate_date_column(df: pd.DataFrame, col: str = "date"):
    """
    Make sure date column is properly formatted and sorted.Raises an AssertionError: If dates are not monotonically increasing
    """
    df[col] = pd.to_datetime(df[col])
    df = df.sort_values(col)
    
    assert df[col].is_monotonic_increasing, f"{col} must be sorted ascending"
    
    return df