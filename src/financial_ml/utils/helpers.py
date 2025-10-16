import numpy as np
import pandas as pd

def safe_div(numer, denom):
    '''Safely divide two values, handling division by zero.
        return nan if negative/zero denominators
    '''
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom
    return out.where((denom > 0) & np.isfinite(out))

