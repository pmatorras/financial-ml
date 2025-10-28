"""Custom argument validators for CLI."""
import argparse

def list_from_string(s):
    """Parses a comma-separated string into a list and removes whitespace."""
    return [item.strip() for item in s.split(',')]

def percentage(value):
    """Ensure input value is between 0 and 100."""
    fval = float(value)
    if not 0 <= fval <= 100:
        raise argparse.ArgumentTypeError(f"{value} is not a valid percentage (must be 0-100)")
    return fval

def max_features_type(arg):
    # Try converting to float
    try:
        f_val = float(arg)
        # Add a check here if the float needs to be in a certain range
        if 0.0 < f_val <= 1.0:
            return f_val
        else:
            raise ValueError(f"Float value {f_val} is out of the valid range (0.0, 1.0]")
    except ValueError:
        # If not a float, check for valid strings
        valid_strings = ['log2', 'sqrt', 'None']
        if arg in valid_strings:
            return arg
        else:
            # Raise an ArgumentTypeError if the value is invalid
            raise argparse.ArgumentTypeError(
                f"Invalid value: '{arg}'. Must be a float in (0.0, 1.0] or one of {valid_strings}")

def validate_max_samples(value):
    fval = float(value) if value != 'None' else None
    if fval is not None and not (0.0 < fval <= 1.0):
        raise argparse.ArgumentTypeError("max_samples must be in (0.0, 1.0] or None")
    return fval
