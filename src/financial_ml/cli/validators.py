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
