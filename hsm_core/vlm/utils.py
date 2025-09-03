"""
VLM Utilities

Utility functions for VLM operations.
"""

import numpy as np


def round_nested_values(data, decimals=4):
    """
    Recursively round all float values in nested dictionaries, lists, and tuples to specified decimal places.

    Args:
        data: The input data structure (dict, list, tuple, or primitive type)
        decimals: Number of decimal places to round to (default: 4)

    Returns:
        The data structure with all float values rounded
    """
    if isinstance(data, dict):
        return {key: round_nested_values(value, decimals) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_nested_values(item, decimals) for item in data]
    elif isinstance(data, tuple):
        return tuple(round_nested_values(item, decimals) for item in data)
    elif isinstance(data, (float, np.float32, np.float64)):
        return round(float(data), decimals)
    return data
