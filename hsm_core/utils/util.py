import numpy as np

def numpy_to_python(obj):
    """Convert numpy types to native Python types recursively."""
    if isinstance(obj, np.ndarray):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj