from pathlib import Path
from typing import Union


def to_relative_path(path: Union[str, Path], root: Union[str, Path]) -> str:
    """Convert an absolute path to a relative path with respect to root.
    Args:
        path: The absolute path to convert.
        root: The root directory to make the path relative to.
    Returns:
        str: The relative path.
    Raises:
        ValueError: If the path is not under the root.
    """
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        rel = path.relative_to(root)
        return str(rel)
    except ValueError:
        raise ValueError(f"Path {path} is not under root {root}")


def to_absolute_path(rel_path: Union[str, Path], root: Union[str, Path]) -> Path:
    """Convert a relative path to an absolute path with respect to root, with traversal protection.
    Args:
        rel_path: The relative path to convert.
        root: The root directory to resolve from.
    Returns:
        Path: The absolute path.
    Raises:
        ValueError: If the resolved path escapes the root.
    """
    abs_path = (Path(root) / rel_path).resolve()
    if not str(abs_path).startswith(str(Path(root).resolve())):
        raise ValueError(f"Resolved path {abs_path} escapes root {root}")
    return abs_path 