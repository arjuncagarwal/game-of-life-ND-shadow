"""Stateless projection functions for N-dimensional grids."""

import numpy as np


def shadow_binary(grid: np.ndarray, axis: int) -> np.ndarray:
    """Project grid onto (N-1)-D plane: 1 where any cell is alive along axis."""
    return np.any(grid > 0, axis=axis).astype(np.uint8)


def shadow_density(grid: np.ndarray, axis: int) -> np.ndarray:
    """Project grid onto (N-1)-D plane: count of alive cells along axis."""
    return np.sum(grid, axis=axis)
