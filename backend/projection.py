"""Stateless projection functions for N-dimensional grids."""

import numpy as np


def shadow_binary(grid: np.ndarray, axis: int) -> np.ndarray:
    """Project grid onto (N-1)-D plane: 1 where any cell is alive along axis."""
    return np.any(grid > 0, axis=axis).astype(np.uint8)


def shadow_density(grid: np.ndarray, axis: int) -> np.ndarray:
    """Project grid onto (N-1)-D plane: count of alive cells along axis."""
    return np.sum(grid, axis=axis)


def shadow_depth_centroid(
    grid: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth_centroid, density) for depth_trails projection mode.

    depth_centroid: float32, same shape as shadow_density output.
        Normalized index-weighted centroid of live cells along `axis`, in [0, 1].
        Zero for columns with no live cells.
    density: float32, same shape.
        Count of live cells along `axis`, normalized by grid.shape[axis], in [0, 1].
        Zero for empty columns.
    """
    D = grid.shape[axis]
    g = np.moveaxis(grid, axis, -1).astype(np.float32)  # (..., D)
    count = g.sum(axis=-1)                               # (...,)
    indices = np.arange(D, dtype=np.float32)
    weighted = (g * indices).sum(axis=-1)                # (...,)

    denom = D - 1 if D > 1 else 1
    with np.errstate(invalid='ignore', divide='ignore'):
        centroid = np.where(count > 0, weighted / (count * denom), 0.0).astype(np.float32)
    density = (count / D).astype(np.float32)
    return centroid, density
