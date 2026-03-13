"""Core cellular automata computation. Dimension-agnostic."""

import numpy as np
from scipy.ndimage import convolve


def step(grid: np.ndarray, birth: set[int], survival: set[int]) -> np.ndarray:
    """Advance the grid by one generation.

    Args:
        grid: uint8 array of shape (N,)*d where nonzero = alive
        birth: neighbor counts that create a new cell
        survival: neighbor counts that keep a cell alive

    Returns:
        New grid (input not mutated).
    """
    kernel = np.ones((3,) * grid.ndim, dtype=np.uint8)
    center = tuple(1 for _ in range(grid.ndim))
    kernel[center] = 0

    neighbors = convolve(grid.astype(np.int32), kernel, mode='wrap')

    alive = grid > 0
    born = (~alive) & np.isin(neighbors, list(birth))
    survived = alive & np.isin(neighbors, list(survival))

    return (born | survived).astype(np.uint8)


def seed_random(shape: tuple[int, ...], density: float) -> np.ndarray:
    """Return a random uint8 grid with given alive probability."""
    return (np.random.random(shape) < density).astype(np.uint8)


def seed_centered_blob(shape: tuple[int, ...], radius: int, density: float) -> np.ndarray:
    """Random fill only within a centered hypercube of given radius."""
    grid = np.zeros(shape, dtype=np.uint8)
    slices = tuple(
        slice(max(0, s // 2 - radius), min(s, s // 2 + radius + 1))
        for s in shape
    )
    blob = (np.random.random(tuple(sl.stop - sl.start for sl in slices)) < density).astype(np.uint8)
    grid[slices] = blob
    return grid
