"""Colormap LUT builder for density shadow rendering."""

import numpy as np
import matplotlib.cm as mcm

COLORMAPS = ['inferno', 'plasma', 'viridis', 'magma', 'hot', 'cool', 'bone', 'turbo']


def build_lut(cmap_name: str) -> np.ndarray:
    """Return a (256, 3) uint8 LUT for the named matplotlib colormap."""
    cmap = mcm.get_cmap(cmap_name)
    indices = np.linspace(0, 1, 256)
    rgba = cmap(indices)            # (256, 4) float64 in [0, 1]
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb
