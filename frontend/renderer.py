"""Shadow rendering to a pygame Surface."""

import numpy as np
import pygame

from backend.config import SimConfig


def render_shadow(
    shadow: np.ndarray,
    surface: pygame.Surface,
    config: SimConfig,
    lut: np.ndarray,
) -> None:
    """Blit the shadow array onto surface.

    Args:
        shadow: 2D array of shape (H, W). Binary: 0/1. Density: int counts.
        surface: pygame Surface, must be (W*cell_size, H*cell_size).
        config: SimConfig for projection_mode, cell_size, show_gridlines.
        lut: (256, 3) uint8 colormap lookup table.
    """
    cell = config.cell_size
    h, w = shadow.shape

    if config.projection_mode == 'binary':
        # 0 → black, 1 → white
        mono = (shadow * 255).astype(np.uint8)                # (H, W)
        rgb = np.stack([mono, mono, mono], axis=-1)           # (H, W, 3)
    else:
        max_val = shadow.max()
        if max_val > 0:
            normalized = (shadow.astype(np.float32) / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(shadow, dtype=np.uint8)
        rgb = lut[normalized]                                  # (H, W, 3)

    # Scale up: repeat each cell pixel cell×cell times
    scaled = np.kron(rgb, np.ones((cell, cell, 1), dtype=np.uint8))  # (H*cell, W*cell, 3)

    if config.show_gridlines and cell >= 4:
        # Draw 1-pixel dark lines between cells (in-place on copy)
        scaled = scaled.copy()
        for row in range(0, h * cell, cell):
            scaled[row, :, :] = scaled[row, :, :] // 2
        for col in range(0, w * cell, cell):
            scaled[:, col, :] = scaled[:, col, :] // 2

    # pygame.surfarray.blit_array expects (width, height, 3) in x-major order
    pygame.surfarray.blit_array(surface, scaled.transpose(1, 0, 2))
