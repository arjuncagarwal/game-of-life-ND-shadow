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


def _hsv_to_rgb_vec(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV → RGB. All inputs float32 in [0, 1]. Returns float32 (..., 3) in [0, 1]."""
    hi = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros(h.shape + (3,), dtype=np.float32)
    for i, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        mask = hi == i
        rgb[mask, 0] = r[mask]
        rgb[mask, 1] = g[mask]
        rgb[mask, 2] = b[mask]
    return rgb


def render_depth_trails(
    depth: np.ndarray,
    density: np.ndarray,
    trail_buffer: np.ndarray,
    surface: pygame.Surface,
    config: SimConfig,
) -> None:
    """Render depth-encoded coloring with temporal trail overlay.

    Args:
        depth: float32 (H, W) in [0, 1] — depth centroid mapped to hue.
        density: float32 (H, W) in [0, 1] — normalized cell count mapped to HSV value.
        trail_buffer: float32 (H, W, 3) in [0, 255], mutated in-place (decay + accumulate).
        surface: pygame Surface, must be (W*cell_size, H*cell_size).
        config: SimConfig for cell_size, trail_decay, show_gridlines.
    """
    cell = config.cell_size
    h, w = depth.shape

    # --- Current frame: full-saturation HSV coloring ---
    sat = (density > 0).astype(np.float32)
    current_rgb = _hsv_to_rgb_vec(depth, sat, density)  # (H, W, 3) in [0, 1]
    current_rgb_255 = current_rgb * 255.0               # (H, W, 3) in [0, 255]

    # --- Update trail: decay then accumulate current frame ---
    trail_buffer *= config.trail_decay
    np.minimum(trail_buffer + current_rgb_255, 255.0, out=trail_buffer)

    # --- Composite: desaturated trail (ghost) + current frame at full saturation ---
    gray = trail_buffer.mean(axis=-1, keepdims=True)
    desaturated = 0.35 * trail_buffer + 0.65 * gray
    output_f = np.clip(desaturated + current_rgb_255, 0.0, 255.0)
    rgb = output_f.astype(np.uint8)

    # Scale up and blit
    scaled = np.kron(rgb, np.ones((cell, cell, 1), dtype=np.uint8))  # (H*cell, W*cell, 3)

    if config.show_gridlines and cell >= 4:
        scaled = scaled.copy()
        for row in range(0, h * cell, cell):
            scaled[row, :, :] = scaled[row, :, :] // 2
        for col in range(0, w * cell, cell):
            scaled[:, col, :] = scaled[:, col, :] // 2

    pygame.surfarray.blit_array(surface, scaled.transpose(1, 0, 2))
