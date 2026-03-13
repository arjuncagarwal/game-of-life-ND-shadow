"""Wireframe shadow casting renderer.

Each live cell is a unit-cube wireframe. A point light projects the edges
onto the floor plane (y=0). The display is the top-down view of that floor.
Overlapping shadow lines accumulate additively.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pygame

from backend.config import SimConfig


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

def cube_edge_offsets() -> np.ndarray:
    """Return (12, 2, 3) int8 array of unit-cube edge endpoint offsets."""
    corners = np.array(
        [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=np.int8,
    )  # (8, 3)
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if np.sum(a != b) == 1:   # differ in exactly one axis → edge
                edges.append([a, b])
    return np.array(edges, dtype=np.int8)  # (12, 2, 3)


_EDGE_OFFSETS = cube_edge_offsets()


# ---------------------------------------------------------------------------
# Core shadow cast
# ---------------------------------------------------------------------------

def cast_shadow(grid: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Project wireframe cell edges onto the floor via point-light shadow.

    Args:
        grid: 3D uint8 array (Nx, Ny, Nz). Must be exactly 3D.
        out_size: (W, H) pixel dimensions of the floor image.

    Returns:
        float32 accumulator of shape (H, W), values ≥ 0.
    """
    W, H = out_size
    N = max(grid.shape)

    # Light position: above the far diagonal corner
    light = np.array([1.5 * N, 3.0 * N, 1.5 * N], dtype=np.float64)

    # Floor display extent (world x,z coords → pixel x,y)
    margin = 0.2 * N
    x_min, x_max = -margin, N + margin
    z_min, z_max = -margin, N + margin

    # Live cell indices: (n_live, 3)
    live = np.argwhere(grid > 0).astype(np.float64)
    if len(live) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Build all edge world-space endpoints
    # live:        (n_live, 3)       → (n_live, 1, 1, 3)
    # _EDGE_OFFSETS:(12, 2, 3)       → (1, 12, 2, 3)
    # result:                           (n_live, 12, 2, 3)
    offsets = _EDGE_OFFSETS.astype(np.float64)
    pts = live[:, None, None, :] + offsets[None, :, :, :]
    # Shift cells up by 1 so y ∈ [1, Ny+1], floor at y=0
    pts[:, :, :, 1] += 1.0

    # Flatten to (E, 2, 3) where E = n_live * 12
    pts = pts.reshape(-1, 2, 3)  # (E, 2, 3)

    # Project each endpoint onto floor y=0 from point light
    # Shadow formula: t = ly/(ly - py);  Sx = lx + t*(px-lx);  Sz = lz + t*(pz-lz)
    lx, ly, lz = light
    py = pts[:, :, 1]               # (E, 2)
    # Guard: if a point is above the light, clamp (shouldn't happen with y-shift)
    denom = np.clip(ly - py, 1e-6, None)
    t = ly / denom                  # (E, 2)
    Sx = lx + t * (pts[:, :, 0] - lx)  # (E, 2)
    Sz = lz + t * (pts[:, :, 2] - lz)  # (E, 2)

    # Map world (x,z) → pixel (col, row)
    col = ((Sx - x_min) / (x_max - x_min) * W)  # (E, 2)
    row = ((Sz - z_min) / (z_max - z_min) * H)  # (E, 2)

    # Sample T points along each line segment
    T = 50
    alpha = np.linspace(0.0, 1.0, T)             # (T,)
    # Interpolate: (E, T)
    col_s = col[:, 0:1] + alpha[None, :] * (col[:, 1:2] - col[:, 0:1])
    row_s = row[:, 0:1] + alpha[None, :] * (row[:, 1:2] - row[:, 0:1])

    # Round and clip to valid pixel range
    col_s = np.clip(np.round(col_s).astype(np.int32), 0, W - 1)
    row_s = np.clip(np.round(row_s).astype(np.int32), 0, H - 1)

    # Flatten and count via bincount (fast additive accumulation)
    flat = (row_s * W + col_s).ravel()
    acc = np.bincount(flat, minlength=H * W).reshape(H, W).astype(np.float32)

    # Soft edge via Gaussian blur
    acc = gaussian_filter(acc, sigma=1.0)
    return acc


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_cast_shadow(
    acc: np.ndarray,
    surface: pygame.Surface,
    lut: np.ndarray,
) -> None:
    """Normalise accumulator, apply LUT, blit to surface.

    Args:
        acc: float32 (H, W) shadow accumulator.
        surface: pygame Surface of size (W, H).
        lut: (256, 3) uint8 colormap table.
    """
    max_val = acc.max()
    if max_val > 0:
        # Invert: floor is white (255), heavier shadow → darker
        gray = (255 - acc / max_val * 255).astype(np.uint8)
    else:
        gray = np.full(acc.shape, 255, dtype=np.uint8)

    # Grayscale RGB — no colormap, keeps shadows neutral
    rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)
    # pygame surfarray expects (W, H, 3) — transpose axes 0 and 1
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
