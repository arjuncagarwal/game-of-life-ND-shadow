"""SimConfig dataclass for life-shadow."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimConfig:
    dimensions: tuple[int, ...]
    projection_axis: int = -1          # -1 means last axis
    projection_mode: Literal['binary', 'density'] = 'density'
    seed_density: float = 0.15
    seed_mode: Literal['random', 'blob'] = 'blob'
    blob_radius: int = 8
    sim_rate: int = 10                 # steps per second
    colormap: str = 'inferno'
    cell_size: int = 8                 # pixels per cell
    show_hud: bool = True
    show_gridlines: bool = False

    def __post_init__(self) -> None:
        self.dimensions = tuple(self.dimensions)
        if self.projection_axis == -1:
            self.projection_axis = len(self.dimensions) - 1

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    @property
    def shadow_shape(self) -> tuple[int, ...]:
        """Shape of the 2D projection output used for display."""
        dims = list(self.dimensions)
        # Remove the projection axis; if result is >2D, remove last axis again
        remaining = [d for i, d in enumerate(dims) if i != self.projection_axis]
        # For 4D grids, double-project to 2D
        while len(remaining) > 2:
            remaining.pop()
        return tuple(remaining)
