"""SimState dataclass and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import SimConfig
from backend.rules import Ruleset


@dataclass
class SimState:
    grid: np.ndarray
    generation: int
    ruleset: Ruleset
    config: SimConfig


def save_state(state: SimState, path: str | Path) -> None:
    """Persist state to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid=state.grid,
        generation=np.array(state.generation),
        ruleset_name=np.array(state.ruleset.name),
        ruleset_birth=np.array(sorted(state.ruleset.birth)),
        ruleset_survival=np.array(sorted(state.ruleset.survival)),
        dimensions=np.array(state.config.dimensions),
        projection_axis=np.array(state.config.projection_axis),
        projection_mode=np.array(state.config.projection_mode),
        seed_density=np.array(state.config.seed_density),
        seed_mode=np.array(state.config.seed_mode),
        blob_radius=np.array(state.config.blob_radius),
        sim_rate=np.array(state.config.sim_rate),
        colormap=np.array(state.config.colormap),
        cell_size=np.array(state.config.cell_size),
    )


def load_state(path: str | Path) -> SimState:
    """Load state from a .npz file."""
    data = np.load(path, allow_pickle=False)
    ruleset = Ruleset(
        name=str(data['ruleset_name']),
        birth=frozenset(int(x) for x in data['ruleset_birth']),
        survival=frozenset(int(x) for x in data['ruleset_survival']),
    )
    config = SimConfig(
        dimensions=tuple(int(x) for x in data['dimensions']),
        projection_axis=int(data['projection_axis']),
        projection_mode=str(data['projection_mode']),
        seed_density=float(data['seed_density']),
        seed_mode=str(data['seed_mode']),
        blob_radius=int(data['blob_radius']),
        sim_rate=int(data['sim_rate']),
        colormap=str(data['colormap']),
        cell_size=int(data['cell_size']),
    )
    return SimState(
        grid=data['grid'],
        generation=int(data['generation']),
        ruleset=ruleset,
        config=config,
    )
