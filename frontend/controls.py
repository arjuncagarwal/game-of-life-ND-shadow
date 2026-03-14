"""Keyboard event dispatch for life-shadow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from backend.config import SimConfig
from backend.engine import seed_random, seed_centered_blob
from backend.rules import PRESETS
from backend.state import SimState, save_state
from frontend.colormap import COLORMAPS

if TYPE_CHECKING:
    pass

# Ordered list of presets for EXPLORING mode cycling
_PRESET_CYCLE = list(PRESETS.values())


def handle_events(
    events: list[pygame.event.Event],
    state: SimState,
    config: SimConfig,
    mode_ref: list[str],   # ['RUNNING'] | ['PAUSED'] | ['EXPLORING']
    step_fn,               # callable: () -> None — advances one generation
) -> bool:
    """Process pygame events. Returns True if the app should quit."""
    for event in events:
        if event.type == pygame.QUIT:
            return True

        if event.type != pygame.KEYDOWN:
            continue

        key = event.key

        if key == pygame.K_q:
            return True

        elif key == pygame.K_SPACE:
            mode_ref[0] = 'PAUSED' if mode_ref[0] == 'RUNNING' else 'RUNNING'

        elif key == pygame.K_RIGHT:
            if mode_ref[0] == 'PAUSED':
                step_fn()

        elif key == pygame.K_x:
            config.projection_axis = 0

        elif key == pygame.K_y:
            config.projection_axis = 1

        elif key == pygame.K_z:
            config.projection_axis = min(2, config.ndim - 1)

        elif key == pygame.K_d:
            _cycle = ['binary', 'density', 'depth_trails']
            config.projection_mode = _cycle[(_cycle.index(config.projection_mode) + 1) % len(_cycle)]

        elif key == pygame.K_c:
            idx = COLORMAPS.index(config.colormap) if config.colormap in COLORMAPS else 0
            config.colormap = COLORMAPS[(idx + 1) % len(COLORMAPS)]

        elif key == pygame.K_r:
            _reseed(state, config)

        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            config.sim_rate = min(60, config.sim_rate + 1)

        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            config.sim_rate = max(1, config.sim_rate - 1)

        elif key == pygame.K_g:
            config.show_gridlines = not config.show_gridlines

        elif key == pygame.K_s:
            _save_snapshot(state)

        elif key == pygame.K_e:
            mode_ref[0] = 'RUNNING' if mode_ref[0] == 'EXPLORING' else 'EXPLORING'

        elif key == pygame.K_h:
            config.show_hud = not config.show_hud

        elif key == pygame.K_k:
            config.show_keybindings = not config.show_keybindings

    return False


def _reseed(state: SimState, config: SimConfig) -> None:
    if config.seed_mode == 'blob':
        state.grid = seed_centered_blob(config.dimensions, config.blob_radius, config.seed_density)
    else:
        state.grid = seed_random(config.dimensions, config.seed_density)
    state.generation = 0


def _save_snapshot(state: SimState) -> None:
    presets_dir = Path('presets')
    presets_dir.mkdir(exist_ok=True)
    path = presets_dir / f"gen{state.generation}_{state.ruleset.name.replace('/', '_')}.npz"
    save_state(state, path)
    print(f"Saved snapshot → {path}")


def explore_advance(state: SimState, config: SimConfig, explore_counter: list[int]) -> None:
    """In EXPLORING mode, auto-cycle rules every 100 generations."""
    explore_counter[0] += 1
    if explore_counter[0] % 100 == 0:
        idx = (_PRESET_CYCLE.index(state.ruleset) + 1) % len(_PRESET_CYCLE)
        state.ruleset = _PRESET_CYCLE[idx]
        _reseed(state, config)
