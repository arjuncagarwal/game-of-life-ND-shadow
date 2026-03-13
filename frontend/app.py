"""Main application loop for life-shadow."""

from __future__ import annotations

import pygame
import numpy as np

from backend.config import SimConfig
from backend.engine import step
from backend.projection import shadow_binary, shadow_density
from backend.state import SimState
from frontend.colormap import build_lut
from frontend.renderer import render_shadow
from frontend import hud, controls


class App:
    """Pygame application that runs the CA and displays the shadow projection."""

    def __init__(self, state: SimState) -> None:
        self.state = state
        self.config = state.config

        # App state machine
        self._mode: list[str] = ['RUNNING']   # mutable ref passed to controls
        self._accumulator: float = 0.0
        self._explore_counter: list[int] = [0]

        pygame.init()
        pygame.font.init()

        w, h = self._window_size()
        self._surface = pygame.display.set_mode((w, h))
        pygame.display.set_caption('life-shadow')

        self._clock = pygame.time.Clock()
        self._lut = build_lut(self.config.colormap)
        self._prev_cmap = self.config.colormap

    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            dt = self._clock.tick(60) / 1000.0
            fps = self._clock.get_fps()

            events = pygame.event.get()
            should_quit = controls.handle_events(
                events,
                self.state,
                self.config,
                self._mode,
                self._do_step,
            )
            if should_quit:
                break

            mode = self._mode[0]

            if mode in ('RUNNING', 'EXPLORING'):
                self._accumulator += dt
                step_interval = 1.0 / max(1, self.config.sim_rate)
                while self._accumulator >= step_interval:
                    self._do_step()
                    self._accumulator -= step_interval
                    if mode == 'EXPLORING':
                        controls.explore_advance(self.state, self.config, self._explore_counter)

            # Rebuild LUT if colormap changed
            if self.config.colormap != self._prev_cmap:
                self._lut = build_lut(self.config.colormap)
                self._prev_cmap = self.config.colormap

            # Resize window if shadow shape changed (e.g. axis switch)
            expected_w, expected_h = self._window_size()
            if (expected_w, expected_h) != self._surface.get_size():
                self._surface = pygame.display.set_mode((expected_w, expected_h))

            shadow = self._project()
            render_shadow(shadow, self._surface, self.config, self._lut)
            hud.draw(self._surface, self.state, self.config, mode, fps)
            pygame.display.flip()

        pygame.quit()

    # ------------------------------------------------------------------

    def _do_step(self) -> None:
        self.state.grid = step(
            self.state.grid,
            self.state.ruleset.birth,
            self.state.ruleset.survival,
        )
        self.state.generation += 1

    def _project(self) -> np.ndarray:
        """Collapse the grid to a 2D shadow, double-projecting for 4D+."""
        grid = self.state.grid
        axis = self.config.projection_axis
        mode = self.config.projection_mode

        shadow = _project_once(grid, axis, mode)

        # Double-project if still >2D (e.g. 4D grid)
        while shadow.ndim > 2:
            shadow = _project_once(shadow, -1, mode)

        return shadow

    def _window_size(self) -> tuple[int, int]:
        sh = self.config.shadow_shape
        cell = self.config.cell_size
        # shadow_shape is (H, W); window is (W*cell, H*cell)
        return sh[1] * cell, sh[0] * cell


def _project_once(grid: np.ndarray, axis: int, mode: str) -> np.ndarray:
    if mode == 'binary':
        return shadow_binary(grid, axis)
    return shadow_density(grid, axis)
