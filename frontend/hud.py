"""HUD overlay rendered on top of the shadow surface."""

from __future__ import annotations

import pygame

from backend.config import SimConfig
from backend.state import SimState


_font: pygame.font.Font | None = None
_PAD = 6
_LINE_H = 18


def _get_font() -> pygame.font.Font:
    global _font
    if _font is None:
        _font = pygame.font.SysFont('monospace', 14)
    return _font


def draw(
    surface: pygame.Surface,
    state: SimState,
    config: SimConfig,
    mode: str,
    fps: float,
) -> None:
    """Render HUD text in the top-left corner with a semi-transparent backdrop."""
    if not config.show_hud:
        return

    font = _get_font()
    population = int(state.grid.sum())
    total = state.grid.size
    shadow_fill = population / total * 100 if total else 0.0

    lines = [
        f"gen   {state.generation}",
        f"rule  {state.ruleset.notation()}",
        f"pop   {population}",
        f"fill  {shadow_fill:.1f}%",
        f"fps   {fps:.1f}",
        f"axis  {config.projection_axis}",
        f"mode  {config.projection_mode}",
        f"sim   {mode}",
    ]

    w = max(font.size(line)[0] for line in lines) + _PAD * 2
    h = len(lines) * _LINE_H + _PAD * 2

    backdrop = pygame.Surface((w, h), pygame.SRCALPHA)
    backdrop.fill((0, 0, 0, 160))
    surface.blit(backdrop, (0, 0))

    for i, line in enumerate(lines):
        text_surf = font.render(line, True, (220, 220, 220))
        surface.blit(text_surf, (_PAD, _PAD + i * _LINE_H))
