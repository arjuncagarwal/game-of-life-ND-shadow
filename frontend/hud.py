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


_KEYBINDINGS = [
    ("Space",   "play / pause"),
    ("→",       "step (paused)"),
    ("R",       "re-seed"),
    ("E",       "explore mode"),
    ("D",       "cycle projection mode"),
    ("X/Y/Z",   "projection axis"),
    ("C",       "cycle colormap"),
    ("+/-",     "sim rate"),
    ("G",       "gridlines"),
    ("S",       "save snapshot"),
    ("H",       "toggle HUD"),
    ("K",       "keybindings"),
    ("Q",       "quit"),
]


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

    stats = [
        f"gen   {state.generation}",
        f"rule  {state.ruleset.notation()}",
        f"pop   {population}",
        f"fill  {shadow_fill:.1f}%",
        f"fps   {fps:.1f}",
        f"axis  {config.projection_axis}",
        f"mode  {config.projection_mode}",
        f"sim   {mode}",
    ]

    kb_lines: list[str] = []
    if config.show_keybindings:
        kb_lines = ["", "  keybindings"] + [f"  {k:<7} {desc}" for k, desc in _KEYBINDINGS]

    all_lines = stats + kb_lines

    w = max(font.size(line)[0] for line in all_lines) + _PAD * 2
    h = len(all_lines) * _LINE_H + _PAD * 2

    backdrop = pygame.Surface((w, h), pygame.SRCALPHA)
    backdrop.fill((0, 0, 0, 160))
    surface.blit(backdrop, (0, 0))

    for i, line in enumerate(all_lines):
        # Dim the keybinding section slightly relative to stats
        color = (180, 180, 180) if i >= len(stats) else (220, 220, 220)
        text_surf = font.render(line, True, color)
        surface.blit(text_surf, (_PAD, _PAD + i * _LINE_H))
