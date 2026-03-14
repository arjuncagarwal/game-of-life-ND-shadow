# life-shadow

Runs Conway-style cellular automata in N dimensions and visualizes the result as an (N−1)-dimensional shadow projection. The primary target is 3D → 2D, with 4D → 3D supported via double-projection.

Two render modes:

- **Wireframe shadow cast** (default) — each live cell is a wireframe cube; a point light above the volume casts edge shadows onto a floor plane. Overlapping shadows accumulate and are softened with a Gaussian blur.
- **Heatmap** (`--heatmap`) — mathematical projection along one axis, displayed as a colormapped 2D array. Supports three projection modes: binary, density, and depth-trails.

## Install

```
pip install -e ".[dev]"
```

Requires Python 3.10+, numpy, scipy, pygame, matplotlib.

## Run

```
python main.py                                             # 3D, 32³, wireframe shadow cast
python main.py --dims 64 64 64 --rule B5/S567             # larger grid, custom rule
python main.py --heatmap                                   # heatmap projection
python main.py --heatmap --projection-mode depth_trails    # depth + trail mode
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dims` | `32 32 32` | Grid dimensions (any number of ints for N-D) |
| `--rule` | `B5/S567` | Birth/survival rule in B/S notation |
| `--density` | `0.15` | Initial seed density |
| `--seed-mode` | `blob` | `random` or `blob` |
| `--blob-radius` | `8` | Radius of centered blob seed |
| `--projection-axis` | `-1` | Axis to collapse (−1 = last) |
| `--projection-mode` | `density` | `binary`, `density`, or `depth_trails` |
| `--sim-rate` | `10` | Simulation steps per second |
| `--colormap` | `inferno` | Any matplotlib colormap name |
| `--cell-size` | `32` | Pixels per cell |
| `--heatmap` | off | Use heatmap render instead of wireframe shadow cast |

## Keybindings

| Key | Action |
|---|---|
| Space | Play / pause |
| → | Single step (while paused) |
| D | Cycle projection mode: binary → density → depth_trails |
| X / Y / Z | Set projection axis |
| C | Cycle colormap |
| R | Re-seed grid |
| E | Toggle explore mode (auto-cycles rules every 100 generations) |
| +/− | Adjust sim rate |
| G | Toggle gridlines |
| S | Save snapshot to `presets/` |
| H | Toggle HUD |
| K | Toggle keybindings overlay |
| Q | Quit |

## Projection modes (heatmap)

| Mode | Description |
|---|---|
| `binary` | White where any cell is alive along the axis |
| `density` | Brightness proportional to cell count, colormapped |
| `depth_trails` | HSV coloring: hue = depth centroid, value = density. An exponential decay buffer (factor 0.85) accumulates past frames as desaturated ghost trails behind the current frame. |

## Preset rules

| Name | Notation | Character |
|---|---|---|
| `conway_2d` | B3/S23 | Classic 2D Life |
| `3d_amoeba` | B5/S567 | Organic growth in 3D |
| `3d_crystal` | B4/S34 | Stable crystalline structures |
| `3d_coral` | B6,7,8/S5,6,7,8 | Slow dendritic growth |
| `3d_clouds` | B14-19/S13+ | Converges to stable blobs at ~50% density |

Custom rules use B/S notation: `B3/S23`, `B5/S567`, `B14-19/S13-26`. Ranges and comma lists are supported.

## Project structure

```
backend/    — Pure computation (engine, rules, projection, config, state). No pygame.
frontend/   — Pygame display, shadow cast renderer, heatmap renderer, HUD, controls.
tests/      — pytest suite (27 tests).
presets/    — Saved .npz snapshots.
docs/       — SPEC.md with full module signatures and data flow.
main.py     — Entry point.
```

## Tests

```
python -m pytest tests/
```

## Extension: 4D

Pass four dimensions: `--dims 16 16 16 16`. The backend handles it with zero changes. The frontend double-projects 4D → 3D → 2D for display.
