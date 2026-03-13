# Architecture Spec — life-shadow

## Overview

A cellular automata engine that runs Conway-style Game of Life in N dimensions and visualizes
the result as an (N-1)-dimensional "shadow" projection. The shadow is the collapse of one
spatial axis — either binary (any live cell in column?) or density (how many live cells in column?).

## Module Details

### backend/engine.py

Core CA computation. Dimension-agnostic.

**`step(grid: np.ndarray, birth: set[int], survival: set[int]) -> np.ndarray`**
- Build kernel: `np.ones((3,) * grid.ndim, dtype=np.uint8)` with center index zeroed
- Neighbor count: `scipy.ndimage.convolve(grid, kernel, mode='wrap')`
- Apply rules: new cell is born if dead and neighbors in birth set, survives if alive and neighbors in survival set
- Returns new grid (do not mutate input)

**`seed_random(shape: tuple[int, ...], density: float) -> np.ndarray`**
- Returns random uint8 grid with given alive probability

**`seed_centered_blob(shape: tuple[int, ...], radius: int, density: float) -> np.ndarray`**
- Random fill only within a centered hypercube of given radius

### backend/projection.py

Stateless projection functions.

**`shadow_binary(grid: np.ndarray, axis: int) -> np.ndarray`**
- `np.any(grid > 0, axis=axis).astype(np.uint8)`

**`shadow_density(grid: np.ndarray, axis: int) -> np.ndarray`**
- `np.sum(grid, axis=axis)` — returns int array, max value = grid.shape[axis]

### backend/rules.py

**`Ruleset` dataclass**: `name: str`, `birth: frozenset[int]`, `survival: frozenset[int]`

**`parse_rule(notation: str) -> Ruleset`**
- Parses "B3/S23" or "B5/S567" notation

**`PRESETS: dict[str, Ruleset]`** — curated rules:
- `conway_2d`: B3/S23
- `3d_amoeba`: B5/S567 (good for 3D, produces organic growth)
- `3d_crystal`: B4/S34 (produces stable crystalline structures)
- `3d_coral`: B6,7,8/S5,6,7,8 (slow growth, coral-like)
- `3d_clouds`: B14-19/S13+ (from Brown University research, converges to stable 3D structures with 50% init density)

**`score_rule(ruleset, shape, steps, density) -> dict`**
- Runs simulation, returns: `pop_mean`, `pop_std`, `shadow_entropy`, `survived` (bool), `oscillation_period` (or None)

### backend/state.py

**`SimState` dataclass**:
- `grid: np.ndarray`
- `generation: int`
- `ruleset: Ruleset`
- `config: SimConfig`

**`save_state(state, path)`** / **`load_state(path) -> SimState`**
- Uses `np.savez_compressed`

### backend/config.py

**`SimConfig` dataclass**:
- `dimensions: tuple[int, ...]` — e.g. (32, 32, 32) for 3D
- `projection_axis: int` — which axis to collapse (default: last)
- `projection_mode: Literal['binary', 'density']`
- `seed_density: float` — initial alive probability (default: 0.15 for 3D)
- `seed_mode: Literal['random', 'blob']`
- `blob_radius: int`
- `sim_rate: int` — steps per second (default: 10)
- `colormap: str` — matplotlib colormap name (default: 'inferno')

### frontend/app.py

Main loop.

- Creates pygame window sized to projection output × cell_size pixels
- Accumulator-based sim timing: decouple sim rate from render rate
- State machine: RUNNING, PAUSED, STEPPING, EXPLORING
- Event loop delegates to controls.py
- Each frame: optionally step engine → project → render → HUD → flip

### frontend/renderer.py

**`render_shadow(shadow: np.ndarray, surface: pygame.Surface, config: SimConfig, lut: np.ndarray)`**
- Binary mode: shadow * 255, expand to RGB, scale up with np.kron
- Density mode: normalize to 0-255, apply LUT, scale up with np.kron
- Blit via pygame.surfarray.blit_array

### frontend/controls.py

Keyboard dispatch table. All handlers take `(state, config)` and mutate in place.

| Key | Action |
|---|---|
| Space | Toggle RUNNING ↔ PAUSED |
| → | Single step (when paused) |
| X/Y/Z | Set projection_axis to 0/1/2 |
| D | Toggle binary ↔ density |
| C | Cycle colormap |
| R | Re-seed grid |
| +/- | Adjust sim_rate |
| G | Toggle gridlines |
| S | Save snapshot to presets/ |
| E | Toggle EXPLORING mode (auto-cycle rules) |
| H | Toggle HUD |
| Q | Quit |

### frontend/hud.py

Overlay text rendered with pygame.font.SysFont.
Shows: generation, ruleset notation, population, shadow fill %, FPS, projection axis, mode.
Semi-transparent background rect behind text.

### frontend/colormap.py

**`build_lut(cmap_name: str) -> np.ndarray`**
- shape (256, 3), dtype uint8
- Uses matplotlib.cm.get_cmap to sample 256 values, convert to uint8 RGB

### main.py

Entry point. Argparse for all SimConfig fields + rule override.
Constructs config → state → app, calls app.run().

## Data Flow (one frame)

```
User input → controls.handle_events(state, config)
           → engine.step(state.grid, ruleset.birth, ruleset.survival)
           → projection.shadow_density(state.grid, config.projection_axis)
           → renderer.render_shadow(shadow, surface, config, lut)
           → hud.draw(surface, state, config)
           → pygame.display.flip()
```

## Extension: 4D → 3D

Backend: zero changes. `(N, N, N, N)` grid works with existing engine and projection.
Projection yields a 3D array. Two frontend paths:
1. Double-project: 4D→3D→2D, show 2D result in existing pygame frontend
2. Volume render: use pyvista/vispy to display the 3D shadow — separate frontend target

## Performance Notes

- 32³ = 32K cells, fine for real-time on CPU
- 64³ = 260K cells, still interactive (scipy convolve is fast)
- 128³ = 2M cells, may need to drop sim_rate
- 32⁴ = 1M cells, borderline — consider cupy as numpy drop-in if needed
