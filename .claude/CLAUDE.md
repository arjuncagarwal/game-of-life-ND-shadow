# life-shadow

Dimension-agnostic cellular automata engine that visualizes N-dimensional Game of Life
through its (N-1)-dimensional "shadow" (projection). Primary: 3D → 2D. Extension: 4D → 3D.

## Project Structure

```
backend/    — Pure computation, no rendering. numpy/scipy only.
frontend/   — Pygame display, input handling, HUD.
presets/    — Saved .npz snapshots of interesting configurations.
docs/       — SPEC.md has full module signatures, data flow, keybindings.
main.py     — Entry point, argparse for config overrides.
```

Read `docs/SPEC.md` before implementing any module.

## Commands

- `python main.py` — Run with defaults (3D, 32³, density projection)
- `python main.py --dims 64 64 64 --rule B5/S567` — Custom grid/rule
- `python -m pytest tests/` — Run tests
- `pip install -e ".[dev]"` — Install with dev deps

## Critical Design Decisions

- Engine is dimension-agnostic. Grid is `np.ndarray` of shape `(N,)*d`. Neighbor counting
  uses `scipy.ndimage.convolve` with `np.ones((3,)*d)` kernel, center zeroed.
- Projection is stateless functions: `shadow_binary` uses `np.any`, `shadow_density` uses
  `np.sum`. Output is always ndim-1 dimensional.
- Backend never imports pygame. Frontend receives numpy arrays only.
- Grid dtype is `np.uint8`, not bool (future multi-state extension).
- Toroidal boundaries via `mode='wrap'` in convolve.
- Rules parameterized as `(birth_set, survival_set)` from B/S string notation like `B3/S23`.
- Type hints on all function signatures. Dataclasses for config/state, not dicts.
- Pure functions preferred in engine.py and projection.py. Class only for SimState.

## Verification

- Engine: Conway 2D blinker oscillates with period 2.
- Engine: known-dead config stays dead after N steps.
- Projection: single live cell at (x,y,z) → shadow pixel at (x,y) when axis=2.
- Integration: step→project pipeline runs without error for 2D, 3D, and 4D grids.

## Invariants

These override convenience, momentum, and "I'll fix it after."

1. **The specification is the source of truth.** No artifact — plan, code, config, or test —
   may introduce decisions not already in `docs/SPEC.md`. When any artifact would introduce
   new information, update the spec FIRST.

2. **Gap analysis updates the spec before the plan.** When generating or updating a plan:
   read the spec → identify gaps → add a Gaps section to the spec FIRST → then write the plan
   referencing those gaps. Never write gaps only into the plan.

3. **Completion requires surfacing human tasks.** When declaring a phase complete:
   state what was built, check the spec for external setup, list every manual step the human
   must complete, and do not say "ready to test" until all human tasks are surfaced.

## Rules

- After generating an implementation plan, list all external setup steps grouped by phase.
- After completing each phase, remind the human of manual steps before the next phase.
- After any filesystem operation producing multiple files, list results to confirm before continuing.
- When the spec is ambiguous, stop and ask — do not assume.
- Before declaring done, run `python -m pytest tests/` to verify tests pass.
