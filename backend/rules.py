"""Rule parsing and preset rulesets for N-dimensional Game of Life."""

from dataclasses import dataclass
import re

import numpy as np

from backend.engine import step
from backend.projection import shadow_binary


@dataclass(frozen=True)
class Ruleset:
    name: str
    birth: frozenset[int]
    survival: frozenset[int]

    def notation(self) -> str:
        b = ''.join(str(n) for n in sorted(self.birth))
        s = ''.join(str(n) for n in sorted(self.survival))
        return f"B{b}/S{s}"


def parse_rule(notation: str) -> Ruleset:
    """Parse 'B3/S23' or 'B5/S567' notation into a Ruleset."""
    notation = notation.strip()
    match = re.fullmatch(r'B([0-9,\-]*)/S([0-9,\-]*)', notation)
    if not match:
        raise ValueError(f"Invalid rule notation: {notation!r}")

    def parse_part(part: str) -> frozenset[int]:
        if not part:
            return frozenset()
        nums: set[int] = set()
        if ',' in part:
            # Comma-separated tokens (can be multi-digit or ranges)
            for token in part.split(','):
                token = token.strip()
                if '-' in token:
                    lo, hi = token.split('-', 1)
                    nums.update(range(int(lo), int(hi) + 1))
                else:
                    nums.add(int(token))
        elif '-' in part:
            # Single range like "14-19"
            lo, hi = part.split('-', 1)
            nums.update(range(int(lo), int(hi) + 1))
        else:
            # Plain digit string: "23" → {2, 3}, "567" → {5, 6, 7}
            for ch in part:
                nums.add(int(ch))
        return frozenset(nums)

    birth = parse_part(match.group(1))
    survival = parse_part(match.group(2))
    return Ruleset(name=notation, birth=birth, survival=survival)


PRESETS: dict[str, Ruleset] = {
    'conway_2d': parse_rule('B3/S23'),
    '3d_amoeba': parse_rule('B5/S567'),
    '3d_crystal': parse_rule('B4/S34'),
    '3d_coral': parse_rule('B6,7,8/S5,6,7,8'),
    '3d_clouds': parse_rule('B14-19/S13-26'),
}
# Give presets readable names
PRESETS['conway_2d'] = Ruleset('conway_2d', PRESETS['conway_2d'].birth, PRESETS['conway_2d'].survival)
PRESETS['3d_amoeba'] = Ruleset('3d_amoeba', PRESETS['3d_amoeba'].birth, PRESETS['3d_amoeba'].survival)
PRESETS['3d_crystal'] = Ruleset('3d_crystal', PRESETS['3d_crystal'].birth, PRESETS['3d_crystal'].survival)
PRESETS['3d_coral'] = Ruleset('3d_coral', PRESETS['3d_coral'].birth, PRESETS['3d_coral'].survival)
PRESETS['3d_clouds'] = Ruleset('3d_clouds', PRESETS['3d_clouds'].birth, PRESETS['3d_clouds'].survival)


def score_rule(
    ruleset: Ruleset,
    shape: tuple[int, ...],
    steps: int,
    density: float,
) -> dict:
    """Run simulation and return quality metrics.

    Returns:
        pop_mean: mean population across steps
        pop_std: std of population across steps
        shadow_entropy: entropy of final shadow (bits)
        survived: True if population is nonzero at end
        oscillation_period: detected period or None
    """
    from backend.engine import seed_random

    grid = seed_random(shape, density)
    pops: list[int] = []
    grids: list[np.ndarray] = []

    for _ in range(steps):
        grid = step(grid, ruleset.birth, ruleset.survival)
        pops.append(int(grid.sum()))
        grids.append(grid.copy())

    pop_arr = np.array(pops, dtype=float)
    pop_mean = float(pop_arr.mean())
    pop_std = float(pop_arr.std())
    survived = pops[-1] > 0

    # Shadow entropy
    shadow = shadow_binary(grid, axis=-1).astype(float)
    p = shadow.mean()
    if 0 < p < 1:
        shadow_entropy = float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    else:
        shadow_entropy = 0.0

    # Oscillation: check if any recent grid matches an earlier one
    oscillation_period = None
    for period in range(1, min(20, steps // 2 + 1)):
        if np.array_equal(grids[-1], grids[-1 - period]):
            oscillation_period = period
            break

    return {
        'pop_mean': pop_mean,
        'pop_std': pop_std,
        'shadow_entropy': shadow_entropy,
        'survived': survived,
        'oscillation_period': oscillation_period,
    }
