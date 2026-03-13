"""Verification tests for engine.py, rules.py, and projection.py."""

import numpy as np
import pytest

from backend.engine import step, seed_random, seed_centered_blob
from backend.projection import shadow_binary, shadow_density
from backend.rules import parse_rule, PRESETS, Ruleset


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

class TestParseRule:
    def test_conway(self):
        r = parse_rule('B3/S23')
        assert r.birth == frozenset({3})
        assert r.survival == frozenset({2, 3})

    def test_3d_amoeba(self):
        r = parse_rule('B5/S567')
        assert r.birth == frozenset({5})
        assert r.survival == frozenset({5, 6, 7})

    def test_range_notation(self):
        r = parse_rule('B14-16/S13-15')
        assert r.birth == frozenset({14, 15, 16})
        assert r.survival == frozenset({13, 14, 15})

    def test_comma_notation(self):
        r = parse_rule('B6,7,8/S5,6,7,8')
        assert r.birth == frozenset({6, 7, 8})
        assert r.survival == frozenset({5, 6, 7, 8})

    def test_empty_parts(self):
        r = parse_rule('B/S')
        assert r.birth == frozenset()
        assert r.survival == frozenset()

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_rule('invalid')

    def test_presets_loaded(self):
        assert 'conway_2d' in PRESETS
        assert '3d_amoeba' in PRESETS
        assert '3d_crystal' in PRESETS
        assert '3d_coral' in PRESETS
        assert '3d_clouds' in PRESETS


# ---------------------------------------------------------------------------
# Engine: known-dead config stays dead
# ---------------------------------------------------------------------------

class TestDeadGrid:
    def test_all_dead_stays_dead_2d(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        r = PRESETS['conway_2d']
        for _ in range(5):
            grid = step(grid, r.birth, r.survival)
        assert grid.sum() == 0

    def test_all_dead_stays_dead_3d(self):
        grid = np.zeros((8, 8, 8), dtype=np.uint8)
        r = PRESETS['3d_amoeba']
        for _ in range(3):
            grid = step(grid, r.birth, r.survival)
        assert grid.sum() == 0


# ---------------------------------------------------------------------------
# Engine: Conway 2D blinker oscillates with period 2
# ---------------------------------------------------------------------------

class TestBlinker:
    def _make_blinker(self) -> np.ndarray:
        """Horizontal blinker centred in a 5x5 grid."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1] = 1
        grid[2, 2] = 1
        grid[2, 3] = 1
        return grid

    def test_blinker_period_2(self):
        r = PRESETS['conway_2d']
        g0 = self._make_blinker()
        g1 = step(g0, r.birth, r.survival)
        g2 = step(g1, r.birth, r.survival)
        assert np.array_equal(g0, g2), "Blinker must return to initial state after 2 steps"

    def test_blinker_changes_after_step(self):
        r = PRESETS['conway_2d']
        g0 = self._make_blinker()
        g1 = step(g0, r.birth, r.survival)
        assert not np.array_equal(g0, g1), "Blinker must change after one step"

    def test_blinker_vertical_after_step(self):
        """After one step the horizontal blinker becomes vertical."""
        r = PRESETS['conway_2d']
        g0 = self._make_blinker()
        g1 = step(g0, r.birth, r.survival)
        assert g1[1, 2] == 1
        assert g1[2, 2] == 1
        assert g1[3, 2] == 1
        assert g1[2, 1] == 0
        assert g1[2, 3] == 0


# ---------------------------------------------------------------------------
# Engine: input grid is not mutated
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_step_does_not_mutate_input(self):
        r = PRESETS['conway_2d']
        grid = np.ones((6, 6), dtype=np.uint8)
        original = grid.copy()
        step(grid, r.birth, r.survival)
        assert np.array_equal(grid, original)


# ---------------------------------------------------------------------------
# Engine: output dtype
# ---------------------------------------------------------------------------

class TestDtype:
    def test_output_is_uint8(self):
        r = PRESETS['conway_2d']
        grid = seed_random((8, 8), density=0.3)
        out = step(grid, r.birth, r.survival)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Projection: single live cell
# ---------------------------------------------------------------------------

class TestProjection:
    def test_shadow_binary_single_cell(self):
        """Live cell at (x,y,z) → shadow pixel at (x,y) when axis=2."""
        grid = np.zeros((4, 4, 4), dtype=np.uint8)
        grid[1, 2, 3] = 1
        shadow = shadow_binary(grid, axis=2)
        assert shadow.shape == (4, 4)
        assert shadow[1, 2] == 1
        # All other pixels must be 0
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1, 2] = 1
        assert np.array_equal(shadow, expected)

    def test_shadow_density_single_cell(self):
        grid = np.zeros((4, 4, 4), dtype=np.uint8)
        grid[1, 2, 3] = 1
        shadow = shadow_density(grid, axis=2)
        assert shadow.shape == (4, 4)
        assert shadow[1, 2] == 1
        assert shadow.sum() == 1

    def test_shadow_density_multiple_cells_same_column(self):
        grid = np.zeros((4, 4, 4), dtype=np.uint8)
        grid[1, 2, 0] = 1
        grid[1, 2, 2] = 1
        shadow = shadow_density(grid, axis=2)
        assert shadow[1, 2] == 2

    def test_shadow_binary_all_dead(self):
        grid = np.zeros((4, 4, 4), dtype=np.uint8)
        shadow = shadow_binary(grid, axis=2)
        assert shadow.sum() == 0

    def test_shadow_binary_axis0(self):
        grid = np.zeros((4, 4, 4), dtype=np.uint8)
        grid[2, 1, 3] = 1
        shadow = shadow_binary(grid, axis=0)
        assert shadow.shape == (4, 4)
        assert shadow[1, 3] == 1


# ---------------------------------------------------------------------------
# Integration: step→project pipeline for 2D, 3D, 4D
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_pipeline_2d(self):
        r = PRESETS['conway_2d']
        grid = seed_random((16, 16), density=0.3)
        grid = step(grid, r.birth, r.survival)
        shadow = shadow_density(grid, axis=1)
        assert shadow.shape == (16,)

    def test_pipeline_3d(self):
        r = PRESETS['3d_amoeba']
        grid = seed_random((16, 16, 16), density=0.15)
        grid = step(grid, r.birth, r.survival)
        shadow = shadow_density(grid, axis=2)
        assert shadow.shape == (16, 16)

    def test_pipeline_4d(self):
        r = PRESETS['3d_amoeba']
        grid = seed_random((8, 8, 8, 8), density=0.1)
        grid = step(grid, r.birth, r.survival)
        shadow = shadow_density(grid, axis=3)
        assert shadow.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------

class TestSeeds:
    def test_seed_random_density(self):
        np.random.seed(42)
        grid = seed_random((100, 100), density=0.5)
        assert grid.dtype == np.uint8
        # Should be roughly 50% alive (within 5%)
        assert abs(grid.mean() - 0.5) < 0.05

    def test_seed_random_zeros(self):
        grid = seed_random((10, 10), density=0.0)
        assert grid.sum() == 0

    def test_seed_random_ones(self):
        grid = seed_random((10, 10), density=1.0)
        assert grid.sum() == 100

    def test_seed_centered_blob_shape(self):
        grid = seed_centered_blob((20, 20, 20), radius=3, density=0.5)
        assert grid.shape == (20, 20, 20)
        assert grid.dtype == np.uint8

    def test_seed_centered_blob_bounded(self):
        """Blob cells must lie within the central radius."""
        grid = seed_centered_blob((20, 20, 20), radius=3, density=1.0)
        # Outside central band must be zero
        assert grid[:7, :, :].sum() == 0
        assert grid[14:, :, :].sum() == 0
