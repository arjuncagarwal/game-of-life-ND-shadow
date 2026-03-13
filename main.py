"""Entry point for life-shadow."""

import argparse

from backend.config import SimConfig
from backend.engine import seed_random, seed_centered_blob
from backend.rules import parse_rule
from backend.state import SimState
from frontend.app import App


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N-dimensional Game of Life shadow projection")
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 32, 32],
                        help="Grid dimensions, e.g. --dims 32 32 32 for 3D")
    parser.add_argument("--rule", type=str, default="B5/S567",
                        help="Birth/survival rule in B/S notation")
    parser.add_argument("--density", type=float, default=0.15,
                        help="Initial seed density")
    parser.add_argument("--seed-mode", choices=["random", "blob"], default="blob",
                        help="Seeding strategy")
    parser.add_argument("--blob-radius", type=int, default=8,
                        help="Blob radius for blob seed mode")
    parser.add_argument("--projection-axis", type=int, default=-1,
                        help="Axis to project along (-1 for last)")
    parser.add_argument("--projection-mode", choices=["binary", "density"], default="density",
                        help="Shadow projection mode")
    parser.add_argument("--sim-rate", type=int, default=10,
                        help="Simulation steps per second")
    parser.add_argument("--colormap", type=str, default="inferno",
                        help="Matplotlib colormap name for density mode")
    parser.add_argument("--cell-size", type=int, default=8,
                        help="Pixels per cell in display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SimConfig(
        dimensions=tuple(args.dims),
        projection_axis=args.projection_axis,
        projection_mode=args.projection_mode,
        seed_density=args.density,
        seed_mode=args.seed_mode,
        blob_radius=args.blob_radius,
        sim_rate=args.sim_rate,
        colormap=args.colormap,
        cell_size=args.cell_size,
    )

    ruleset = parse_rule(args.rule)

    if config.seed_mode == 'blob':
        grid = seed_centered_blob(config.dimensions, config.blob_radius, config.seed_density)
    else:
        grid = seed_random(config.dimensions, config.seed_density)

    state = SimState(grid=grid, generation=0, ruleset=ruleset, config=config)
    App(state).run()


if __name__ == "__main__":
    main()
