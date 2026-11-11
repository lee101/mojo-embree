"""Low-noise A/B timer used while tuning kernels.

The main benchmark runs each case once per call and takes the best of three,
which is fine for a quiet machine but too noisy to accept or reject a 5%
kernel change. This harness pins one core, repeats the inner call many times
inside one timing window, and reports the 10th percentile of those windows, so
a contended neighbour core cannot turn one bad window into the result.

    python bench/ab.py            # print numbers
    python bench/ab.py --save old # store them in bench/ab-old.json

Only for development; the README table still comes from ``pixi run bench``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from bench import Reference, grid_mesh, rays  # noqa: E402

from mojo_embree import BVH  # noqa: E402


def pin(core: int) -> None:
    try:
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        pass


def timed(call, repeat: int, windows: int) -> float:
    samples = []
    for _ in range(windows):
        start = time.perf_counter()
        for _ in range(repeat):
            call()
        samples.append((time.perf_counter() - start) / repeat)
    samples.sort()
    return samples[len(samples) // 10]


def serial(fn, workers: int = 1):
    """Dispatch ``fn`` on the calling thread only, bypassing the thread pool."""
    import mojo_embree.core as core

    original_workers, original_min = core.MAX_RAY_WORKERS, core.RAY_PARALLEL_MIN
    core.MAX_RAY_WORKERS, core.RAY_PARALLEL_MIN = workers, 10**9
    try:
        return fn()
    finally:
        core.MAX_RAY_WORKERS, core.RAY_PARALLEL_MIN = original_workers, original_min


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None)
    parser.add_argument("--core", type=int, default=60)
    args = parser.parse_args()
    pin(args.core)

    vertices, triangles = grid_mesh(128)
    origins, directions, _, _ = rays(100_000)
    bvh = BVH(vertices, triangles)
    reference = Reference(vertices, triangles)
    bvh.intersect(origins[:16], directions[:16])
    reference.intersect(
        origins[:16],
        directions[:16],
        np.zeros(16, np.float32),
        np.full(16, np.inf, np.float32),
    )

    result = {}
    result["build"] = timed(lambda: BVH(vertices, triangles), 3, 30)
    result["closest_hit"] = timed(lambda: bvh.intersect(origins, directions), 1, 15)
    result["occlusion"] = timed(lambda: bvh.occluded(origins, directions), 1, 15)
    result["closest_hit_serial"] = timed(
        lambda: serial(lambda: bvh.intersect(origins, directions), 1), 1, 9
    )
    result["occlusion_serial"] = timed(
        lambda: serial(lambda: bvh.occluded(origins, directions), 1), 1, 9
    )

    small = slice(0, 1_000)
    result["closest_hit_1k"] = timed(
        lambda: serial(lambda: bvh.intersect(origins[small], directions[small]), 1),
        200,
        30,
    )
    result["occlusion_1k"] = timed(
        lambda: serial(lambda: bvh.occluded(origins[small], directions[small]), 1),
        200,
        30,
    )
    reference.close()

    for name, seconds in result.items():
        print(f"{name:24s} {seconds * 1e3:9.3f} ms")
    if args.save:
        Path(__file__).with_name(f"ab-{args.save}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(f"saved bench/ab-{args.save}.json")


if __name__ == "__main__":
    main()
