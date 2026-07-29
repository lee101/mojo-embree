"""Locked benchmarks against the real Embree 4 library."""

from __future__ import annotations

import ctypes
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojo_embree import BVH  # noqa: E402

I = ctypes.c_ssize_t
P = ctypes.c_void_p
REFERENCE = ctypes.CDLL(str(ROOT / "dist" / "libembree-reference.so"))
REFERENCE.reference_create.argtypes = [I, I, I, I]
REFERENCE.reference_create.restype = P
REFERENCE.reference_destroy.argtypes = [P]
REFERENCE.reference_intersect.argtypes = [P] + [I] * 10
REFERENCE.reference_occluded.argtypes = [P] + [I] * 6


def address(array: np.ndarray) -> int:
    return int(array.ctypes.data)


class Reference:
    def __init__(self, vertices, triangles):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        self.handle = REFERENCE.reference_create(
            address(self.vertices),
            len(self.vertices),
            address(self.triangles),
            len(self.triangles),
        )

    def close(self):
        REFERENCE.reference_destroy(self.handle)

    def intersect(self, origins, directions, near, far):
        count = len(origins)
        ids = np.empty(count, dtype=np.int32)
        distance = np.empty(count, dtype=np.float32)
        u = np.empty(count, dtype=np.float32)
        v = np.empty(count, dtype=np.float32)
        normal = np.empty((count, 3), dtype=np.float32)
        REFERENCE.reference_intersect(
            self.handle,
            address(origins),
            address(directions),
            address(near),
            address(far),
            address(ids),
            address(distance),
            address(u),
            address(v),
            address(normal),
            count,
        )
        return ids

    def occluded(self, origins, directions, near, far):
        result = np.empty(len(origins), dtype=np.uint8)
        REFERENCE.reference_occluded(
            self.handle,
            address(origins),
            address(directions),
            address(near),
            address(far),
            address(result),
            len(origins),
        )
        return result


def grid_mesh(side: int):
    axis = np.linspace(-1, 1, side + 1, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    z = 0.12 * np.sin(4 * x) * np.cos(3 * y)
    vertices = np.ascontiguousarray(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    row = side + 1
    lower = np.arange(side * side, dtype=np.int32)
    lower = lower + lower // side
    triangles = np.empty((2 * side * side, 3), dtype=np.int32)
    triangles[0::2] = np.column_stack((lower, lower + 1, lower + row + 1))
    triangles[1::2] = np.column_stack((lower, lower + row + 1, lower + row))
    return vertices, triangles


def rays(count: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    origins = rng.uniform(-1.2, 1.2, size=(count, 3)).astype(np.float32)
    origins[:, 2] = rng.uniform(1.0, 2.0, count)
    directions = rng.normal(0, 0.08, size=(count, 3)).astype(np.float32)
    directions[:, 2] = -1.0
    near = np.zeros(count, dtype=np.float32)
    far = np.full(count, np.inf, dtype=np.float32)
    return (
        np.ascontiguousarray(origins),
        np.ascontiguousarray(directions),
        near,
        far,
    )


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        text = Path("/proc/cpuinfo").read_text()
        return next(
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def main():
    vertices, triangles = grid_mesh(128)
    origins, directions, near, far = rays(100_000)
    ours = BVH(vertices, triangles)
    reference = Reference(vertices, triangles)
    ours.intersect(origins[:16], directions[:16])
    reference.intersect(origins[:16], directions[:16], near[:16], far[:16])

    rows = []

    def build_ours():
        BVH(vertices, triangles)

    def build_reference():
        scene = Reference(vertices, triangles)
        scene.close()

    rows.append(
        (
            "BVH build, 32,768 triangles",
            best_time(build_ours),
            best_time(build_reference),
        )
    )
    rows.append(
        (
            "closest-hit stream, 100,000 rays",
            best_time(lambda: ours.intersect(origins, directions)),
            best_time(
                lambda: reference.intersect(origins, directions, near, far)
            ),
        )
    )
    rows.append(
        (
            "occlusion stream, 100,000 rays",
            best_time(lambda: ours.occluded(origins, directions)),
            best_time(
                lambda: reference.occluded(origins, directions, near, far)
            ),
        )
    )
    reference.close()

    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs")
    print()
    print("| case | mojo-embree | Embree 4.4.1 | relative |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, embree_time in rows:
        ratio = embree_time / mojo_time
        relative = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{mojo_time / embree_time:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{embree_time * 1e3:.2f} ms | {relative} |"
        )


if __name__ == "__main__":
    main()
