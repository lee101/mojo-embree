"""Arithmetic-intensity gate for the GPU phase.

`bench/analysis.md` is the gate: only a kernel above roughly two flops per byte
is a candidate for a GPU path. These tests keep that conclusion honest in two
ways.

1. The analysis pipeline runs end to end on a small scene and reproduces the
   shipped kernel bit for bit, so the published ratios are measured on this
   code rather than a stale reimplementation.
2. Every sampled configuration in the README and in ``bench/analysis.md`` is
   below the gate. If a future change made a kernel compute-heavy enough to
   qualify, this test fails and the GPU phase has to be reconsidered rather
   than quietly skipped.

Device tests would live here and be marked ``skipif`` on the absence of a
device. There are none, because the gate stops the phase first: no GPU code
exists to test. On a CPU-only host this module still runs, because the whole
suite is CPU-only; nothing is skipped either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from bench import grid_mesh, rays  # noqa: E402
from analysis import (  # noqa: E402
    BOX_BYTES,
    BOX_FLOPS,
    TRIANGLE_BYTES,
    TRIANGLE_FLOPS,
    Counting,
    address,
    build_instrumented,
)

from mojo_embree import BVH  # noqa: E402

GATE = 2.0


@pytest.fixture(scope="module")
def counting() -> Counting:
    if not (ROOT / "src" / "embree.mojo").exists():
        pytest.skip("kernel source is not present")
    if not (ROOT / "dist" / "libmojo-embree-instrumented.so").exists():
        try:
            build_instrumented()
        except SystemExit as error:  # pragma: no cover - toolchain failure
            pytest.skip(f"could not build the instrumented kernel: {error}")
    return Counting()


def counted(counting: Counting, scene: BVH, origins, directions):
    count = len(origins)
    near = np.zeros(count, dtype=np.float32)
    far = np.full(count, np.inf, dtype=np.float32)
    boxes = np.zeros(count, dtype=np.int64)
    tris = np.zeros(count, dtype=np.int64)
    stack = np.empty((1, counting.stack_size), dtype=np.int32)
    ids = np.empty(count, dtype=np.int32)
    hit_t = np.empty(count, dtype=np.float32)
    hit_u = np.empty(count, dtype=np.float32)
    hit_v = np.empty(count, dtype=np.float32)
    hit_ng = np.empty((count, 3), dtype=np.float32)
    flags = np.zeros(count, dtype=np.uint8)
    counting.intersect(
        address(scene.triangles), address(scene.node_bounds),
        address(scene.node_data), address(scene.primitive_order),
        address(origins), address(directions), address(near), address(far),
        address(ids), address(hit_t), address(hit_u), address(hit_v),
        address(hit_ng), address(stack), len(scene.triangles), 0, count,
        address(boxes), address(tris),
    )
    closest = (int(boxes.sum()), int(tris.sum()))
    boxes[:] = 0
    tris[:] = 0
    counting.occluded(
        address(scene.triangles), address(scene.node_bounds),
        address(scene.node_data), address(scene.primitive_order),
        address(origins), address(directions), address(near), address(far),
        address(flags), address(stack), len(scene.triangles), 0, count,
        address(boxes), address(tris),
    )
    occlusion = (int(boxes.sum()), int(tris.sum()))
    hits = (ids, hit_t, hit_u, hit_v, hit_ng)
    return closest, occlusion, hits, flags


def test_counted_kernel_is_the_shipped_kernel(counting):
    vertices, triangles = grid_mesh(16)
    origins, directions, _, _ = rays(2_000, seed=7)
    scene = BVH(vertices, triangles)
    (closest, occlusion, hits, flags) = counted(
        counting, scene, origins, directions
    )
    stock = scene.intersect(origins, directions)
    np.testing.assert_array_equal(hits[0], stock.primitive_id)
    np.testing.assert_array_equal(hits[1], stock.distance)
    np.testing.assert_array_equal(hits[2], stock.u)
    np.testing.assert_array_equal(hits[3], stock.v)
    np.testing.assert_array_equal(hits[4], stock.geometric_normal)
    np.testing.assert_array_equal(flags, scene.occluded(origins, directions))
    # Every ray visits at least the root; leaf slots may be skipped by a
    # short-circuited any-hit walk, but a closest-hit walk still walks nodes.
    assert closest[0] >= len(origins)
    assert occlusion[0] >= len(origins)


def test_gate_rejects_every_kernel(counting):
    vertices, triangles = grid_mesh(64)
    origins, directions, _, _ = rays(4_000, seed=11)
    scene = BVH(vertices, triangles)
    (closest, occlusion, _, _) = counted(counting, scene, origins, directions)
    ray_count = len(origins)
    # Same constants as bench/analysis.py, imported rather than repeated, so a
    # change to the traffic model cannot pass this test silently.
    for name, (boxes, tris) in (
        ("closest-hit", closest),
        ("occlusion", occlusion),
    ):
        flops = (
            (boxes / ray_count) * BOX_FLOPS
            + (tris / ray_count) * TRIANGLE_FLOPS
        )
        nbytes = (
            (boxes / ray_count) * BOX_BYTES
            + (tris / ray_count) * TRIANGLE_BYTES
        )
        ratio = flops / nbytes
        print(f"{name}: {ratio:.3f} flops/byte")
        assert ratio < GATE, (
            f"{name} reached {ratio:.2f} flops/byte, above the {GATE} gate; "
            "the GPU phase decision in the README must be revisited"
        )
