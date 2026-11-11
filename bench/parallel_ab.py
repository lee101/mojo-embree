"""Serial/parallel A/B for the in-kernel parallel paths.

Builds the parallel library and a serial-comparison library from the same
``src/embree.mojo`` (the ``SERIAL_BUILD`` / ``SERIAL_TRAVERSAL`` / worker-count
switches at the top of the file), then times both in this process.

``libmojo-embree.so`` is copied aside and restored afterwards, so a run never
leaves the serial comparison build in ``dist``.

    python bench/parallel_ab.py sweep     # ray-count sweep to find the crossover
    python bench/parallel_ab.py workers   # 1/2/4/6/8 threads at the bench size
    python bench/parallel_ab.py build     # build time vs triangle count
    python bench/parallel_ab.py identity  # serial vs parallel bit-equality

Timing follows ``bench/ab.py``: pin to a core, repeat the inner call inside one
timing window, and report the 10th percentile of the windows.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DIST = ROOT / "dist"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "bench"))

from bench import grid_mesh, rays  # noqa: E402

WORKERS = 4


def patch_source(serial_build: bool, serial_traversal: bool, workers: int) -> Path:
    text = (SRC / "embree.mojo").read_text()
    replacements = {
        "comptime SERIAL_BUILD = ": serial_build,
        "comptime SERIAL_TRAVERSAL = ": serial_traversal,
        "comptime MAX_PARALLEL_WORKERS = ": workers,
    }
    for prefix, value in replacements.items():
        lines = []
        found = False
        for line in text.splitlines():
            if line.startswith(prefix):
                line = f"{prefix}{value}"
                found = True
            lines.append(line)
        if not found:
            raise SystemExit(f"switch not found in src/embree.mojo: {prefix}")
        text = "\n".join(lines) + "\n"
    handle = tempfile.NamedTemporaryFile("w", suffix=".mojo", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def build(source: Path, output: Path) -> None:
    result = subprocess.run(
        ["pixi", "run", "mojo", "build", "--emit", "shared-lib", "-I", str(SRC),
         str(source), "-o", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode or not output.exists():
        raise SystemExit((result.stderr or result.stdout)[-4000:])


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


class Kernel:
    def __init__(self, path: Path, workers: int = WORKERS):
        self.library = ctypes.CDLL(str(path))
        self.workers = workers
        ssize = ctypes.c_ssize_t
        self.build = self.library.me_build_bvh
        self.build.argtypes = [ssize] * 11
        self.build.restype = ssize
        self.intersect = self.library.me_intersect_stream
        self.intersect.argtypes = [ssize] * 17
        self.occluded = self.library.me_occluded_stream
        self.occluded.argtypes = [ssize] * 12
        self.stack_size = int(self.library.me_stack_size())


def addr(array) -> int:
    return int(array.ctypes.data)


class Scene:
    """Owns the buffers a build or traversal call needs, like core.BVH does."""

    def __init__(self, kernel: Kernel, vertices, triangles):
        self.kernel = kernel
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles, dtype=np.int32)
        self.triangles = np.ascontiguousarray(
            vertices[triangles].reshape(-1, 9), dtype=np.float32
        )
        self.count = len(triangles)
        self.primitive_bounds = np.empty((6, self.count), dtype=np.float32)
        capacity = max(1, 2 * self.count - 1)
        self.node_bounds = np.empty((capacity, 6), dtype=np.float32)
        self.node_data = np.empty((capacity, 4), dtype=np.int32)
        self.primitive_order = np.empty(self.count, dtype=np.int32)
        self.capacity = capacity
        # + 8 slots of headroom, exactly what core.BVH allocates
        self.build_stack = np.empty(capacity + 8, dtype=np.int32)
        self.bin_bounds = np.empty((4, 32 * 3 * 6), dtype=np.float32)
        self.bin_counts = np.empty((4, 32 * 3), dtype=np.int32)
        self.right_bounds = np.empty((4, 32 * 6), dtype=np.float32)
        self.right_counts = np.empty((4, 32), dtype=np.int32)
        # One row per worker, matching the count handed to the kernel.
        self.stacks = np.empty((kernel.workers, kernel.stack_size), dtype=np.int32)
        self.node_count = 0

    def build_bvh(self) -> int:
        self.node_count = int(
            self.kernel.build(
                addr(self.triangles), addr(self.primitive_bounds), self.count,
                addr(self.node_bounds), addr(self.node_data),
                addr(self.primitive_order), addr(self.build_stack),
                addr(self.bin_bounds), addr(self.bin_counts),
                addr(self.right_bounds), addr(self.right_counts),
            )
        )
        return self.node_count

    def run_intersect(self, origins, directions):
        count = len(origins)
        hit_ids = np.empty(count, dtype=np.int32)
        hit_t = np.empty(count, dtype=np.float32)
        hit_u = np.empty(count, dtype=np.float32)
        hit_v = np.empty(count, dtype=np.float32)
        hit_ng = np.empty((count, 3), dtype=np.float32)
        near = np.zeros(count, dtype=np.float32)
        far = np.full(count, np.inf, dtype=np.float32)
        self.kernel.intersect(
            addr(self.triangles), addr(self.node_bounds), addr(self.node_data),
            addr(self.primitive_order), addr(origins), addr(directions),
            addr(near), addr(far), addr(hit_ids), addr(hit_t), addr(hit_u),
            addr(hit_v), addr(hit_ng), addr(self.stacks), self.count, 0, count,
            len(self.stacks),
        )
        return hit_ids, hit_t, hit_u, hit_v, hit_ng

    def run_occluded(self, origins, directions):
        count = len(origins)
        result = np.zeros(count, dtype=np.uint8)
        near = np.zeros(count, dtype=np.float32)
        far = np.full(count, np.inf, dtype=np.float32)
        self.kernel.occluded(
            addr(self.triangles), addr(self.node_bounds), addr(self.node_data),
            addr(self.primitive_order), addr(origins), addr(directions),
            addr(near), addr(far), addr(result), addr(self.stacks), self.count,
            0, count, len(self.stacks),
        )
        return result


def build_variants(serial_build: bool, serial_traversal: bool, workers: int):
    source = patch_source(serial_build, serial_traversal, workers)
    output = DIST / f"libparallel-ab-{source.stem}.so"
    build(source, output)
    return Kernel(output, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sweep", "workers", "build", "identity"])
    parser.add_argument("--core", type=int, default=60)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    pin(args.core)

    reference = DIST / "libmojo-embree.so"
    backup = DIST / "libmojo-embree.parallel-run.so"
    if backup.exists():
        backup.unlink()
    if reference.exists():
        shutil.copy2(reference, backup)
    try:
        main_mode(args)
    finally:
        if backup.exists():
            shutil.copy2(backup, reference)
            backup.unlink()
            print(f"restored {reference.name} from the parallel build")


def main_mode(args) -> None:
    results: dict[str, dict[str, float]] = {}
    if args.mode == "sweep":
        vertices, triangles = grid_mesh(128)
        origins, directions, _, _ = rays(40_000)
        parallel = build_variants(False, False, WORKERS)
        serial = build_variants(False, True, WORKERS)
        p_scene = Scene(parallel, vertices, triangles)
        s_scene = Scene(serial, vertices, triangles)
        p_scene.build_bvh()
        s_scene.build_bvh()
        rows = []
        for count in (1_000, 4_000, 8_192, 16_384, 32_768, 40_000):
            o, d = origins[:count], directions[:count]
            p_hit = timed(lambda: p_scene.run_intersect(o, d), 1, 15)
            s_hit = timed(lambda: s_scene.run_intersect(o, d), 1, 15)
            p_occ = timed(lambda: p_scene.run_occluded(o, d), 1, 15)
            s_occ = timed(lambda: s_scene.run_occluded(o, d), 1, 15)
            rows.append((count, s_hit, p_hit, s_occ, p_occ))
            print(
                f"{count:7d} rays  hit serial {s_hit*1e3:7.3f} ms "
                f"parallel {p_hit*1e3:7.3f} ms ({s_hit/p_hit:4.2f}x)   "
                f"occ serial {s_occ*1e3:7.3f} ms parallel {p_occ*1e3:7.3f} ms "
                f"({s_occ/p_occ:4.2f}x)",
                flush=True,
            )
        results["sweep"] = {
            str(row[0]): dict(
                hit_serial=row[1], hit_parallel=row[2],
                occ_serial=row[3], occ_parallel=row[4],
            )
            for row in rows
        }
    elif args.mode == "workers":
        vertices, triangles = grid_mesh(128)
        origins, directions, _, _ = rays(100_000)
        rows = []
        for workers in (1, 2, 4, 6, 8):
            kernel = build_variants(False, False, workers)
            scene = Scene(kernel, vertices, triangles)
            scene.build_bvh()
            hit = timed(lambda: scene.run_intersect(origins, directions), 1, 15)
            occ = timed(lambda: scene.run_occluded(origins, directions), 1, 15)
            rows.append((workers, hit, occ))
            print(
                f"{workers} workers  closest-hit {hit*1e3:7.3f} ms   "
                f"occlusion {occ*1e3:7.3f} ms",
                flush=True,
            )
        results["workers"] = {
            str(row[0]): dict(hit=row[1], occ=row[2]) for row in rows
        }
    elif args.mode == "build":
        parallel = build_variants(False, False, WORKERS)
        serial = build_variants(True, False, WORKERS)
        rows = []
        for count in (4_096, 8_192, 16_384, 32_768, 65_536):
            soup = np.random.default_rng(3).normal(
                size=(count, 3, 3)
            ).astype(np.float32)
            vertices = soup.reshape(-1, 3)
            indices = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
            p_scene = Scene(parallel, vertices, indices)
            s_scene = Scene(serial, vertices, indices)
            p_time = timed(lambda: p_scene.build_bvh(), 1, 15)
            s_time = timed(lambda: s_scene.build_bvh(), 1, 15)
            rows.append((count, s_time, p_time))
            print(
                f"{count:7d} triangles  serial {s_time*1e3:7.3f} ms "
                f"parallel {p_time*1e3:7.3f} ms ({s_time/p_time:4.2f}x)",
                flush=True,
            )
        results["build"] = {
            str(row[0]): dict(serial=row[1], parallel=row[2]) for row in rows
        }
    elif args.mode == "identity":
        vertices, triangles = grid_mesh(32)
        origins, directions, _, _ = rays(50_000)
        parallel = build_variants(False, False, WORKERS)
        serial = build_variants(True, True, WORKERS)
        p_scene = Scene(parallel, vertices, triangles)
        s_scene = Scene(serial, vertices, triangles)
        p_scene.build_bvh()
        s_scene.build_bvh()
        assert p_scene.node_count == s_scene.node_count
        live = p_scene.node_count
        # The buffers are capacity-sized; only the live node prefix is defined.
        np.testing.assert_array_equal(
            p_scene.node_bounds[:live], s_scene.node_bounds[:live]
        )
        np.testing.assert_array_equal(
            p_scene.node_data[:live], s_scene.node_data[:live]
        )
        np.testing.assert_array_equal(
            p_scene.primitive_order, s_scene.primitive_order
        )
        p_hits = p_scene.run_intersect(origins, directions)
        s_hits = s_scene.run_intersect(origins, directions)
        for a, b in zip(p_hits, s_hits):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            p_scene.run_occluded(origins, directions),
            s_scene.run_occluded(origins, directions),
        )
        print("identity: BVH arrays, closest-hit and occlusion are bit-identical")

    if args.save:
        (ROOT / "bench" / f"parallel-{args.save}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(f"saved bench/parallel-{args.save}.json")


if __name__ == "__main__":
    main()
