# mojo-embree

`mojo-embree` is a standalone Mojo port of the compute-heavy triangle tracing
core of [Intel Embree](https://github.com/RenderKit/embree). It builds a
reusable binned surface-area-heuristic BVH and exposes closest-hit and
any-hit ray queries through a NumPy API.

This is a derived port of Embree, whose source is licensed under the
[Apache License 2.0](https://github.com/RenderKit/embree/blob/master/LICENSE.txt).
The port's original code is MIT licensed. `NOTICE` records the upstream
copyright and attribution, and each non-obvious kernel identifies the exact
upstream file and function it follows.

## Coverage

Implemented:

- adaptive 32-bin object SAH construction for static float32 triangles;
- four-triangle SAH block costs, fallback splitting, and leaves of up to 32
  triangles;
- watertight modified Plücker edge equations with Embree's float32 epsilon
  and reciprocal thresholds;
- closest-hit scalar, 1/4/8/16-ray packet, and arbitrary stream APIs;
- early-exit occlusion queries;
- bounded parallel stream traversal with caller-owned per-worker stacks.

The repository deliberately flattens Embree's scene hierarchy into a compact
binary BVH. It does not yet include Embree's four/eight-wide node layouts,
spatial splits, quantized nodes, motion blur, filters, instances, curves,
quads, subdivision surfaces, refitting, SYCL, or GPU execution. Packet calls
currently use the same divergence-safe stream traversal kernel rather than a
separate native SIMD packet traverser.

## Install

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-embree.so`. The second shared library in
`dist` is only a benchmark adapter around the conda-forge Embree dependency;
the `mojo_embree` package does not link to it.

## Usage

```python
import numpy as np
from mojo_embree import BVH

vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    dtype=np.float32,
)
triangles = np.array([[0, 1, 2]], dtype=np.int32)

bvh = BVH(vertices, triangles)
hits = bvh.intersect(
    origins=np.array([[0.25, 0.25, 1]], dtype=np.float32),
    directions=np.array([[0, 0, -1]], dtype=np.float32),
)

assert hits.primitive_id[0] == 0
assert np.isclose(hits.distance[0], 1.0)
assert bvh.occluded([[0.25, 0.25, 1]], [[0, 0, -1]])[0]
```

`intersect_stream` is also available explicitly. `intersect_packet` validates
packet widths of 1, 4, 8, or 16. Ray intervals can be scalars or one float32
value per ray; the interval follows Embree's default triangle behavior,
`tnear < t <= tfar`.

Inputs may be strided and are copied into contiguous, call-owned float32 or
int32 buffers before entering Mojo. Integer indices are range-checked before
conversion, and values that overflow float32 are rejected. Calls are
synchronous, so Python keeps every input, output, and scratch buffer alive
until all Mojo worker tasks have returned.

## Correctness

The parity suite calls the real Embree 4.4 C API through ctypes. It compares
primitive IDs, distances, barycentric coordinates, geometric normals,
occlusion decisions, endpoint behavior, and watertight shared edges. It also
covers empty and single-triangle meshes, duplicate vertices, zero-area faces,
non-manifold edges, unreferenced vertices, randomized streams, all packet
widths, and BVH structural invariants.

```bash
pixi run test
```

There is no maintained Python binding needed for the reference: tests bind
the installed upstream shared library directly. This is stronger than a
NumPy surrogate and avoids comparing one port against another.

## How it works

Python gathers an indexed mesh once into a contiguous `(triangle_count, 9)`
float32 triangle buffer. The Mojo builder partitions a flat primitive-index
array in place and fills flat node-bound and node-metadata arrays. Triangle
bounds are cached once in a structure-of-arrays buffer; large independent
subtrees build with four worker tasks above a measured size threshold, while
smaller meshes remain serial. The hot path uses no pointer graphs and performs
no per-ray allocation. Each of at most 32 traversal tasks owns a fixed
128-entry stack supplied by NumPy.

All arrays remain Python-owned. The ctypes layer sends only integer addresses
and validated element counts across the C ABI, reconstructing `UnsafePointer`
values inside non-parametric exported Mojo functions. Results are written
directly into contiguous NumPy output arrays. Traversal uses fixed per-worker
scratch stacks; a depth guard switches pathological trees to a safe linear
scan instead of overflowing that storage.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 system with 72
logical CPUs. Times are the best of three warm runs.
The reference is conda-forge Embree 4.4.1 called from a compiled C++ loop, so
neither side pays a per-ray Python call.

| case | mojo-embree | Embree 4.4.1 | relative |
|---|---:|---:|---:|
| BVH build, 32,768 triangles | 20.67 ms | 5.57 ms | 3.71x slower |
| closest-hit stream, 100,000 rays | 5.88 ms | 17.84 ms | 3.03x faster |
| occlusion stream, 100,000 rays | 5.39 ms | 14.30 ms | 2.66x faster |

Embree's highly optimized parallel wide-node builder remains substantially
faster than this compact binary builder. The Mojo builder clears only active
adaptive SAH bins and distributes independent frontier subtrees over four
workers above 8,192 triangles. The stream kernels distribute large ray batches
over up to 32 physical-core tasks. This is not a claim that one scalar
traversal is faster than Embree.

No GPU path is included. The builder is dominated by bounds traffic and random
bin updates, while traversal has divergent control flow and irregular node and
triangle reads; both remain below roughly two arithmetic operations per byte
moved. CPU traversal is also already ahead of the reference in this benchmark,
so a GPU path would add transfer and launch overhead without a justified target.
