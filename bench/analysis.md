# Kernel arithmetic intensity

Produced by `python bench/analysis.py` (which builds and drives `bench/instrument.py`). Counts are instrumented bumps at every `intersect_box` and `intersect_triangle` call in the shipped `src/embree.mojo`, run on the `grid_mesh(128)` benchmark scene (32,768 triangles, 20,000 rays).

Regenerate with `pixi run analysis`. The instrumented build is checked bit-for-bit against the stock export on every output field before its counts are used, so these numbers describe the shipped kernel rather than a reimplementation of it.

Traffic is charged one 64-byte line per random record touched: every box test fetches a node bounds row and a node data row, and every triangle test fetches an order word and a triangle row. Those arrays are separate and a ray's indices are effectively random, so no line is assumed to be reused between rays. That is the conservative reading for the gate: it maximizes traffic and minimizes the ratio. A sector-level accounting would only raise the ratios, and the closest kernel is already 4x under the gate, so the verdict holds either way.

Arithmetic: 48 flops per box test, 142 per triangle test. Branches excluded on both sides.

| kernel | box tests/ray | triangle tests/ray | flops/ray | bytes/ray | flops/byte | gate |
|---|---:|---:|---:|---:|---:|---|
| closest-hit | 18.2 | 5.1 | 1591 | 2976 | 0.535 | CPU only |
| occlusion | 24.2 | 3.1 | 1605 | 3501 | 0.459 | CPU only |

Builder traffic is charged per source access the phase makes: `precompute_bounds` streams 36 B of vertices in and 24 B of bounds out per triangle; binning reads the bounds record and the order slot, writes three bin updates, rewrites the order array during partitioning, and runs the three SAH sweeps, which is 384 B at the same 64-byte-record granularity.
Traversal counts are per ray: 1.0 box test per node visited (the counter sits directly on the `intersect_box` call), and 1.0 triangle test per leaf slot examined.

| phase | flops/primitive | bytes/primitive | flops/byte | gate |
|---|---:|---:|---:|---|
| precompute bounds | 18 | 60 | 0.300 | CPU only |
| bin + extend + SAH | 96 | 384 | 0.250 | CPU only |

Highest measured ratio across all phases: **0.53 flops/byte**. Every kernel is below the 2 flops/byte gate, so the port stays CPU only; there is no compute-heavy kernel for a GPU path to run.

Traversal is also latency-bound rather than bandwidth-bound: each ray is a short dependent walk whose next address depends on the previous test, so the CPU spends most of its time waiting on a handful of scattered misses. That is the shape a GPU is good at, but it needs the ratio above to first justify moving the data at all.
