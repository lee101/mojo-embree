"""Instrumented copy of the real traversal kernel, for arithmetic-intensity measurement.

The hand-made replay in ``bench/analysis.py`` had already drifted from
``trace_one`` (wrong slab-test clamp, wrong live-node prefix), so its counts
could not be trusted. This module instead rewrites ``src/embree.mojo`` itself,
inserting counter bumps at the exact lines, and builds that. What is counted
is the shipped kernel.

Counted per ray: one ``box`` per ``intersect_box`` evaluation (one node
touched) and one ``tri`` per ``intersect_triangle`` evaluation (one triangle
touched).

The instrumented build forces the serial range path (``RAY_PARALLEL_MIN`` above
any input) so counters bump through a single, non-racy code path, and adds two
exports named ``me_count_*``. The stock exports keep their signatures; only
the rewritten internal functions gain counter parameters.

    python bench/instrument.py     # build dist/libmojo-embree-instrumented.so
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "dist" / "libmojo-embree-instrumented.so"

# Each shot: (old, new, expected occurrences). Applied in order.
TRANSFORMS: list[tuple[str, str, int]] = [
    # Serial only: every input goes through the range functions.
    (
        "comptime RAY_PARALLEL_MIN = 32_768\n",
        "comptime RAY_PARALLEL_MIN = 1 << 62\n",
        1,
    ),
    (
        "comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]\n",
        "comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]\n"
        "comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]\n",
        1,
    ),
    # trace_one / occluded_one signatures gain the counters.
    (
        "    primitive_count: Int,\n    ray_index: Int,\n):",
        "    primitive_count: Int,\n    ray_index: Int,\n"
        "    box_counts: I64Ptr,\n    tri_counts: I64Ptr,\n):",
        2,
    ),
    # trace_range / occluded_range signatures gain the counters. Both are
    # matched at once; they share the same five-line signature head.
    (
        "    ray_start: Int,\n    ray_stop: Int,\n    stack_row: Int,\n):\n"
        "    var stack = traversal_stacks",
        "    ray_start: Int,\n    ray_stop: Int,\n    stack_row: Int,\n"
        "    box_counts: I64Ptr,\n    tri_counts: I64Ptr,\n):\n"
        "    var stack = traversal_stacks",
        2,
    ),
    # trace_range -> trace_one.
    (
        "            primitive_count,\n            ray,\n        )\n\n\n"
        "# Embree: kernels/bvh/bvh_intersector1.cpp BVHNIntersector1::occluded.",
        "            primitive_count,\n            ray,\n"
        "            box_counts,\n            tri_counts,\n        )\n\n\n"
        "# Embree: kernels/bvh/bvh_intersector1.cpp BVHNIntersector1::occluded.",
        1,
    ),
    # occluded_range -> occluded_one.
    (
        "            primitive_count,\n            ray,\n        )\n\n\n"
        "@export(\"me_build_bvh\")",
        "            primitive_count,\n            ray,\n"
        "            box_counts,\n            tri_counts,\n        )\n\n\n"
        "@export(\"me_build_bvh\")",
        1,
    ),
    # trace_lane / occlusion_lane forward to the range functions too; they
    # are dead under RAY_PARALLEL_MIN = 1 << 62, so give them the 0 default.
    (
        "        primitive_count,\n        start,\n        stop,\n        row,\n    )",
        "        primitive_count,\n        start,\n        stop,\n        row,\n"
        "        I64Ptr(unsafe_from_address=8),\n"
        "        I64Ptr(unsafe_from_address=8),\n    )",
        2,
    ),
    # Both serial call sites inside the stock exports, each followed by the
    # parallel branch it falls back from.
    (
        "            primitive_count,\n            ray_start,\n            ray_stop,\n"
        "            0,\n        )\n    else:\n        parallel_trace(",
        "            primitive_count,\n            ray_start,\n            ray_stop,\n"
        "            0,\n            box_counts,\n            tri_counts,\n        )\n"
        "    else:\n        parallel_trace(",
        1,
    ),
    (
        "            primitive_count,\n            ray_start,\n            ray_stop,\n"
        "            0,\n        )\n    else:\n        parallel_occluded(",
        "            primitive_count,\n            ray_start,\n            ray_stop,\n"
        "            0,\n            box_counts,\n            tri_counts,\n        )\n"
        "    else:\n        parallel_occluded(",
        1,
    ),
    # One box count per visited node, in both kernels.
    (
        "        var node = Int(traversal_stack[stack_size])\n"
        "        if intersect_box(",
        "        var node = Int(traversal_stack[stack_size])\n"
        "        box_counts[ray_index] += 1\n"
        "        if intersect_box(",
        2,
    ),
    # One triangle count per leaf slot tested, in both kernels.
    (
        "                var primitive = Int(primitive_order[slot])\n"
        "                var distance = intersect_triangle(",
        "                var primitive = Int(primitive_order[slot])\n"
        "                tri_counts[ray_index] += 1\n"
        "                var distance = intersect_triangle(",
        2,
    ),
    # The stock exports now pass counters into the range calls, so they need
    # the two addresses as trailing (defaulted) parameters.
    (
        "    ray_stop: Int,\n    traversal_stack_count: Int = 1,\n"
        ") abi(\"C\"):\n    initialize_runtime()\n    if ray_stop <= ray_start:\n"
        "        return\n    var triangles = fp(triangles_address)\n"
        "    var node_bounds = fp(node_bounds_address)",
        "    ray_stop: Int,\n    traversal_stack_count: Int = 1,\n"
        "    box_counts_address: Int = 0,\n    tri_counts_address: Int = 0,\n"
        ") abi(\"C\"):\n    initialize_runtime()\n    if ray_stop <= ray_start:\n"
        "        return\n"
        "    var box_counts = I64Ptr(unsafe_from_address=box_counts_address)\n"
        "    var tri_counts = I64Ptr(unsafe_from_address=tri_counts_address)\n"
        "    var triangles = fp(triangles_address)\n"
        "    var node_bounds = fp(node_bounds_address)",
        2,
    ),
]


COUNT_EXPORTS = '''comptime I64Ptr0 = I64Ptr


@export("me_count_intersect")
def me_count_intersect(
    triangles_address: Int,
    node_bounds_address: Int,
    node_data_address: Int,
    primitive_order_address: Int,
    origins_address: Int,
    directions_address: Int,
    ray_nears_address: Int,
    ray_fars_address: Int,
    hit_ids_address: Int,
    hit_t_address: Int,
    hit_u_address: Int,
    hit_v_address: Int,
    hit_ng_address: Int,
    traversal_stacks_address: Int,
    primitive_count: Int,
    ray_start: Int,
    ray_stop: Int,
    box_counts_address: Int,
    tri_counts_address: Int,
) abi("C"):
    trace_range(
        fp(triangles_address),
        fp(node_bounds_address),
        ip(node_data_address),
        ip(primitive_order_address),
        ip(traversal_stacks_address),
        fp(origins_address),
        fp(directions_address),
        fp(ray_nears_address),
        fp(ray_fars_address),
        ip(hit_ids_address),
        fp(hit_t_address),
        fp(hit_u_address),
        fp(hit_v_address),
        fp(hit_ng_address),
        primitive_count,
        ray_start,
        ray_stop,
        0,
        I64Ptr(unsafe_from_address=box_counts_address),
        I64Ptr(unsafe_from_address=tri_counts_address),
    )


@export("me_count_occluded")
def me_count_occluded(
    triangles_address: Int,
    node_bounds_address: Int,
    node_data_address: Int,
    primitive_order_address: Int,
    origins_address: Int,
    directions_address: Int,
    ray_nears_address: Int,
    ray_fars_address: Int,
    occluded_address: Int,
    traversal_stacks_address: Int,
    primitive_count: Int,
    ray_start: Int,
    ray_stop: Int,
    box_counts_address: Int,
    tri_counts_address: Int,
) abi("C"):
    occluded_range(
        fp(triangles_address),
        fp(node_bounds_address),
        ip(node_data_address),
        ip(primitive_order_address),
        ip(traversal_stacks_address),
        fp(origins_address),
        fp(directions_address),
        fp(ray_nears_address),
        fp(ray_fars_address),
        bp(occluded_address),
        primitive_count,
        ray_start,
        ray_stop,
        0,
        I64Ptr(unsafe_from_address=box_counts_address),
        I64Ptr(unsafe_from_address=tri_counts_address),
    )


@export("me_stack_size")
'''


def patched_source() -> str:
    text = (SRC / "embree.mojo").read_text()
    for old, new, expected in TRANSFORMS:
        found = text.count(old)
        if found != expected:
            raise SystemExit(
                f"anchor found {found}x, expected {expected}x:\n{old}"
            )
        text = text.replace(old, new)
    anchor = '@export("me_stack_size")\n'
    if text.count(anchor) != 1:
        raise SystemExit("me_stack_size anchor is not unique")
    return text.replace(anchor, COUNT_EXPORTS)


def build() -> Path:
    source = ROOT / "build" / "instrumented_embree.mojo"
    source.parent.mkdir(exist_ok=True)
    source.write_text(patched_source())
    result = subprocess.run(
        ["pixi", "run", "mojo", "build", "--emit", "shared-lib", "-I", str(SRC),
         str(source), "-o", str(OUTPUT)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        errors = [
            line for line in (result.stderr or result.stdout).splitlines()
            if "error" in line
        ]
        raise SystemExit("\n".join(errors) or (result.stderr or result.stdout)[-4000:])
    return OUTPUT


if __name__ == "__main__":
    print(build())
