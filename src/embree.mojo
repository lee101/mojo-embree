"""Embree-derived float32 BVH construction and ray traversal kernels."""

from max.algorithm import parallelize
from std.math import abs, iota
from std.runtime import initialize_runtime
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime MAX_BINS = 32
comptime MAX_LEAF_SIZE = 32
comptime STACK_SIZE = 128
comptime MAX_WORKERS = 32
comptime PARALLEL_THRESHOLD = 4_096
comptime BUILD_PREPROCESS_PARALLEL_THRESHOLD = 1_000_000
comptime BUILD_PREPROCESS_MAX_WORKERS = 8
comptime BUILD_PARALLEL_THRESHOLD = 8_192
comptime BUILD_MAX_WORKERS = 4
comptime LARGE = Float32(3.402823466e38)
comptime ULP = Float32(1.1920928955078125e-7)
comptime MIN_RCP_INPUT = Float32(1.0e-18)
comptime CENTROID_EPS = Float32(1.0e-34)


@always_inline
def fp(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


@always_inline
def ip(address: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=address)


@always_inline
def bp(address: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=address)


# Embree: common/math/vec3.h halfArea.
@always_inline
def half_area(
    lx: Float32,
    ly: Float32,
    lz: Float32,
    ux: Float32,
    uy: Float32,
    uz: Float32,
) -> Float32:
    var dx = ux - lx
    var dy = uy - ly
    var dz = uz - lz
    return dx * (dy + dz) + dy * dz


# Embree: kernels/builders/heuristic_binning.h BinMapping::bin.
@always_inline
def bin_index(center: Float32, offset: Float32, scale: Float32, count: Int) -> Int:
    var index = Int((center - offset) * scale)
    return max(0, min(count - 1, index))


@always_inline
def extend_bin(
    bin_bounds: F32Ptr,
    bin_counts: I32Ptr,
    bin: Int,
    dim: Int,
    lx: Float32,
    ly: Float32,
    lz: Float32,
    ux: Float32,
    uy: Float32,
    uz: Float32,
):
    var index = bin * 3 + dim
    bin_counts[index] += 1
    var bb = index * 6
    bin_bounds[bb] = min(bin_bounds[bb], lx)
    bin_bounds[bb + 1] = min(bin_bounds[bb + 1], ly)
    bin_bounds[bb + 2] = min(bin_bounds[bb + 2], lz)
    bin_bounds[bb + 3] = max(bin_bounds[bb + 3], ux)
    bin_bounds[bb + 4] = max(bin_bounds[bb + 4], uy)
    bin_bounds[bb + 5] = max(bin_bounds[bb + 5], uz)


@always_inline
def precompute_bounds_chunk(
    triangles: F32Ptr,
    primitive_bounds: F32Ptr,
    primitive_count: Int,
    begin: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var primitive = begin
    while primitive + W <= end:
        var base = primitive * 9
        var ax = (triangles + base).strided_load[width=W](9)
        var ay = (triangles + base + 1).strided_load[width=W](9)
        var az = (triangles + base + 2).strided_load[width=W](9)
        var bx = (triangles + base + 3).strided_load[width=W](9)
        var by = (triangles + base + 4).strided_load[width=W](9)
        var bz = (triangles + base + 5).strided_load[width=W](9)
        var cx = (triangles + base + 6).strided_load[width=W](9)
        var cy = (triangles + base + 7).strided_load[width=W](9)
        var cz = (triangles + base + 8).strided_load[width=W](9)
        primitive_bounds.store(primitive, min(ax, min(bx, cx)))
        primitive_bounds.store(
            primitive_count + primitive, min(ay, min(by, cy))
        )
        primitive_bounds.store(
            2 * primitive_count + primitive, min(az, min(bz, cz))
        )
        primitive_bounds.store(
            3 * primitive_count + primitive, max(ax, max(bx, cx))
        )
        primitive_bounds.store(
            4 * primitive_count + primitive, max(ay, max(by, cy))
        )
        primitive_bounds.store(
            5 * primitive_count + primitive, max(az, max(bz, cz))
        )
        primitive += W

    while primitive < end:
        var base = primitive * 9
        primitive_bounds[primitive] = min(
            triangles[base], min(triangles[base + 3], triangles[base + 6])
        )
        primitive_bounds[primitive_count + primitive] = min(
            triangles[base + 1],
            min(triangles[base + 4], triangles[base + 7]),
        )
        primitive_bounds[2 * primitive_count + primitive] = min(
            triangles[base + 2],
            min(triangles[base + 5], triangles[base + 8]),
        )
        primitive_bounds[3 * primitive_count + primitive] = max(
            triangles[base], max(triangles[base + 3], triangles[base + 6])
        )
        primitive_bounds[4 * primitive_count + primitive] = max(
            triangles[base + 1],
            max(triangles[base + 4], triangles[base + 7]),
        )
        primitive_bounds[5 * primitive_count + primitive] = max(
            triangles[base + 2],
            max(triangles[base + 5], triangles[base + 8]),
        )
        primitive += 1


def precompute_bounds(
    triangles: F32Ptr,
    primitive_bounds: F32Ptr,
    primitive_count: Int,
):
    var tasks = 1
    if primitive_count >= BUILD_PREPROCESS_PARALLEL_THRESHOLD:
        tasks = min(
            BUILD_PREPROCESS_MAX_WORKERS,
            min(num_physical_cores(), max(1, primitive_count // 4_096)),
        )
    if tasks > 1:

        @parameter
        @__copy_capture(triangles, primitive_bounds, primitive_count, tasks)
        @always_inline
        def process_chunk(task: Int):
            var begin = primitive_count * task // tasks
            var end = primitive_count * (task + 1) // tasks
            precompute_bounds_chunk(
                triangles, primitive_bounds, primitive_count, begin, end
            )

        parallelize[process_chunk](tasks, tasks)
    else:
        precompute_bounds_chunk(
            triangles, primitive_bounds, primitive_count, 0, primitive_count
        )


@always_inline
def build_subtree(
    primitive_bounds: F32Ptr,
    primitive_count: Int,
    node_bounds: F32Ptr,
    node_data: I32Ptr,
    primitive_order: I32Ptr,
    build_stack: I32Ptr,
    bin_bounds: F32Ptr,
    bin_counts: I32Ptr,
    right_bounds: F32Ptr,
    right_counts: I32Ptr,
    mut nodes: Int,
    root_node: Int,
    only_root: Bool,
) -> Bool:
    comptime W = simdwidthof[DType.float64]()
    var stack_size = 1
    build_stack[0] = Int32(root_node)

    while stack_size > 0:
        stack_size -= 1
        var node = Int(build_stack[stack_size])
        var data_base = node * 4
        var begin = Int(node_data[data_base])
        var end = Int(node_data[data_base + 1])
        var count = end - begin

        var glx = LARGE
        var gly = LARGE
        var glz = LARGE
        var gux = -LARGE
        var guy = -LARGE
        var guz = -LARGE
        var clx = LARGE
        var cly = LARGE
        var clz = LARGE
        var cux = -LARGE
        var cuy = -LARGE
        var cuz = -LARGE
        var vglx = SIMD[DType.float32, W](LARGE)
        var vgly = SIMD[DType.float32, W](LARGE)
        var vglz = SIMD[DType.float32, W](LARGE)
        var vgux = SIMD[DType.float32, W](-LARGE)
        var vguy = SIMD[DType.float32, W](-LARGE)
        var vguz = SIMD[DType.float32, W](-LARGE)
        var vclx = SIMD[DType.float32, W](LARGE)
        var vcly = SIMD[DType.float32, W](LARGE)
        var vclz = SIMD[DType.float32, W](LARGE)
        var vcux = SIMD[DType.float32, W](-LARGE)
        var vcuy = SIMD[DType.float32, W](-LARGE)
        var vcuz = SIMD[DType.float32, W](-LARGE)
        var slot = begin
        while slot + W <= end:
            var primitives = primitive_order.load[width=W](slot).cast[DType.int]()
            var vlx = primitive_bounds.gather(primitives)
            var vly = primitive_bounds.gather(primitive_count + primitives)
            var vlz = primitive_bounds.gather(2 * primitive_count + primitives)
            var vux = primitive_bounds.gather(3 * primitive_count + primitives)
            var vuy = primitive_bounds.gather(4 * primitive_count + primitives)
            var vuz = primitive_bounds.gather(5 * primitive_count + primitives)
            vglx = min(vglx, vlx)
            vgly = min(vgly, vly)
            vglz = min(vglz, vlz)
            vgux = max(vgux, vux)
            vguy = max(vguy, vuy)
            vguz = max(vguz, vuz)
            vclx = min(vclx, vlx + vux)
            vcly = min(vcly, vly + vuy)
            vclz = min(vclz, vlz + vuz)
            vcux = max(vcux, vlx + vux)
            vcuy = max(vcuy, vly + vuy)
            vcuz = max(vcuz, vlz + vuz)
            slot += W
        for lane in range(W):
            glx = min(glx, vglx[lane])
            gly = min(gly, vgly[lane])
            glz = min(glz, vglz[lane])
            gux = max(gux, vgux[lane])
            guy = max(guy, vguy[lane])
            guz = max(guz, vguz[lane])
            clx = min(clx, vclx[lane])
            cly = min(cly, vcly[lane])
            clz = min(clz, vclz[lane])
            cux = max(cux, vcux[lane])
            cuy = max(cuy, vcuy[lane])
            cuz = max(cuz, vcuz[lane])
        while slot < end:
            var primitive = Int(primitive_order[slot])
            var lx = primitive_bounds[primitive]
            var ly = primitive_bounds[primitive_count + primitive]
            var lz = primitive_bounds[2 * primitive_count + primitive]
            var ux = primitive_bounds[3 * primitive_count + primitive]
            var uy = primitive_bounds[4 * primitive_count + primitive]
            var uz = primitive_bounds[5 * primitive_count + primitive]
            glx = min(glx, lx)
            gly = min(gly, ly)
            glz = min(glz, lz)
            gux = max(gux, ux)
            guy = max(guy, uy)
            guz = max(guz, uz)
            clx = min(clx, lx + ux)
            cly = min(cly, ly + uy)
            clz = min(clz, lz + uz)
            cux = max(cux, lx + ux)
            cuy = max(cuy, ly + uy)
            cuz = max(cuz, lz + uz)
            slot += 1

        var bounds_base = node * 6
        node_bounds[bounds_base] = glx
        node_bounds[bounds_base + 1] = gly
        node_bounds[bounds_base + 2] = glz
        node_bounds[bounds_base + 3] = gux
        node_bounds[bounds_base + 4] = guy
        node_bounds[bounds_base + 5] = guz

        if count <= 4:
            node_data[data_base] = -1
            node_data[data_base + 1] = -1
            node_data[data_base + 2] = Int32(begin)
            node_data[data_base + 3] = Int32(count)
            if only_root:
                return False
            continue

        var num_bins = min(MAX_BINS, Int(Float32(4.0) + Float32(0.05) * Float32(count)))
        var dx = max(CENTROID_EPS, cux - clx)
        var dy = max(CENTROID_EPS, cuy - cly)
        var dz = max(CENTROID_EPS, cuz - clz)
        var sx = Float32(0.0)
        var sy = Float32(0.0)
        var sz = Float32(0.0)
        if dx > CENTROID_EPS:
            sx = Float32(0.99) * Float32(num_bins) / dx
        if dy > CENTROID_EPS:
            sy = Float32(0.99) * Float32(num_bins) / dy
        if dz > CENTROID_EPS:
            sz = Float32(0.99) * Float32(num_bins) / dz

        for index in range(num_bins * 3):
            bin_counts[index] = 0
            var bb = index * 6
            bin_bounds[bb] = LARGE
            bin_bounds[bb + 1] = LARGE
            bin_bounds[bb + 2] = LARGE
            bin_bounds[bb + 3] = -LARGE
            bin_bounds[bb + 4] = -LARGE
            bin_bounds[bb + 5] = -LARGE

        for slot in range(begin, end):
            var primitive = Int(primitive_order[slot])
            var lx = primitive_bounds[primitive]
            var ly = primitive_bounds[primitive_count + primitive]
            var lz = primitive_bounds[2 * primitive_count + primitive]
            var ux = primitive_bounds[3 * primitive_count + primitive]
            var uy = primitive_bounds[4 * primitive_count + primitive]
            var uz = primitive_bounds[5 * primitive_count + primitive]
            var ix = bin_index(lx + ux, clx, sx, num_bins)
            var iy = bin_index(ly + uy, cly, sy, num_bins)
            var iz = bin_index(lz + uz, clz, sz, num_bins)
            extend_bin(bin_bounds, bin_counts, ix, 0, lx, ly, lz, ux, uy, uz)
            extend_bin(bin_bounds, bin_counts, iy, 1, lx, ly, lz, ux, uy, uz)
            extend_bin(bin_bounds, bin_counts, iz, 2, lx, ly, lz, ux, uy, uz)

        var best_sah = LARGE
        var best_dim = -1
        var best_pos = 0
        for dim in range(3):
            var rlx = LARGE
            var rly = LARGE
            var rlz = LARGE
            var rux = -LARGE
            var ruy = -LARGE
            var ruz = -LARGE
            var rcount: Int32 = 0
            var bin = num_bins - 1
            while bin > 0:
                var index = bin * 3 + dim
                var bb = index * 6
                rcount += bin_counts[index]
                rlx = min(rlx, bin_bounds[bb])
                rly = min(rly, bin_bounds[bb + 1])
                rlz = min(rlz, bin_bounds[bb + 2])
                rux = max(rux, bin_bounds[bb + 3])
                ruy = max(ruy, bin_bounds[bb + 4])
                ruz = max(ruz, bin_bounds[bb + 5])
                right_counts[bin] = rcount
                var rb = bin * 6
                right_bounds[rb] = rlx
                right_bounds[rb + 1] = rly
                right_bounds[rb + 2] = rlz
                right_bounds[rb + 3] = rux
                right_bounds[rb + 4] = ruy
                right_bounds[rb + 5] = ruz
                bin -= 1

            var llx = LARGE
            var lly = LARGE
            var llz = LARGE
            var lux = -LARGE
            var luy = -LARGE
            var luz = -LARGE
            var lcount: Int32 = 0
            for split_pos in range(1, num_bins):
                var index = (split_pos - 1) * 3 + dim
                var bb = index * 6
                lcount += bin_counts[index]
                llx = min(llx, bin_bounds[bb])
                lly = min(lly, bin_bounds[bb + 1])
                llz = min(llz, bin_bounds[bb + 2])
                lux = max(lux, bin_bounds[bb + 3])
                luy = max(luy, bin_bounds[bb + 4])
                luz = max(luz, bin_bounds[bb + 5])
                var rb = split_pos * 6
                var left_blocks = (Int(lcount) + 3) // 4
                var right_blocks = (Int(right_counts[split_pos]) + 3) // 4
                var sah = (
                    half_area(llx, lly, llz, lux, luy, luz) * Float32(left_blocks)
                    + half_area(
                        right_bounds[rb],
                        right_bounds[rb + 1],
                        right_bounds[rb + 2],
                        right_bounds[rb + 3],
                        right_bounds[rb + 4],
                        right_bounds[rb + 5],
                    )
                    * Float32(right_blocks)
                )
                if sah < best_sah:
                    if (dim == 0 and sx != 0.0) or (dim == 1 and sy != 0.0) or (
                        dim == 2 and sz != 0.0
                    ):
                        best_sah = sah
                        best_dim = dim
                        best_pos = split_pos

        var make_leaf = False
        if count <= MAX_LEAF_SIZE and best_dim >= 0:
            var leaf_sah = half_area(glx, gly, glz, gux, guy, guz) * Float32((count + 3) // 4)
            var split_sah = half_area(glx, gly, glz, gux, guy, guz) + best_sah
            make_leaf = leaf_sah <= split_sah
        if make_leaf:
            node_data[data_base] = -1
            node_data[data_base + 1] = -1
            node_data[data_base + 2] = Int32(begin)
            node_data[data_base + 3] = Int32(count)
            if only_root:
                return False
            continue

        var middle = (begin + end) // 2
        if best_dim >= 0:
            var offset = clx
            var scale = sx
            if best_dim == 1:
                offset = cly
                scale = sy
            elif best_dim == 2:
                offset = clz
                scale = sz
            var left = begin
            var right = end - 1
            while left <= right:
                var primitive = Int(primitive_order[left])
                var center = (
                    primitive_bounds[best_dim * primitive_count + primitive]
                    + primitive_bounds[
                        (best_dim + 3) * primitive_count + primitive
                    ]
                )
                if bin_index(center, offset, scale, num_bins) < best_pos:
                    left += 1
                else:
                    var temporary = primitive_order[left]
                    primitive_order[left] = primitive_order[right]
                    primitive_order[right] = temporary
                    right -= 1
            middle = left
            if middle == begin or middle == end:
                middle = (begin + end) // 2

        var left_node = nodes
        var right_node = left_node + 1
        nodes += 2
        node_data[data_base] = Int32(left_node)
        node_data[data_base + 1] = Int32(right_node)
        node_data[data_base + 2] = 0
        node_data[data_base + 3] = 0
        node_data[left_node * 4] = Int32(begin)
        node_data[left_node * 4 + 1] = Int32(middle)
        node_data[right_node * 4] = Int32(middle)
        node_data[right_node * 4 + 1] = Int32(end)
        if only_root:
            return True
        build_stack[stack_size] = Int32(right_node)
        build_stack[stack_size + 1] = Int32(left_node)
        stack_size += 2

    return False


# Embree: kernels/builders/heuristic_binning.h BinMapping/BinInfoT::best
# and kernels/builders/bvh_builder_sah.h GeneralBVHBuilder::recurse.
def build_binned_sah(
    triangles: F32Ptr,
    primitive_bounds: F32Ptr,
    primitive_count: Int,
    node_bounds: F32Ptr,
    node_data: I32Ptr,
    primitive_order: I32Ptr,
    build_stack: I32Ptr,
    bin_bounds: F32Ptr,
    bin_counts: I32Ptr,
    right_bounds: F32Ptr,
    right_counts: I32Ptr,
) -> Int:
    if primitive_count <= 0:
        return 0

    precompute_bounds(triangles, primitive_bounds, primitive_count)
    comptime W = simdwidthof[DType.float64]()
    var primitive = 0
    while primitive + W <= primitive_count:
        primitive_order.store(
            primitive,
            iota[DType.int32, W](Int32(primitive)),
        )
        primitive += W
    while primitive < primitive_count:
        primitive_order[primitive] = Int32(primitive)
        primitive += 1

    node_data[0] = 0
    node_data[1] = Int32(primitive_count)
    var nodes = 1
    if primitive_count < BUILD_PARALLEL_THRESHOLD:
        _ = build_subtree(
            primitive_bounds,
            primitive_count,
            node_bounds,
            node_data,
            primitive_order,
            build_stack,
            bin_bounds,
            bin_counts,
            right_bounds,
            right_counts,
            nodes,
            0,
            False,
        )
        return nodes

    var desired_tasks = min(
        BUILD_MAX_WORKERS,
        min(num_physical_cores(), max(2, primitive_count // 2_048)),
    )
    var frontier_count = 1
    var frontier_cursor = 0
    build_stack[0] = 0
    while frontier_count < desired_tasks and frontier_cursor < frontier_count:
        var node = Int(build_stack[frontier_cursor])
        var internal = build_subtree(
            primitive_bounds,
            primitive_count,
            node_bounds,
            node_data,
            primitive_order,
            build_stack + BUILD_MAX_WORKERS,
            bin_bounds,
            bin_counts,
            right_bounds,
            right_counts,
            nodes,
            node,
            True,
        )
        if internal:
            var data_base = node * 4
            build_stack[frontier_cursor] = node_data[data_base]
            build_stack[frontier_count] = node_data[data_base + 1]
            frontier_count += 1
            frontier_cursor += 1
        else:
            frontier_count -= 1
            if frontier_cursor < frontier_count:
                build_stack[frontier_cursor] = build_stack[frontier_count]

    if frontier_count <= 0:
        return nodes
    var top_node_count = nodes
    for task in range(frontier_count):
        var node = Int(build_stack[task])
        var data_base = node * 4
        build_stack[BUILD_MAX_WORKERS + task] = Int32(
            Int(node_data[data_base + 1]) - Int(node_data[data_base])
        )

    @parameter
    @__copy_capture(
        primitive_bounds,
        primitive_count,
        node_bounds,
        node_data,
        primitive_order,
        build_stack,
        bin_bounds,
        bin_counts,
        right_bounds,
        right_counts,
        frontier_count,
        top_node_count,
    )
    @always_inline
    def build_frontier(task: Int):
        var stack_offset = 2 * BUILD_MAX_WORKERS
        var node_region = top_node_count
        for previous in range(task):
            var previous_count = Int(
                build_stack[BUILD_MAX_WORKERS + previous]
            )
            stack_offset += 2 * previous_count - 1
            node_region += 2 * previous_count - 2
        var node = Int(build_stack[task])
        var local_nodes = node_region
        _ = build_subtree(
            primitive_bounds,
            primitive_count,
            node_bounds,
            node_data,
            primitive_order,
            build_stack + stack_offset,
            bin_bounds + task * (MAX_BINS * 3 * 6),
            bin_counts + task * (MAX_BINS * 3),
            right_bounds + task * (MAX_BINS * 6),
            right_counts + task * MAX_BINS,
            local_nodes,
            node,
            False,
        )
        right_counts[task * MAX_BINS] = Int32(local_nodes)

    parallelize[build_frontier](frontier_count, frontier_count)

    var compact_node = top_node_count
    var source_node = top_node_count
    for task in range(frontier_count):
        var node = Int(build_stack[task])
        var end_node = Int(right_counts[task * MAX_BINS])
        var built_count = end_node - source_node
        var delta = compact_node - source_node
        if built_count > 0 and delta != 0:
            var data_base = node * 4
            node_data[data_base] += Int32(delta)
            node_data[data_base + 1] += Int32(delta)
            var source_bound = source_node * 6
            var compact_bound = compact_node * 6
            var bound_count = built_count * 6
            var bound = 0
            while bound + W <= bound_count:
                node_bounds.store(
                    compact_bound + bound,
                    node_bounds.load[width=W](source_bound + bound),
                )
                bound += W
            while bound < bound_count:
                node_bounds[compact_bound + bound] = node_bounds[
                    source_bound + bound
                ]
                bound += 1
            for local_node in range(built_count):
                var source_data = (source_node + local_node) * 4
                var compact_data = (compact_node + local_node) * 4
                var left = node_data[source_data]
                var right = node_data[source_data + 1]
                if node_data[source_data + 3] == 0:
                    left += Int32(delta)
                    right += Int32(delta)
                node_data[compact_data] = left
                node_data[compact_data + 1] = right
                node_data[compact_data + 2] = node_data[source_data + 2]
                node_data[compact_data + 3] = node_data[source_data + 3]
        compact_node += built_count
        var root_count = Int(build_stack[BUILD_MAX_WORKERS + task])
        source_node += 2 * root_count - 2
    return compact_node


# Embree: common/math/vec3.h rcp_safe.
@always_inline
def safe_rcp(value: Float32) -> Float32:
    if abs(value) < MIN_RCP_INPUT:
        return Float32(1.0) / MIN_RCP_INPUT
    return Float32(1.0) / value


# Embree: kernels/bvh/node_intersector1.h intersectNode<AABBNode, false>.
@always_inline
def intersect_box(
    bounds: F32Ptr,
    node: Int,
    ox: Float32,
    oy: Float32,
    oz: Float32,
    rdx: Float32,
    rdy: Float32,
    rdz: Float32,
    ray_near: Float32,
    ray_far: Float32,
) -> Float32:
    var base = node * 6
    var tx0 = -LARGE
    var tx1 = LARGE
    var ty0 = -LARGE
    var ty1 = LARGE
    var tz0 = -LARGE
    var tz1 = LARGE
    if abs(rdx) >= Float32(1.0) / MIN_RCP_INPUT:
        if ox < bounds[base] or ox > bounds[base + 3]:
            return LARGE
    else:
        tx0 = (bounds[base] - ox) * rdx
        tx1 = (bounds[base + 3] - ox) * rdx
    if abs(rdy) >= Float32(1.0) / MIN_RCP_INPUT:
        if oy < bounds[base + 1] or oy > bounds[base + 4]:
            return LARGE
    else:
        ty0 = (bounds[base + 1] - oy) * rdy
        ty1 = (bounds[base + 4] - oy) * rdy
    if abs(rdz) >= Float32(1.0) / MIN_RCP_INPUT:
        if oz < bounds[base + 2] or oz > bounds[base + 5]:
            return LARGE
    else:
        tz0 = (bounds[base + 2] - oz) * rdz
        tz1 = (bounds[base + 5] - oz) * rdz
    var near_value = max(ray_near, max(min(tx0, tx1), max(min(ty0, ty1), min(tz0, tz1))))
    var far_value = min(ray_far, min(max(tx0, tx1), min(max(ty0, ty1), max(tz0, tz1))))
    if near_value <= far_value:
        return near_value
    return LARGE


# Embree: kernels/geometry/triangle_intersector_pluecker.h PlueckerIntersector1::intersect.
# Embree: kernels/geometry/triangle_intersector_moeller.h MoellerTrumboreIntersector1::intersect depth interval.
@always_inline
def intersect_triangle(
    triangles: F32Ptr,
    primitive: Int,
    ox: Float32,
    oy: Float32,
    oz: Float32,
    dx: Float32,
    dy: Float32,
    dz: Float32,
    ray_near: Float32,
    ray_far: Float32,
    accept_equal_far: Bool,
    write_hit: Bool,
    hit_u: F32Ptr,
    hit_v: F32Ptr,
    hit_ng: F32Ptr,
    ray_index: Int,
) -> Float32:
    var base = primitive * 9
    var v0x = triangles[base] - ox
    var v0y = triangles[base + 1] - oy
    var v0z = triangles[base + 2] - oz
    var v1x = triangles[base + 3] - ox
    var v1y = triangles[base + 4] - oy
    var v1z = triangles[base + 5] - oz
    var v2x = triangles[base + 6] - ox
    var v2y = triangles[base + 7] - oy
    var v2z = triangles[base + 8] - oz

    var e0x = v2x - v0x
    var e0y = v2y - v0y
    var e0z = v2z - v0z
    var e1x = v0x - v1x
    var e1y = v0y - v1y
    var e1z = v0z - v1z
    var e2x = v1x - v2x
    var e2y = v1y - v2y
    var e2z = v1z - v2z

    var u = (
        (e0y * (v2z + v0z) - e0z * (v2y + v0y)) * dx
        + (e0z * (v2x + v0x) - e0x * (v2z + v0z)) * dy
        + (e0x * (v2y + v0y) - e0y * (v2x + v0x)) * dz
    )
    var v = (
        (e1y * (v0z + v1z) - e1z * (v0y + v1y)) * dx
        + (e1z * (v0x + v1x) - e1x * (v0z + v1z)) * dy
        + (e1x * (v0y + v1y) - e1y * (v0x + v1x)) * dz
    )
    var w = (
        (e2y * (v1z + v2z) - e2z * (v1y + v2y)) * dx
        + (e2z * (v1x + v2x) - e2x * (v1z + v2z)) * dy
        + (e2x * (v1y + v2y) - e2y * (v1x + v2x)) * dz
    )
    var uvw = u + v + w
    var epsilon = ULP * abs(uvw)
    if not (min(u, min(v, w)) >= -epsilon or max(u, max(v, w)) <= epsilon):
        return LARGE

    var abx = e0z * e1y
    var aby = e0x * e1z
    var abz = e0y * e1x
    var bcx = e1z * e2y
    var bcy = e1x * e2z
    var bcz = e1y * e2x
    var cross_ab_x = e0y * e1z - abx
    var cross_ab_y = e0z * e1x - aby
    var cross_ab_z = e0x * e1y - abz
    var cross_bc_x = e1y * e2z - bcx
    var cross_bc_y = e1z * e2x - bcy
    var cross_bc_z = e1x * e2y - bcz
    var ngx = cross_bc_x
    var ngy = cross_bc_y
    var ngz = cross_bc_z
    if abs(abx) < abs(bcx):
        ngx = cross_ab_x
    if abs(aby) < abs(bcy):
        ngy = cross_ab_y
    if abs(abz) < abs(bcz):
        ngz = cross_ab_z

    var denominator = Float32(2.0) * (ngx * dx + ngy * dy + ngz * dz)
    if denominator == 0.0:
        return LARGE
    var numerator = Float32(2.0) * (v0x * ngx + v0y * ngy + v0z * ngz)
    var distance = numerator / denominator
    if distance <= ray_near or distance > ray_far or (
        distance == ray_far and not accept_equal_far
    ):
        return LARGE

    if write_hit:
        var reciprocal_uvw = Float32(0.0)
        if abs(uvw) >= MIN_RCP_INPUT:
            reciprocal_uvw = Float32(1.0) / uvw
        hit_u[ray_index] = min(u * reciprocal_uvw, Float32(1.0))
        hit_v[ray_index] = min(v * reciprocal_uvw, Float32(1.0))
        var ng_base = ray_index * 3
        hit_ng[ng_base] = ngx
        hit_ng[ng_base + 1] = ngy
        hit_ng[ng_base + 2] = ngz
    return distance


# Embree: kernels/bvh/bvh_intersector1.cpp BVHNIntersector1::intersect.
def trace_one(
    triangles: F32Ptr,
    node_bounds: F32Ptr,
    node_data: I32Ptr,
    primitive_order: I32Ptr,
    traversal_stack: I32Ptr,
    origins: F32Ptr,
    directions: F32Ptr,
    ray_nears: F32Ptr,
    ray_fars: F32Ptr,
    hit_ids: I32Ptr,
    hit_t: F32Ptr,
    hit_u: F32Ptr,
    hit_v: F32Ptr,
    hit_ng: F32Ptr,
    primitive_count: Int,
    ray_index: Int,
):
    var ray_base = ray_index * 3
    var ox = origins[ray_base]
    var oy = origins[ray_base + 1]
    var oz = origins[ray_base + 2]
    var dx = directions[ray_base]
    var dy = directions[ray_base + 1]
    var dz = directions[ray_base + 2]
    var rdx = safe_rcp(dx)
    var rdy = safe_rcp(dy)
    var rdz = safe_rcp(dz)
    var ray_near = max(Float32(0.0), ray_nears[ray_index])
    var closest = max(Float32(0.0), ray_fars[ray_index])
    var closest_id: Int32 = -1
    hit_u[ray_index] = 0.0
    hit_v[ray_index] = 0.0
    hit_ng[ray_base] = 0.0
    hit_ng[ray_base + 1] = 0.0
    hit_ng[ray_base + 2] = 0.0

    var stack_size = 1
    traversal_stack[0] = 0
    while stack_size > 0:
        stack_size -= 1
        var node = Int(traversal_stack[stack_size])
        if intersect_box(
            node_bounds, node, ox, oy, oz, rdx, rdy, rdz, ray_near, closest
        ) == LARGE:
            continue
        var data_base = node * 4
        var leaf_count = Int(node_data[data_base + 3])
        if leaf_count > 0:
            var start = Int(node_data[data_base + 2])
            for slot in range(start, start + leaf_count):
                var primitive = Int(primitive_order[slot])
                var distance = intersect_triangle(
                    triangles,
                    primitive,
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    ray_near,
                    closest,
                    closest_id < 0,
                    True,
                    hit_u,
                    hit_v,
                    hit_ng,
                    ray_index,
                )
                if distance != LARGE:
                    closest = distance
                    closest_id = Int32(primitive)
        else:
            var left = Int(node_data[data_base])
            var right = Int(node_data[data_base + 1])
            var left_near = intersect_box(
                node_bounds, left, ox, oy, oz, rdx, rdy, rdz, ray_near, closest
            )
            var right_near = intersect_box(
                node_bounds, right, ox, oy, oz, rdx, rdy, rdz, ray_near, closest
            )
            if left_near != LARGE and right_near != LARGE:
                if stack_size + 2 > STACK_SIZE:
                    # A pathological, highly unbalanced BVH can exceed the
                    # fixed caller-owned stack. Linear fallback prevents an
                    # out-of-bounds write across the FFI boundary.
                    for primitive in range(primitive_count):
                        var distance = intersect_triangle(
                            triangles, primitive, ox, oy, oz, dx, dy, dz,
                            ray_near, closest, closest_id < 0, True, hit_u,
                            hit_v, hit_ng, ray_index,
                        )
                        if distance != LARGE:
                            closest = distance
                            closest_id = Int32(primitive)
                    stack_size = 0
                    break
                if left_near < right_near:
                    traversal_stack[stack_size] = Int32(right)
                    traversal_stack[stack_size + 1] = Int32(left)
                else:
                    traversal_stack[stack_size] = Int32(left)
                    traversal_stack[stack_size + 1] = Int32(right)
                stack_size += 2
            elif left_near != LARGE:
                traversal_stack[stack_size] = Int32(left)
                stack_size += 1
            elif right_near != LARGE:
                traversal_stack[stack_size] = Int32(right)
                stack_size += 1

    hit_ids[ray_index] = closest_id
    hit_t[ray_index] = closest


# Embree: kernels/bvh/bvh_intersector1.cpp BVHNIntersector1::occluded.
def occluded_one(
    triangles: F32Ptr,
    node_bounds: F32Ptr,
    node_data: I32Ptr,
    primitive_order: I32Ptr,
    traversal_stack: I32Ptr,
    origins: F32Ptr,
    directions: F32Ptr,
    ray_nears: F32Ptr,
    ray_fars: F32Ptr,
    occluded: U8Ptr,
    primitive_count: Int,
    ray_index: Int,
):
    var ray_base = ray_index * 3
    var ox = origins[ray_base]
    var oy = origins[ray_base + 1]
    var oz = origins[ray_base + 2]
    var dx = directions[ray_base]
    var dy = directions[ray_base + 1]
    var dz = directions[ray_base + 2]
    var rdx = safe_rcp(dx)
    var rdy = safe_rcp(dy)
    var rdz = safe_rcp(dz)
    var ray_near = max(Float32(0.0), ray_nears[ray_index])
    var ray_far = max(Float32(0.0), ray_fars[ray_index])
    occluded[ray_index] = 0
    var stack_size = 1
    traversal_stack[0] = 0
    while stack_size > 0:
        stack_size -= 1
        var node = Int(traversal_stack[stack_size])
        if intersect_box(
            node_bounds, node, ox, oy, oz, rdx, rdy, rdz, ray_near, ray_far
        ) == LARGE:
            continue
        var data_base = node * 4
        var leaf_count = Int(node_data[data_base + 3])
        if leaf_count > 0:
            var start = Int(node_data[data_base + 2])
            for slot in range(start, start + leaf_count):
                var primitive = Int(primitive_order[slot])
                var distance = intersect_triangle(
                    triangles,
                    primitive,
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    ray_near,
                    ray_far,
                    True,
                    False,
                    triangles,
                    triangles,
                    triangles,
                    ray_index,
                )
                if distance != LARGE:
                    occluded[ray_index] = 1
                    return
        else:
            if stack_size + 2 > STACK_SIZE:
                for primitive in range(primitive_count):
                    if intersect_triangle(
                        triangles, primitive, ox, oy, oz, dx, dy, dz,
                        ray_near, ray_far, True, False, triangles, triangles,
                        triangles, ray_index,
                    ) != LARGE:
                        occluded[ray_index] = 1
                        return
                return
            traversal_stack[stack_size] = node_data[data_base]
            traversal_stack[stack_size + 1] = node_data[data_base + 1]
            stack_size += 2


@export("me_build_bvh")
def me_build_bvh(
    triangles_address: Int,
    primitive_bounds_address: Int,
    primitive_count: Int,
    node_bounds_address: Int,
    node_data_address: Int,
    primitive_order_address: Int,
    build_stack_address: Int,
    bin_bounds_address: Int,
    bin_counts_address: Int,
    right_bounds_address: Int,
    right_counts_address: Int,
) abi("C") -> Int:
    initialize_runtime()
    return build_binned_sah(
        fp(triangles_address),
        fp(primitive_bounds_address),
        primitive_count,
        fp(node_bounds_address),
        ip(node_data_address),
        ip(primitive_order_address),
        ip(build_stack_address),
        fp(bin_bounds_address),
        ip(bin_counts_address),
        fp(right_bounds_address),
        ip(right_counts_address),
    )


@export("me_intersect_stream")
def me_intersect_stream(
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
    ray_count: Int,
) abi("C"):
    initialize_runtime()
    if ray_count <= 0:
        return
    var triangles = fp(triangles_address)
    var node_bounds = fp(node_bounds_address)
    var node_data = ip(node_data_address)
    var primitive_order = ip(primitive_order_address)
    var origins = fp(origins_address)
    var directions = fp(directions_address)
    var ray_nears = fp(ray_nears_address)
    var ray_fars = fp(ray_fars_address)
    var hit_ids = ip(hit_ids_address)
    var hit_t = fp(hit_t_address)
    var hit_u = fp(hit_u_address)
    var hit_v = fp(hit_v_address)
    var hit_ng = fp(hit_ng_address)
    var traversal_stacks = ip(traversal_stacks_address)
    var tasks = 1
    if ray_count >= PARALLEL_THRESHOLD:
        tasks = min(MAX_WORKERS, min(num_physical_cores(), max(1, ray_count // 1024)))
    if tasks > 1:

        @parameter
        @__copy_capture(
            triangles,
            node_bounds,
            node_data,
            primitive_order,
            traversal_stacks,
            origins,
            directions,
            ray_nears,
            ray_fars,
            hit_ids,
            hit_t,
            hit_u,
            hit_v,
            hit_ng,
            primitive_count,
            ray_count,
            tasks,
        )
        @always_inline
        def process_chunk(task: Int):
            var begin = ray_count * task // tasks
            var end = ray_count * (task + 1) // tasks
            for ray in range(begin, end):
                trace_one(
                    triangles,
                    node_bounds,
                    node_data,
                    primitive_order,
                    traversal_stacks + task * STACK_SIZE,
                    origins,
                    directions,
                    ray_nears,
                    ray_fars,
                    hit_ids,
                    hit_t,
                    hit_u,
                    hit_v,
                    hit_ng,
                    primitive_count,
                    ray,
                )

        parallelize[process_chunk](tasks, tasks)
    else:
        for ray in range(ray_count):
            trace_one(
                triangles,
                node_bounds,
                node_data,
                primitive_order,
                traversal_stacks,
                origins,
                directions,
                ray_nears,
                ray_fars,
                hit_ids,
                hit_t,
                hit_u,
                hit_v,
                hit_ng,
                primitive_count,
                ray,
            )


@export("me_occluded_stream")
def me_occluded_stream(
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
    ray_count: Int,
) abi("C"):
    initialize_runtime()
    if ray_count <= 0:
        return
    var triangles = fp(triangles_address)
    var node_bounds = fp(node_bounds_address)
    var node_data = ip(node_data_address)
    var primitive_order = ip(primitive_order_address)
    var origins = fp(origins_address)
    var directions = fp(directions_address)
    var ray_nears = fp(ray_nears_address)
    var ray_fars = fp(ray_fars_address)
    var result = bp(occluded_address)
    var traversal_stacks = ip(traversal_stacks_address)
    var tasks = 1
    if ray_count >= PARALLEL_THRESHOLD:
        tasks = min(MAX_WORKERS, min(num_physical_cores(), max(1, ray_count // 1024)))
    if tasks > 1:

        @parameter
        @__copy_capture(
            triangles,
            node_bounds,
            node_data,
            primitive_order,
            traversal_stacks,
            origins,
            directions,
            ray_nears,
            ray_fars,
            result,
            primitive_count,
            ray_count,
            tasks,
        )
        @always_inline
        def process_chunk(task: Int):
            var begin = ray_count * task // tasks
            var end = ray_count * (task + 1) // tasks
            for ray in range(begin, end):
                occluded_one(
                    triangles,
                    node_bounds,
                    node_data,
                    primitive_order,
                    traversal_stacks + task * STACK_SIZE,
                    origins,
                    directions,
                    ray_nears,
                    ray_fars,
                    result,
                    primitive_count,
                    ray,
                )

        parallelize[process_chunk](tasks, tasks)
    else:
        for ray in range(ray_count):
            occluded_one(
                triangles,
                node_bounds,
                node_data,
                primitive_order,
                traversal_stacks,
                origins,
                directions,
                ray_nears,
                ray_fars,
                result,
                primitive_count,
                ray,
            )


@export("me_stack_size")
def me_stack_size() abi("C") -> Int:
    return STACK_SIZE
