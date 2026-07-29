from __future__ import annotations

import numpy as np
import pytest

from embree_reference import EmbreeScene
from mojo_embree import BVH


def square_mesh():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, triangles


def compare(vertices, triangles, origins, directions, tnear=0.0, tfar=np.inf):
    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    count = len(origins)
    near = np.broadcast_to(np.asarray(tnear, dtype=np.float32), (count,)).copy()
    far = np.broadcast_to(np.asarray(tfar, dtype=np.float32), (count,)).copy()
    ours = BVH(vertices, triangles).intersect(
        origins, directions, tnear=near, tfar=far
    )
    reference = EmbreeScene(vertices, triangles)
    expected = reference.intersect(origins, directions, near, far)
    reference.close()
    np.testing.assert_array_equal(ours.hit, expected[0] >= 0)
    common = ours.hit
    np.testing.assert_array_equal(ours.primitive_id[common], expected[0][common])
    np.testing.assert_allclose(
        ours.distance[common], expected[1][common], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(ours.u[common], expected[2][common], atol=3e-6)
    np.testing.assert_allclose(ours.v[common], expected[3][common], atol=3e-6)
    np.testing.assert_allclose(
        ours.geometric_normal[common], expected[4][common], rtol=2e-6, atol=2e-6
    )
    return ours


def test_single_triangle_front_back_and_miss_parity():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    origins = [[0.25, 0.25, 1], [0.25, 0.25, -1], [2, 2, 1]]
    directions = [[0, 0, -1], [0, 0, 1], [0, 0, -1]]
    hits = compare(vertices, triangles, origins, directions)
    np.testing.assert_array_equal(hits.hit, [True, True, False])


def test_random_stream_closest_hit_parity():
    rng = np.random.default_rng(42)
    vertices = rng.uniform(-1, 1, size=(240, 3)).astype(np.float32)
    triangles = np.arange(240, dtype=np.int32).reshape(-1, 3)
    origins = rng.uniform(-2, 2, size=(400, 3)).astype(np.float32)
    directions = -origins + rng.normal(0, 0.25, size=(400, 3)).astype(np.float32)
    compare(vertices, triangles, origins, directions)


@pytest.mark.parametrize("packet_size", [1, 4, 8, 16])
def test_packet_sizes_match_embree(packet_size):
    vertices, triangles = square_mesh()
    x = np.linspace(-0.25, 1.25, packet_size, dtype=np.float32)
    origins = np.column_stack((x, np.full(packet_size, 0.5), np.ones(packet_size)))
    directions = np.tile([0, 0, -1], (packet_size, 1)).astype(np.float32)
    bvh = BVH(vertices, triangles)
    packet = bvh.intersect_packet(origins, directions)
    stream = compare(vertices, triangles, origins, directions)
    np.testing.assert_array_equal(packet.primitive_id, stream.primitive_id)
    np.testing.assert_allclose(packet.distance, stream.distance)


def test_ray_interval_boundaries_match_embree():
    vertices, triangles = square_mesh()
    origins = np.tile([0.25, 0.25, 1], (5, 1)).astype(np.float32)
    directions = np.tile([0, 0, -1], (5, 1)).astype(np.float32)
    compare(
        vertices,
        triangles,
        origins,
        directions,
        tnear=np.array([0, 1, 1.0001, 0, 0], np.float32),
        tfar=np.array([0.9999, 1, 2, 1, np.inf], np.float32),
    )


def test_occlusion_matches_embree():
    vertices, triangles = square_mesh()
    origins = np.array(
        [[0.25, 0.25, 1], [1.5, 0.5, 1], [0.75, 0.75, -1], [0.5, 0.5, 2]],
        dtype=np.float32,
    )
    directions = np.array(
        [[0, 0, -1], [0, 0, -1], [0, 0, 1], [0, 0, -1]], dtype=np.float32
    )
    near = np.zeros(4, dtype=np.float32)
    far = np.array([2, 2, 2, 1], dtype=np.float32)
    ours = BVH(vertices, triangles).occluded(
        origins, directions, tnear=near, tfar=far
    )
    reference = EmbreeScene(vertices, triangles)
    expected = reference.occluded(origins, directions, near, far)
    reference.close()
    np.testing.assert_array_equal(ours, expected)


def test_parallel_stream_and_occlusion_match_embree():
    vertices, triangles = square_mesh()
    rng = np.random.default_rng(19)
    origins = rng.uniform(-0.25, 1.25, size=(5000, 3)).astype(np.float32)
    origins[:, 2] = 1.0
    directions = np.tile([0, 0, -1], (len(origins), 1)).astype(np.float32)
    hits = compare(vertices, triangles, origins, directions)
    bvh = BVH(vertices, triangles)
    actual_occluded = bvh.occluded(origins, directions)
    reference = EmbreeScene(vertices, triangles)
    near = np.zeros(len(origins), dtype=np.float32)
    far = np.full(len(origins), np.inf, dtype=np.float32)
    expected_occluded = reference.occluded(origins, directions, near, far)
    reference.close()
    np.testing.assert_array_equal(actual_occluded, expected_occluded)
    np.testing.assert_array_equal(hits.hit, actual_occluded)


def test_watertight_shared_edge_has_no_cracks():
    vertices, triangles = square_mesh()
    values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    origins = np.column_stack((values, values, np.ones_like(values)))
    directions = np.tile([0, 0, -1], (len(values), 1)).astype(np.float32)
    hits = compare(vertices, triangles, origins, directions)
    assert hits.hit.all()


def test_empty_mesh_and_empty_stream():
    bvh = BVH(np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))
    hits = bvh.intersect([[0, 0, 0]], [[0, 0, 1]])
    assert bvh.node_count == 0
    assert not hits.hit[0]
    assert not bvh.occluded([[0, 0, 0]], [[0, 0, 1]])[0]
    empty = bvh.intersect(np.empty((0, 3)), np.empty((0, 3)))
    assert len(empty.distance) == 0


def test_duplicate_vertices_zero_area_and_unreferenced_vertices():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0.5, 0.5, 7],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32)
    origins = np.array([[0.2, 0.2, 1], [0.8, 0.8, 1]], dtype=np.float32)
    directions = np.tile([0, 0, -1], (2, 1)).astype(np.float32)
    compare(vertices, triangles, origins, directions)


def test_non_manifold_edge_parity():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32)
    origins = np.array(
        [[0.5, 0.01, 2], [0.5, 0.2, 1], [0.5, -0.2, 1]], np.float32
    )
    directions = np.tile([0, 0, -1], (3, 1)).astype(np.float32)
    compare(vertices, triangles, origins, directions)


def test_bvh_build_invariants():
    rng = np.random.default_rng(7)
    soup = rng.normal(size=(257, 3, 3)).astype(np.float32)
    bvh = BVH.from_triangle_soup(soup)
    assert 1 <= bvh.node_count <= 2 * len(soup) - 1
    np.testing.assert_array_equal(np.sort(bvh.primitive_order), np.arange(len(soup)))
    root = bvh.node_bounds[0]
    np.testing.assert_allclose(root[:3], soup.min(axis=(0, 1)))
    np.testing.assert_allclose(root[3:], soup.max(axis=(0, 1)))
    leaves = bvh.node_data[: bvh.node_count, 3] > 0
    assert bvh.node_data[: bvh.node_count, 3][leaves].sum() == len(soup)
    assert bvh.node_data[: bvh.node_count, 3][leaves].max() <= 32


def test_simd_tail_and_serial_build_threshold():
    rng = np.random.default_rng(23)
    soup = rng.normal(size=(259, 3, 3)).astype(np.float32)
    bvh = BVH.from_triangle_soup(soup)
    np.testing.assert_array_equal(
        np.sort(bvh.primitive_order), np.arange(len(soup))
    )
    np.testing.assert_allclose(bvh.node_bounds[0, :3], soup.min(axis=(0, 1)))
    np.testing.assert_allclose(bvh.node_bounds[0, 3:], soup.max(axis=(0, 1)))


def test_parallel_build_threshold_and_node_compaction():
    count = 262_147
    triangle = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )
    soup = np.broadcast_to(triangle, (count, 3, 3)).copy()
    bvh = BVH.from_triangle_soup(soup)
    assert bvh.node_count < 2 * count - 1
    np.testing.assert_array_equal(
        np.sort(bvh.primitive_order), np.arange(count)
    )
    hit = bvh.intersect([[0.25, 0.25, 1]], [[0, 0, -1]])
    assert hit.hit[0]
    assert hit.distance[0] == pytest.approx(1.0)


def test_input_validation():
    vertices, triangles = square_mesh()
    with pytest.raises(ValueError, match="out of bounds"):
        BVH(vertices, [[0, 1, 99]])
    with pytest.raises(ValueError, match="finite"):
        BVH([[0, 0, np.nan]], np.empty((0, 3), np.int32))
    bvh = BVH(vertices, triangles)
    with pytest.raises(ValueError, match="same ray count"):
        bvh.intersect([[0, 0, 1]], [[0, 0, -1], [0, 0, -1]])
    with pytest.raises(ValueError, match="packet size"):
        bvh.intersect_packet(np.zeros((3, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="ray intervals"):
        bvh.occluded([[0, 0, 1]], [[0, 0, -1]], tnear=2, tfar=1)
    with pytest.raises(ValueError, match="int32 range"):
        BVH(vertices, np.array([[0, 1, 2**32]], dtype=np.uint64))
    with pytest.raises(TypeError, match="integer indices"):
        BVH(vertices, np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        bvh.intersect([[0, 0, 1e100]], [[0, 0, -1]])
    with pytest.raises(ValueError, match="ray intervals"):
        bvh.intersect([[0, 0, 1]], [[0, 0, -1]], tnear=np.nan)
    with pytest.raises(TypeError, match="real numeric"):
        bvh.intersect([["x", 0, 1]], [[0, 0, -1]])


def test_noncontiguous_and_temporary_inputs_remain_valid_during_ffi_call():
    vertices, triangles = square_mesh()
    padded_vertices = np.zeros((len(vertices), 6), dtype=np.float64)
    padded_vertices[:, ::2] = vertices
    padded_rays = np.zeros((32, 6), dtype=np.float64)
    padded_rays[:, ::2] = [0.25, 0.25, 1.0]
    padded_directions = np.zeros((32, 6), dtype=np.float64)
    padded_directions[:, ::2] = [0.0, 0.0, -1.0]
    near = np.zeros(64, dtype=np.float64)[::2]
    far = np.full(64, 2.0, dtype=np.float64)[::2]

    bvh = BVH(padded_vertices[:, ::2], triangles[:, ::-1][:, ::-1])
    hits = bvh.intersect(
        padded_rays[:, ::2] + 0.0,
        padded_directions[:, ::2] + 0.0,
        tnear=near,
        tfar=far,
    )
    assert hits.hit.all()
    np.testing.assert_allclose(hits.distance, 1.0)


def test_float32_conversion_overflow_is_rejected():
    vertices, triangles = square_mesh()
    with pytest.raises(ValueError, match="finite"):
        BVH(vertices.astype(np.float64) * 1e100, triangles)
    bvh = BVH(vertices, triangles)
    with pytest.raises(ValueError, match="float32"):
        bvh.intersect([[0, 0, 1]], [[0, 0, -1]], tfar=1e100)
