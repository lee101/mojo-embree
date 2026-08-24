"""Python API for Embree-derived Mojo ray tracing kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._lib import lib

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
INT32_MAX = np.iinfo(np.int32).max


def _address(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty array")
    return address


def _float32(value: ArrayLike, name: str) -> FloatArray:
    source = np.asarray(value)
    if source.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        if source.ndim == 0:
            return source.astype(np.float32)
        return np.ascontiguousarray(source, dtype=np.float32)


def _vectors(value: ArrayLike, name: str) -> FloatArray:
    array = _float32(value, name)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or (n, 3)")
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (n, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _limits(value: ArrayLike | float, count: int, name: str) -> FloatArray:
    source = np.asarray(value)
    array = _float32(value, name)
    if np.any(np.isfinite(source) & np.isinf(array)):
        raise ValueError(f"{name} values must be representable as float32")
    if array.ndim == 0:
        return np.full(count, array.item(), dtype=np.float32)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    return array


@dataclass(frozen=True, slots=True)
class RayHits:
    """Closest-hit results for a ray stream."""

    primitive_id: IntArray
    distance: FloatArray
    u: FloatArray
    v: FloatArray
    geometric_normal: FloatArray

    @property
    def hit(self) -> NDArray[np.bool_]:
        return self.primitive_id >= 0


class BVH:
    """A reusable binned-SAH BVH over an indexed float32 triangle mesh."""

    def __init__(self, vertices: ArrayLike, triangles: ArrayLike):
        vertex_array = _float32(vertices, "vertices")
        raw_indices = np.asarray(triangles)
        if raw_indices.dtype.kind not in "iu":
            raise TypeError("triangles must contain integer indices")
        if raw_indices.size and (
            raw_indices.min() < 0 or raw_indices.max() > INT32_MAX
        ):
            raise ValueError("triangle indices are out of int32 range")
        index_array = np.ascontiguousarray(raw_indices, dtype=np.int32)
        if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
            raise ValueError("vertices must have shape (vertex_count, 3)")
        if index_array.ndim != 2 or index_array.shape[1] != 3:
            raise ValueError("triangles must have shape (triangle_count, 3)")
        if not np.isfinite(vertex_array).all():
            raise ValueError("vertices must contain finite values")
        if index_array.size and (
            index_array.min() < 0 or index_array.max() >= len(vertex_array)
        ):
            raise ValueError("triangle indices are out of bounds")

        self.vertices = vertex_array
        self.indices = index_array
        self.triangles = np.ascontiguousarray(
            vertex_array[index_array].reshape(-1, 9), dtype=np.float32
        )
        count = len(index_array)
        if count > INT32_MAX:
            raise ValueError("triangle count exceeds the int32 kernel limit")
        self.primitive_count = count
        self.primitive_bounds = np.empty((6, count), dtype=np.float32)
        capacity = max(1, 2 * count - 1)
        self.node_bounds = np.empty((capacity, 6), dtype=np.float32)
        self.node_data = np.empty((capacity, 4), dtype=np.int32)
        self.primitive_order = np.empty(count, dtype=np.int32)
        self.node_count = 0
        self._stack_size = int(lib().me_stack_size())
        if self._stack_size <= 0:
            raise RuntimeError("kernel returned an invalid traversal stack size")
        self._traversal_stack = np.empty((32, self._stack_size), dtype=np.int32)

        if count:
            build_stack = np.empty(capacity + 8, dtype=np.int32)
            bin_bounds = np.empty((4, 32 * 3 * 6), dtype=np.float32)
            bin_counts = np.empty((4, 32 * 3), dtype=np.int32)
            right_bounds = np.empty((4, 32 * 6), dtype=np.float32)
            right_counts = np.empty((4, 32), dtype=np.int32)
            self.node_count = int(
                lib().me_build_bvh(
                    _address(self.triangles),
                    _address(self.primitive_bounds),
                    count,
                    _address(self.node_bounds),
                    _address(self.node_data),
                    _address(self.primitive_order),
                    _address(build_stack),
                    _address(bin_bounds),
                    _address(bin_counts),
                    _address(right_bounds),
                    _address(right_counts),
                )
            )
            if not 1 <= self.node_count <= capacity:
                raise RuntimeError("kernel returned an invalid BVH node count")

    @classmethod
    def from_triangle_soup(cls, triangles: ArrayLike) -> "BVH":
        """Build from an array with shape ``(triangle_count, 3, 3)``."""
        soup = _float32(triangles, "triangles")
        if soup.ndim != 3 or soup.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (triangle_count, 3, 3)")
        vertices = soup.reshape(-1, 3)
        indices = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        return cls(vertices, indices)

    def intersect_stream(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        tnear: ArrayLike | float = 0.0,
        tfar: ArrayLike | float = np.inf,
    ) -> RayHits:
        """Return the closest triangle hit for each ray."""
        origin_array = _vectors(origins, "origins")
        direction_array = _vectors(directions, "directions")
        if len(origin_array) != len(direction_array):
            raise ValueError("origins and directions must contain the same ray count")
        count = len(origin_array)
        near_array = _limits(tnear, count, "tnear")
        far_array = _limits(tfar, count, "tfar")
        if (
            np.any(~np.isfinite(near_array))
            or np.any(np.isnan(far_array))
            or np.any(near_array < 0)
            or np.any(far_array < near_array)
        ):
            raise ValueError("ray intervals require 0 <= tnear <= tfar")

        if self.node_count:
            hit_ids = np.empty(count, dtype=np.int32)
            hit_t = np.empty(count, dtype=np.float32)
            hit_u = np.empty(count, dtype=np.float32)
            hit_v = np.empty(count, dtype=np.float32)
            hit_ng = np.empty((count, 3), dtype=np.float32)
        else:
            hit_ids = np.full(count, -1, dtype=np.int32)
            hit_t = far_array.copy()
            hit_u = np.zeros(count, dtype=np.float32)
            hit_v = np.zeros(count, dtype=np.float32)
            hit_ng = np.zeros((count, 3), dtype=np.float32)
        if count and self.node_count:
            lib().me_intersect_stream(
                _address(self.triangles),
                _address(self.node_bounds),
                _address(self.node_data),
                _address(self.primitive_order),
                _address(origin_array),
                _address(direction_array),
                _address(near_array),
                _address(far_array),
                _address(hit_ids),
                _address(hit_t),
                _address(hit_u),
                _address(hit_v),
                    _address(hit_ng),
                    _address(self._traversal_stack),
                    self.primitive_count,
                    count,
            )
        return RayHits(hit_ids, hit_t, hit_u, hit_v, hit_ng)

    def intersect_packet(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        tnear: ArrayLike | float = 0.0,
        tfar: ArrayLike | float = np.inf,
    ) -> RayHits:
        """Intersect a packet of 1, 4, 8, or 16 rays."""
        origin_array = _vectors(origins, "origins")
        if len(origin_array) not in (1, 4, 8, 16):
            raise ValueError("packet size must be 1, 4, 8, or 16")
        return self.intersect_stream(
            origin_array, directions, tnear=tnear, tfar=tfar
        )

    intersect = intersect_stream

    def occluded(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        tnear: ArrayLike | float = 0.0,
        tfar: ArrayLike | float = np.inf,
    ) -> NDArray[np.bool_]:
        """Return whether each ray segment has any triangle hit."""
        origin_array = _vectors(origins, "origins")
        direction_array = _vectors(directions, "directions")
        if len(origin_array) != len(direction_array):
            raise ValueError("origins and directions must contain the same ray count")
        count = len(origin_array)
        near_array = _limits(tnear, count, "tnear")
        far_array = _limits(tfar, count, "tfar")
        if (
            np.any(~np.isfinite(near_array))
            or np.any(np.isnan(far_array))
            or np.any(near_array < 0)
            or np.any(far_array < near_array)
        ):
            raise ValueError("ray intervals require 0 <= tnear <= tfar")

        result = np.zeros(count, dtype=np.bool_)
        if count and self.node_count:
            lib().me_occluded_stream(
                _address(self.triangles),
                _address(self.node_bounds),
                _address(self.node_data),
                _address(self.primitive_order),
                _address(origin_array),
                _address(direction_array),
                _address(near_array),
                _address(far_array),
                _address(result),
                _address(self._traversal_stack),
                self.primitive_count,
                count,
            )
        return result
