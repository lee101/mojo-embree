"""Minimal ctypes binding to the real Embree 4 C API used for parity tests."""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np

U = ctypes.c_uint32
F = ctypes.c_float
P = ctypes.c_void_p


class RTCRay(ctypes.Structure):
    _fields_ = [
        ("org_x", F),
        ("org_y", F),
        ("org_z", F),
        ("tnear", F),
        ("dir_x", F),
        ("dir_y", F),
        ("dir_z", F),
        ("time", F),
        ("tfar", F),
        ("mask", U),
        ("id", U),
        ("flags", U),
    ]


class RTCHit(ctypes.Structure):
    _fields_ = [
        ("Ng_x", F),
        ("Ng_y", F),
        ("Ng_z", F),
        ("u", F),
        ("v", F),
        ("primID", U),
        ("geomID", U),
        ("instID", U),
        ("instPrimID", U),
        ("padding", U * 3),
    ]


class RTCRayHit(ctypes.Structure):
    _fields_ = [("ray", RTCRay), ("hit", RTCHit)]


def _load() -> ctypes.CDLL:
    path = ctypes.util.find_library("embree4")
    if not path:
        raise RuntimeError("libembree4 is unavailable; install the pixi environment")
    library = ctypes.CDLL(path)
    library.rtcNewDevice.argtypes = [ctypes.c_char_p]
    library.rtcNewDevice.restype = P
    library.rtcReleaseDevice.argtypes = [P]
    library.rtcNewScene.argtypes = [P]
    library.rtcNewScene.restype = P
    library.rtcReleaseScene.argtypes = [P]
    library.rtcCommitScene.argtypes = [P]
    library.rtcNewGeometry.argtypes = [P, ctypes.c_int]
    library.rtcNewGeometry.restype = P
    library.rtcSetSharedGeometryBuffer.argtypes = [
        P,
        ctypes.c_int,
        U,
        ctypes.c_int,
        P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    library.rtcCommitGeometry.argtypes = [P]
    library.rtcAttachGeometry.argtypes = [P, P]
    library.rtcAttachGeometry.restype = U
    library.rtcReleaseGeometry.argtypes = [P]
    library.rtcIntersect1.argtypes = [P, ctypes.POINTER(RTCRayHit), P]
    library.rtcOccluded1.argtypes = [P, ctypes.POINTER(RTCRay), P]
    return library


LIB = _load()
INVALID = 0xFFFFFFFF


class EmbreeScene:
    def __init__(self, vertices, triangles):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        self.device = LIB.rtcNewDevice(None)
        if not self.device:
            raise RuntimeError("rtcNewDevice failed")
        self.scene = LIB.rtcNewScene(self.device)
        geometry = LIB.rtcNewGeometry(self.device, 0)
        LIB.rtcSetSharedGeometryBuffer(
            geometry,
            1,
            0,
            0x9003,
            self.vertices.ctypes.data,
            0,
            12,
            len(self.vertices),
        )
        LIB.rtcSetSharedGeometryBuffer(
            geometry,
            0,
            0,
            0x5003,
            self.triangles.ctypes.data,
            0,
            12,
            len(self.triangles),
        )
        LIB.rtcCommitGeometry(geometry)
        LIB.rtcAttachGeometry(self.scene, geometry)
        LIB.rtcReleaseGeometry(geometry)
        LIB.rtcCommitScene(self.scene)

    def close(self):
        if getattr(self, "scene", None):
            LIB.rtcReleaseScene(self.scene)
            LIB.rtcReleaseDevice(self.device)
            self.scene = None
            self.device = None

    def __del__(self):
        self.close()

    def intersect(self, origins, directions, tnear, tfar):
        count = len(origins)
        ids = np.full(count, -1, dtype=np.int32)
        distance = np.asarray(tfar, dtype=np.float32).copy()
        u = np.zeros(count, dtype=np.float32)
        v = np.zeros(count, dtype=np.float32)
        normal = np.zeros((count, 3), dtype=np.float32)
        for index in range(count):
            rayhit = RTCRayHit()
            rayhit.ray = RTCRay(
                *origins[index],
                tnear[index],
                *directions[index],
                0.0,
                tfar[index],
                INVALID,
                0,
                0,
            )
            rayhit.hit.geomID = INVALID
            rayhit.hit.primID = INVALID
            rayhit.hit.instID = INVALID
            rayhit.hit.instPrimID = INVALID
            LIB.rtcIntersect1(self.scene, ctypes.byref(rayhit), None)
            if rayhit.hit.geomID != INVALID:
                ids[index] = rayhit.hit.primID
                distance[index] = rayhit.ray.tfar
                u[index] = rayhit.hit.u
                v[index] = rayhit.hit.v
                normal[index] = (
                    rayhit.hit.Ng_x,
                    rayhit.hit.Ng_y,
                    rayhit.hit.Ng_z,
                )
        return ids, distance, u, v, normal

    def occluded(self, origins, directions, tnear, tfar):
        result = np.zeros(len(origins), dtype=bool)
        for index in range(len(origins)):
            ray = RTCRay(
                *origins[index],
                tnear[index],
                *directions[index],
                0.0,
                tfar[index],
                INVALID,
                0,
                0,
            )
            LIB.rtcOccluded1(self.scene, ctypes.byref(ray), None)
            result[index] = ray.tfar < 0.0
        return result
