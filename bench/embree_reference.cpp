// Benchmark adapter for Embree's public C API; not part of the Mojo port.

#include <embree4/rtcore.h>
#include <cstddef>
#include <cstdint>

struct ReferenceScene {
  RTCDevice device;
  RTCScene scene;
};

extern "C" void* reference_create(const float* vertices, std::size_t vertex_count,
                                   const std::uint32_t* triangles,
                                   std::size_t triangle_count) {
  auto* reference = new ReferenceScene;
  reference->device = rtcNewDevice(nullptr);
  reference->scene = rtcNewScene(reference->device);
  RTCGeometry geometry =
      rtcNewGeometry(reference->device, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                             RTC_FORMAT_FLOAT3, vertices, 0, 3 * sizeof(float),
                             vertex_count);
  rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                             triangles, 0, 3 * sizeof(std::uint32_t),
                             triangle_count);
  rtcCommitGeometry(geometry);
  rtcAttachGeometry(reference->scene, geometry);
  rtcReleaseGeometry(geometry);
  rtcCommitScene(reference->scene);
  return reference;
}

extern "C" void reference_destroy(void* handle) {
  auto* reference = static_cast<ReferenceScene*>(handle);
  rtcReleaseScene(reference->scene);
  rtcReleaseDevice(reference->device);
  delete reference;
}

extern "C" void reference_intersect(
    void* handle, const float* origins, const float* directions,
    const float* ray_near, const float* ray_far, std::int32_t* hit_ids,
    float* hit_t, float* hit_u, float* hit_v, float* hit_ng,
    std::size_t ray_count) {
  auto* reference = static_cast<ReferenceScene*>(handle);
  for (std::size_t index = 0; index < ray_count; ++index) {
    RTCRayHit rayhit{};
    rayhit.ray.org_x = origins[3 * index + 0];
    rayhit.ray.org_y = origins[3 * index + 1];
    rayhit.ray.org_z = origins[3 * index + 2];
    rayhit.ray.dir_x = directions[3 * index + 0];
    rayhit.ray.dir_y = directions[3 * index + 1];
    rayhit.ray.dir_z = directions[3 * index + 2];
    rayhit.ray.tnear = ray_near[index];
    rayhit.ray.tfar = ray_far[index];
    rayhit.ray.mask = 0xFFFFFFFFu;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(reference->scene, &rayhit);
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
      hit_ids[index] = -1;
      hit_t[index] = ray_far[index];
      hit_u[index] = 0.0f;
      hit_v[index] = 0.0f;
      hit_ng[3 * index + 0] = 0.0f;
      hit_ng[3 * index + 1] = 0.0f;
      hit_ng[3 * index + 2] = 0.0f;
    } else {
      hit_ids[index] = static_cast<std::int32_t>(rayhit.hit.primID);
      hit_t[index] = rayhit.ray.tfar;
      hit_u[index] = rayhit.hit.u;
      hit_v[index] = rayhit.hit.v;
      hit_ng[3 * index + 0] = rayhit.hit.Ng_x;
      hit_ng[3 * index + 1] = rayhit.hit.Ng_y;
      hit_ng[3 * index + 2] = rayhit.hit.Ng_z;
    }
  }
}

extern "C" void reference_occluded(
    void* handle, const float* origins, const float* directions,
    const float* ray_near, const float* ray_far, std::uint8_t* result,
    std::size_t ray_count) {
  auto* reference = static_cast<ReferenceScene*>(handle);
  for (std::size_t index = 0; index < ray_count; ++index) {
    RTCRay ray{};
    ray.org_x = origins[3 * index + 0];
    ray.org_y = origins[3 * index + 1];
    ray.org_z = origins[3 * index + 2];
    ray.dir_x = directions[3 * index + 0];
    ray.dir_y = directions[3 * index + 1];
    ray.dir_z = directions[3 * index + 2];
    ray.tnear = ray_near[index];
    ray.tfar = ray_far[index];
    ray.mask = 0xFFFFFFFFu;
    rtcOccluded1(reference->scene, &ray);
    result[index] = ray.tfar < 0.0f;
  }
}
