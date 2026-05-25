// ROCm/HIP vector-add runner — SDK-required variant.
//
// Textbook HIP form: __global__ kernel + hipLaunchKernelGGL launch +
// Runtime API for buffers and events. Compiled with hipcc when
// find_package(hip CONFIG) and the HIP language enable succeed.

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace example {

namespace {

constexpr unsigned kBlockSize = 256;

__global__ void vector_add(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int find_matching_device(const gpgpu::Device& target) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) return -1;
    for (int i = 0; i < n; ++i) {
        char bdf[32] = {0};
        if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), i) != hipSuccess) continue;
        if (target.id() == std::string("pci-") + bdf) return i;
    }
    return -1;
}

std::chrono::duration<double> elapsed(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, a, b) != hipSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace

RunResult run_vector_add_rocm(const gpgpu::Setup&    setup,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<float>       c) {
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);
    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    const int dev = find_matching_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    hipMalloc(&dA, bytes);
    hipMalloc(&dB, bytes);
    hipMalloc(&dC, bytes);

    hipEvent_t ev_h2d_begin, ev_h2d_end, ev_k_begin, ev_k_end, ev_d2h_end;
    hipEventCreate(&ev_h2d_begin); hipEventCreate(&ev_h2d_end);
    hipEventCreate(&ev_k_begin);   hipEventCreate(&ev_k_end);
    hipEventCreate(&ev_d2h_end);

    const unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);

    const auto t_total_start = std::chrono::steady_clock::now();

    hipEventRecord(ev_h2d_begin, 0);
    hipMemcpyAsync(dA, a.data(), bytes, hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(dB, b.data(), bytes, hipMemcpyHostToDevice, 0);
    hipEventRecord(ev_h2d_end, 0);

    hipEventRecord(ev_k_begin, 0);
    const auto t_launch_start = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(vector_add, dim3(grid), dim3(kBlockSize), 0, 0,
                       dA, dB, dC, static_cast<unsigned>(n));
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;
    hipEventRecord(ev_k_end, 0);

    hipMemcpyAsync(c.data(), dC, bytes, hipMemcpyDeviceToHost, 0);
    hipEventRecord(ev_d2h_end, 0);
    hipEventSynchronize(ev_d2h_end);

    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        r.error = std::string("HIP error: ") + hipGetErrorString(err);
    }

    r.timings.copy_h2d       = elapsed(ev_h2d_begin, ev_h2d_end);
    r.timings.kernel_compute = elapsed(ev_k_begin,   ev_k_end);
    r.timings.copy_d2h       = elapsed(ev_k_end,     ev_d2h_end);

    hipEventDestroy(ev_h2d_begin); hipEventDestroy(ev_h2d_end);
    hipEventDestroy(ev_k_begin);   hipEventDestroy(ev_k_end);
    hipEventDestroy(ev_d2h_end);

    hipFree(dA); hipFree(dB); hipFree(dC);

    if (!r.error.empty()) return r;

    for (std::size_t i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-4f * std::fabs(expected) + 1e-5f) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "first mismatch at i=%zu: c=%g expected=%g", i, c[i], expected);
            r.error = buf;
            r.correct = false;
            return r;
        }
    }
    r.correct = true;
    return r;
}

} // namespace example
