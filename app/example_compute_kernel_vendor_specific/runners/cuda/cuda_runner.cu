// CUDA vector-add runner — SDK-required variant.
//
// Demonstrates the textbook CUDA C++ form: __global__ kernel + <<<>>>
// launch syntax + Runtime API for buffers and events. Compiled with nvcc
// when find_package(CUDAToolkit) succeeds at configure time.

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

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
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return -1;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        char bdf[32];
        std::snprintf(bdf, sizeof(bdf), "pci-%04x:%02x:%02x.0",
                      prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        if (target.id() == bdf) return i;
    }
    return -1;
}

std::chrono::duration<double> elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace

RunResult run_vector_add_cuda(const gpgpu::Setup&    setup,
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
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) {
        r.error = "cudaSetDevice failed"; return r;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaEvent_t ev_h2d_begin, ev_h2d_end, ev_k_begin, ev_k_end, ev_d2h_end;
    cudaEventCreate(&ev_h2d_begin); cudaEventCreate(&ev_h2d_end);
    cudaEventCreate(&ev_k_begin);   cudaEventCreate(&ev_k_end);
    cudaEventCreate(&ev_d2h_end);

    const unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);

    const auto t_total_start = std::chrono::steady_clock::now();

    cudaEventRecord(ev_h2d_begin, 0);
    cudaMemcpyAsync(dA, a.data(), bytes, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(dB, b.data(), bytes, cudaMemcpyHostToDevice, 0);
    cudaEventRecord(ev_h2d_end, 0);

    cudaEventRecord(ev_k_begin, 0);
    const auto t_launch_start = std::chrono::steady_clock::now();
    vector_add<<<grid, kBlockSize>>>(dA, dB, dC, static_cast<unsigned>(n));
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;
    cudaEventRecord(ev_k_end, 0);

    cudaMemcpyAsync(c.data(), dC, bytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(ev_d2h_end, 0);
    cudaEventSynchronize(ev_d2h_end);

    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        r.error = std::string("CUDA error: ") + cudaGetErrorString(err);
    }

    r.timings.copy_h2d       = elapsed(ev_h2d_begin, ev_h2d_end);
    r.timings.kernel_compute = elapsed(ev_k_begin,   ev_k_end);
    r.timings.copy_d2h       = elapsed(ev_k_end,     ev_d2h_end);

    cudaEventDestroy(ev_h2d_begin); cudaEventDestroy(ev_h2d_end);
    cudaEventDestroy(ev_k_begin);   cudaEventDestroy(ev_k_end);
    cudaEventDestroy(ev_d2h_end);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

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
