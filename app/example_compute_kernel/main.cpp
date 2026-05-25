// example_compute_kernel - end-to-end usage example for gpu_runtime.
//
// Discovers backends + devices via gpu_runtime, then dispatches a small
// vector-addition kernel through whichever per-backend runner matches each
// Setup. Each runner is a standalone reference for its backend - there is
// no shared abstraction.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gpgpu/runtime.hpp>

#include "result.hpp"
#include "runners/cuda_runner.hpp"
#include "runners/metal_runner.hpp"
#include "runners/oneapi_runner.hpp"
#include "runners/opencl_runner.hpp"
#include "runners/rocm_runner.hpp"
#include "runners/vulkan_runner.hpp"

namespace {

constexpr std::size_t kElementCount = 1u << 20; // 1 Mi floats per buffer

example::RunResult dispatch(const gpgpu::Setup&    setup,
                            std::span<const float> a,
                            std::span<const float> b,
                            std::span<float>       c) {
    switch (setup.backend.id()) {
        case gpgpu::BackendId::CUDA:   return example::run_vector_add_cuda(setup, a, b, c);
        case gpgpu::BackendId::ROCm:   return example::run_vector_add_rocm(setup, a, b, c);
        case gpgpu::BackendId::OneAPI: return example::run_vector_add_oneapi(setup, a, b, c);
        case gpgpu::BackendId::OpenCL: return example::run_vector_add_opencl(setup, a, b, c);
        case gpgpu::BackendId::Vulkan: return example::run_vector_add_vulkan(setup, a, b, c);
        case gpgpu::BackendId::Metal:  return example::run_vector_add_metal(setup, a, b, c);
    }
    return example::RunResult{};
}

double to_ms(std::chrono::duration<double> d) { return d.count() * 1000.0; }

double gbps(std::size_t bytes, std::chrono::duration<double> d) {
    if (d.count() <= 0.0 || bytes == 0) return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / d.count();
}

void print_table(const std::vector<example::RunResult>& rows) {
    std::printf("\n== Runs (all durations in ms; bandwidth in GiB/s) ==\n");
    std::printf("  %-44s  %-7s  %-9s  %9s  %9s  %9s  %9s  %9s  %-7s  %s\n",
                "device", "backend", "pref",
                "h2d", "h2d_GBs", "launch", "compute", "d2h", "correct", "error");
    std::printf("  %-44s  %-7s  %-9s  %9s  %9s  %9s  %9s  %9s  %-7s  %s\n",
                std::string(44, '-').c_str(),
                std::string(7, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(9, '-').c_str(),
                std::string(7, '-').c_str(),
                std::string(5, '-').c_str());
    for (const auto& r : rows) {
        std::string label = r.device_id;
        if (!r.device_name.empty()) {
            if (label.size() + 3 + r.device_name.size() > 44) {
                label = r.device_id;
            } else {
                label = r.device_id + " | " + r.device_name;
            }
        }
        if (label.size() > 44) label.resize(44);
        const auto& t = r.timings;
        std::printf("  %-44s  %-7s  %-9s  %9.3f  %9.2f  %9.3f  %9.3f  %9.3f  %-7s  %s\n",
                    label.c_str(),
                    std::string(gpgpu::to_string(r.backend)).c_str(),
                    std::string(gpgpu::to_string(r.preferred)).c_str(),
                    to_ms(t.copy_h2d),
                    gbps(t.copy_h2d_size, t.copy_h2d),
                    to_ms(t.kernel_launch),
                    to_ms(t.kernel_compute),
                    to_ms(t.copy_d2h),
                    r.correct ? "yes" : "no",
                    r.error.empty() ? "" : r.error.c_str());
    }
}

} // namespace

int main() {
    std::printf("example_compute_kernel - vector add (%zu elements) across all "
                "available (device, backend) pairs\n",
                kElementCount);

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(i) * 2.0f;
    }

    auto report = gpgpu::Runtime::query();
    std::printf("Discovered %zu (device, backend) setup(s).\n", report.size());

    std::vector<example::RunResult> results;
    results.reserve(report.size());
    for (const auto& setup : report) {
        std::fill(host_c.begin(), host_c.end(), 0.0f);
        auto r = dispatch(setup, host_a, host_b, host_c);
        r.device_id   = setup.device.id();
        r.device_name = setup.device.name();
        r.backend     = setup.backend.id();
        r.preferred   = setup.preferred;
        results.push_back(std::move(r));
    }

    print_table(results);
    const bool any_ok = std::any_of(results.begin(), results.end(),
                                    [](const auto& r) { return r.correct; });
    return any_ok ? 0 : 1;
}
