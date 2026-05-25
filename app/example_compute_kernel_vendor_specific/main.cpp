// example_compute_kernel_vendor_specific — end-to-end usage example for
// gpu_runtime that links against the vendor SDKs at build time.
//
// Sister of app/example_compute_kernel/. Same dispatcher / table /
// host-side verification; the per-backend runners use native SDK
// headers (cuda_runtime.h, hip/hip_runtime.h, vulkan/vulkan.h,
// ze_api.h, CL/cl.h, Metal/Metal.h) instead of dlopen + function-pointer
// resolution.
//
// EXAMPLE_VS_HAVE_<BACKEND> preprocessor defines come from each runner's
// CMake INTERFACE compile_definitions and propagate via target_link_libraries.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gpgpu/runtime.hpp>

#include "result.hpp"

#if defined(EXAMPLE_VS_HAVE_CUDA)
#include "runners/cuda/cuda_runner.hpp"
#endif
#if defined(EXAMPLE_VS_HAVE_ROCM)
#include "runners/rocm/rocm_runner.hpp"
#endif
#if defined(EXAMPLE_VS_HAVE_ONEAPI)
#include "runners/oneapi/oneapi_runner.hpp"
#endif
#if defined(EXAMPLE_VS_HAVE_OPENCL)
#include "runners/opencl/opencl_runner.hpp"
#endif
#if defined(EXAMPLE_VS_HAVE_VULKAN)
#include "runners/vulkan/vulkan_runner.hpp"
#endif
#if defined(EXAMPLE_VS_HAVE_METAL)
#include "runners/metal/metal_runner.hpp"
#endif

namespace {

constexpr std::size_t kElementCount = 1u << 20; // 1 Mi floats per buffer

example::RunResult missing(const char* name) {
    example::RunResult r;
    r.error = std::string(name) + " runner not built into this binary";
    return r;
}

example::RunResult dispatch(const gpgpu::Setup&    setup,
                            std::span<const float> a,
                            std::span<const float> b,
                            std::span<float>       c) {
    switch (setup.backend.id()) {
        case gpgpu::BackendId::CUDA:
#if defined(EXAMPLE_VS_HAVE_CUDA)
            return example::run_vector_add_cuda(setup, a, b, c);
#else
            return missing("CUDA");
#endif
        case gpgpu::BackendId::ROCm:
#if defined(EXAMPLE_VS_HAVE_ROCM)
            return example::run_vector_add_rocm(setup, a, b, c);
#else
            return missing("ROCm");
#endif
        case gpgpu::BackendId::OneAPI:
#if defined(EXAMPLE_VS_HAVE_ONEAPI)
            return example::run_vector_add_oneapi(setup, a, b, c);
#else
            return missing("OneAPI");
#endif
        case gpgpu::BackendId::OpenCL:
#if defined(EXAMPLE_VS_HAVE_OPENCL)
            return example::run_vector_add_opencl(setup, a, b, c);
#else
            return missing("OpenCL");
#endif
        case gpgpu::BackendId::Vulkan:
#if defined(EXAMPLE_VS_HAVE_VULKAN)
            return example::run_vector_add_vulkan(setup, a, b, c);
#else
            return missing("Vulkan");
#endif
        case gpgpu::BackendId::Metal:
#if defined(EXAMPLE_VS_HAVE_METAL)
            return example::run_vector_add_metal(setup, a, b, c);
#else
            return missing("Metal");
#endif
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

void print_build_info() {
    std::printf("Runners compiled into this binary:");
    bool any = false;
    auto note = [&](const char* name, bool yes) {
        if (yes) { std::printf(" %s", name); any = true; }
    };
#if defined(EXAMPLE_VS_HAVE_CUDA)
    note("CUDA",   true);
#endif
#if defined(EXAMPLE_VS_HAVE_ROCM)
    note("ROCm",   true);
#endif
#if defined(EXAMPLE_VS_HAVE_ONEAPI)
    note("OneAPI", true);
#endif
#if defined(EXAMPLE_VS_HAVE_OPENCL)
    note("OpenCL", true);
#endif
#if defined(EXAMPLE_VS_HAVE_VULKAN)
    note("Vulkan", true);
#endif
#if defined(EXAMPLE_VS_HAVE_METAL)
    note("Metal",  true);
#endif
    if (!any) std::printf(" (none)");
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("example_compute_kernel_vendor_specific — vector add (%zu elements)\n",
                kElementCount);
    print_build_info();

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
