#pragma once

#include <chrono>
#include <cstddef>

namespace example {

// Per-phase timing of a single runner invocation.
//
// All durations are seconds expressed as double — std::chrono::duration<double>
// is std::chrono::duration<double, std::ratio<1>>. Convert to milliseconds at
// the print site with `.count() * 1000.0`.
//
// Where the backend exposes device-side profiling events (OpenCL events,
// CUDA / HIP events, Vulkan timestamp queries, Level Zero kernel-timestamp
// events, Metal MTLCommandBuffer GPU{Start,End}Time), kernel_compute and the
// device-side copies are measured by the device clock. Where the upload /
// download is a host memcpy into shared / mapped memory (Vulkan host-coherent,
// Metal shared, Level Zero shared USM), the copy duration is host wall-time
// around the memcpy.
//
// total brackets the full upload → launch → download → sync sequence on the
// host clock, excluding context / queue / module / kernel creation and final
// teardown.
struct Timings {
    std::chrono::duration<double> copy_h2d{0};
    std::size_t                   copy_h2d_size{0};
    std::chrono::duration<double> kernel_launch{0};   // host: time spent in the API launch call
    std::chrono::duration<double> kernel_compute{0};  // device: GPU-reported kernel execution time
    std::chrono::duration<double> copy_d2h{0};
    std::size_t                   copy_d2h_size{0};
    std::chrono::duration<double> total{0};
};

} // namespace example
