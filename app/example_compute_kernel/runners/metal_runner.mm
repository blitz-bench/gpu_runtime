// Metal vector-add runner (macOS only). Uses the linked Metal system
// framework directly — Apple frameworks are not dlopen'd from app code.
//
// metal_runner_stub.cpp provides the run_vector_add_metal symbol on
// non-Apple builds.

#if defined(__APPLE__)

#include "metal_runner.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace example {

namespace {

// Metal Shading Language vector_add: one thread per element.
NSString* const kKernelSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void vector_add(\n"
"    device const float* a [[buffer(0)]],\n"
"    device const float* b [[buffer(1)]],\n"
"    device       float* c [[buffer(2)]],\n"
"    constant uint& n      [[buffer(3)]],\n"
"    uint i                [[thread_position_in_grid]])\n"
"{\n"
"    if (i < n) c[i] = a[i] + b[i];\n"
"}\n";

constexpr NSUInteger kThreadgroup = 256;

id<MTLDevice> find_matching_device(const gpgpu::Device& target) {
    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    if (!devs) return nil;
    for (id<MTLDevice> d in devs) {
        char id_buf[40];
        std::snprintf(id_buf, sizeof(id_buf), "metal-%llx",
                      (unsigned long long)[d registryID]);
        if (std::string(id_buf) == target.id()) return d;
    }
    return devs.count > 0 ? devs[0] : nil;
}

} // namespace

RunResult run_vector_add_metal(const gpgpu::Setup&    setup,
                               std::span<const float> a,
                               std::span<const float> b,
                               std::span<float>       c) {
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);

    @autoreleasepool {
        id<MTLDevice> device = find_matching_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }

        auto t0 = std::chrono::steady_clock::now();

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { r.error = "newCommandQueue failed"; return r; }

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource
                                                  options:nil
                                                    error:&err];
        if (!lib) {
            r.error = std::string("newLibraryWithSource: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"vector_add"];
        if (!fn) { r.error = "newFunctionWithName: vector_add"; return r; }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        id<MTLBuffer> buf_a = [device newBufferWithBytes:a.data()
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [device newBufferWithBytes:b.data()
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (!buf_a || !buf_b || !buf_c) { r.error = "newBufferWith* failed"; return r; }

        std::uint32_t n_arg = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBytes:&n_arg length:sizeof(n_arg) atIndex:3];

        const NSUInteger tg_size = std::min<NSUInteger>(kThreadgroup,
                                                        pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(c.data(), [buf_c contents], bytes);

        auto t1 = std::chrono::steady_clock::now();
        r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    }

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

#endif // __APPLE__
