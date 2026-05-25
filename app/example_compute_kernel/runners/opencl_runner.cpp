// OpenCL vector-add runner — single-file, no shared abstraction.
//
// Pipeline:
//   1. dlopen the OpenCL ICD loader at setup.backend.path().
//   2. Enumerate platforms + devices; match by (name, memory) against
//      setup.device.
//   3. Create context + in-order command queue.
//   4. Build the embedded OpenCL C kernel.
//   5. Allocate three cl_mem buffers; upload a, b; launch; download c.
//   6. Verify c == a + b host-side.

#include "opencl_runner.hpp"
#include "lib_loader.hpp"

#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace example {

namespace {

// OpenCL kernel: one work-item per element.
constexpr const char* kKernelSource = R"CL(
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global       float* c,
    const uint n)
{
    uint i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
)CL";

constexpr std::size_t kLocalSize = 256;

using PFN_clGetPlatformIDs    = cl_int (CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
using PFN_clGetDeviceIDs      = cl_int (CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using PFN_clGetDeviceInfo     = cl_int (CL_API_CALL*)(cl_device_id, cl_device_info, std::size_t, void*, std::size_t*);
using PFN_clCreateContext     = cl_context (CL_API_CALL*)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                                          void (CL_CALLBACK*)(const char*, const void*, std::size_t, void*),
                                                          void*, cl_int*);
using PFN_clCreateCommandQueueWithProperties = cl_command_queue (CL_API_CALL*)(cl_context, cl_device_id,
                                                                               const cl_queue_properties*, cl_int*);
using PFN_clCreateBuffer      = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*);
using PFN_clCreateProgramWithSource = cl_program (CL_API_CALL*)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*);
using PFN_clBuildProgram      = cl_int (CL_API_CALL*)(cl_program, cl_uint, const cl_device_id*, const char*,
                                                      void (CL_CALLBACK*)(cl_program, void*), void*);
using PFN_clGetProgramBuildInfo = cl_int (CL_API_CALL*)(cl_program, cl_device_id, cl_program_build_info,
                                                        std::size_t, void*, std::size_t*);
using PFN_clCreateKernel      = cl_kernel (CL_API_CALL*)(cl_program, const char*, cl_int*);
using PFN_clSetKernelArg      = cl_int (CL_API_CALL*)(cl_kernel, cl_uint, std::size_t, const void*);
using PFN_clEnqueueWriteBuffer = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t,
                                                       const void*, cl_uint, const cl_event*, cl_event*);
using PFN_clEnqueueNDRangeKernel = cl_int (CL_API_CALL*)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*,
                                                         const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*);
using PFN_clEnqueueReadBuffer = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t,
                                                      void*, cl_uint, const cl_event*, cl_event*);
using PFN_clFinish            = cl_int (CL_API_CALL*)(cl_command_queue);
using PFN_clReleaseMemObject  = cl_int (CL_API_CALL*)(cl_mem);
using PFN_clReleaseKernel     = cl_int (CL_API_CALL*)(cl_kernel);
using PFN_clReleaseProgram    = cl_int (CL_API_CALL*)(cl_program);
using PFN_clReleaseCommandQueue = cl_int (CL_API_CALL*)(cl_command_queue);
using PFN_clReleaseContext    = cl_int (CL_API_CALL*)(cl_context);

struct ClApi {
    PFN_clGetPlatformIDs                   GetPlatformIDs;
    PFN_clGetDeviceIDs                     GetDeviceIDs;
    PFN_clGetDeviceInfo                    GetDeviceInfo;
    PFN_clCreateContext                    CreateContext;
    PFN_clCreateCommandQueueWithProperties CreateCommandQueueWithProperties;
    PFN_clCreateBuffer                     CreateBuffer;
    PFN_clCreateProgramWithSource          CreateProgramWithSource;
    PFN_clBuildProgram                     BuildProgram;
    PFN_clGetProgramBuildInfo              GetProgramBuildInfo;
    PFN_clCreateKernel                     CreateKernel;
    PFN_clSetKernelArg                     SetKernelArg;
    PFN_clEnqueueWriteBuffer               EnqueueWriteBuffer;
    PFN_clEnqueueNDRangeKernel             EnqueueNDRangeKernel;
    PFN_clEnqueueReadBuffer                EnqueueReadBuffer;
    PFN_clFinish                           Finish;
    PFN_clReleaseMemObject                 ReleaseMemObject;
    PFN_clReleaseKernel                    ReleaseKernel;
    PFN_clReleaseProgram                   ReleaseProgram;
    PFN_clReleaseCommandQueue              ReleaseCommandQueue;
    PFN_clReleaseContext                   ReleaseContext;
};

bool resolve(const loader::Lib& lib, ClApi& api) {
    api.GetPlatformIDs                   = lib.sym<PFN_clGetPlatformIDs>("clGetPlatformIDs");
    api.GetDeviceIDs                     = lib.sym<PFN_clGetDeviceIDs>("clGetDeviceIDs");
    api.GetDeviceInfo                    = lib.sym<PFN_clGetDeviceInfo>("clGetDeviceInfo");
    api.CreateContext                    = lib.sym<PFN_clCreateContext>("clCreateContext");
    api.CreateCommandQueueWithProperties = lib.sym<PFN_clCreateCommandQueueWithProperties>("clCreateCommandQueueWithProperties");
    api.CreateBuffer                     = lib.sym<PFN_clCreateBuffer>("clCreateBuffer");
    api.CreateProgramWithSource          = lib.sym<PFN_clCreateProgramWithSource>("clCreateProgramWithSource");
    api.BuildProgram                     = lib.sym<PFN_clBuildProgram>("clBuildProgram");
    api.GetProgramBuildInfo              = lib.sym<PFN_clGetProgramBuildInfo>("clGetProgramBuildInfo");
    api.CreateKernel                     = lib.sym<PFN_clCreateKernel>("clCreateKernel");
    api.SetKernelArg                     = lib.sym<PFN_clSetKernelArg>("clSetKernelArg");
    api.EnqueueWriteBuffer               = lib.sym<PFN_clEnqueueWriteBuffer>("clEnqueueWriteBuffer");
    api.EnqueueNDRangeKernel             = lib.sym<PFN_clEnqueueNDRangeKernel>("clEnqueueNDRangeKernel");
    api.EnqueueReadBuffer                = lib.sym<PFN_clEnqueueReadBuffer>("clEnqueueReadBuffer");
    api.Finish                           = lib.sym<PFN_clFinish>("clFinish");
    api.ReleaseMemObject                 = lib.sym<PFN_clReleaseMemObject>("clReleaseMemObject");
    api.ReleaseKernel                    = lib.sym<PFN_clReleaseKernel>("clReleaseKernel");
    api.ReleaseProgram                   = lib.sym<PFN_clReleaseProgram>("clReleaseProgram");
    api.ReleaseCommandQueue              = lib.sym<PFN_clReleaseCommandQueue>("clReleaseCommandQueue");
    api.ReleaseContext                   = lib.sym<PFN_clReleaseContext>("clReleaseContext");
    return api.GetPlatformIDs && api.GetDeviceIDs && api.GetDeviceInfo &&
           api.CreateContext && api.CreateCommandQueueWithProperties &&
           api.CreateBuffer && api.CreateProgramWithSource &&
           api.BuildProgram && api.CreateKernel && api.SetKernelArg &&
           api.EnqueueWriteBuffer && api.EnqueueNDRangeKernel &&
           api.EnqueueReadBuffer && api.Finish &&
           api.ReleaseMemObject && api.ReleaseKernel && api.ReleaseProgram &&
           api.ReleaseCommandQueue && api.ReleaseContext;
}

std::string read_info_string(const ClApi& api, cl_device_id dev, cl_device_info key) {
    std::size_t n = 0;
    if (api.GetDeviceInfo(dev, key, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    if (api.GetDeviceInfo(dev, key, n, s.data(), nullptr) != CL_SUCCESS) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
T read_info_scalar(const ClApi& api, cl_device_id dev, cl_device_info key) {
    T v{};
    api.GetDeviceInfo(dev, key, sizeof(T), &v, nullptr);
    return v;
}

// Find the cl_device_id matching the (name, memory) of the gpgpu Device.
cl_device_id find_matching_device(const ClApi& api, const gpgpu::Device& target) {
    cl_uint n_platforms = 0;
    if (api.GetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        return nullptr;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    api.GetPlatformIDs(n_platforms, platforms.data(), nullptr);

    for (cl_platform_id p : platforms) {
        cl_uint n_devs = 0;
        if (api.GetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS) continue;
        if (n_devs == 0) continue;
        std::vector<cl_device_id> devs(n_devs);
        api.GetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);

        for (cl_device_id d : devs) {
            const std::string name = read_info_string(api, d, CL_DEVICE_NAME);
            const cl_ulong   mem   = read_info_scalar<cl_ulong>(api, d, CL_DEVICE_GLOBAL_MEM_SIZE);
            const bool name_ok = (name == target.name());
            const bool mem_ok  = (!target.memory().has_value() || mem == *target.memory());
            if (name_ok && mem_ok) return d;
        }
    }
    return nullptr;
}

std::string program_build_log(const ClApi& api, cl_program program, cl_device_id dev) {
    std::size_t n = 0;
    if (api.GetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n) != CL_SUCCESS || n == 0)
        return {};
    std::string log(n, '\0');
    api.GetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (!log.empty() && log.back() == '\0') log.pop_back();
    return log;
}

} // namespace

RunResult run_vector_add_opencl(const gpgpu::Setup&    setup,
                                std::span<const float> a,
                                std::span<const float> b,
                                std::span<float>       c) {
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);

    loader::Lib lib(setup.backend.path());
    if (!lib.ok()) { r.error = "dlopen(" + setup.backend.path() + ") failed"; return r; }

    ClApi api{};
    if (!resolve(lib, api)) { r.error = "ICD entry point resolution failed"; return r; }

    cl_device_id device = find_matching_device(api, setup.device);
    if (!device) { r.error = "no OpenCL device matched setup.device"; return r; }

    cl_int err = CL_SUCCESS;
    auto t0 = std::chrono::steady_clock::now();

    cl_context context = api.CreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) { r.error = "clCreateContext err=" + std::to_string(err); return r; }

    const cl_queue_properties qprops[] = {0};
    cl_command_queue queue = api.CreateCommandQueueWithProperties(context, device, qprops, &err);
    if (!queue) {
        api.ReleaseContext(context);
        r.error = "clCreateCommandQueueWithProperties err=" + std::to_string(err);
        return r;
    }

    cl_mem buf_a = api.CreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem buf_b = api.CreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem buf_c = api.CreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (!buf_a || !buf_b || !buf_c) {
        if (buf_a) api.ReleaseMemObject(buf_a);
        if (buf_b) api.ReleaseMemObject(buf_b);
        if (buf_c) api.ReleaseMemObject(buf_c);
        api.ReleaseCommandQueue(queue); api.ReleaseContext(context);
        r.error = "clCreateBuffer err=" + std::to_string(err); return r;
    }

    const char*       src     = kKernelSource;
    const std::size_t src_len = std::strlen(kKernelSource);
    cl_program program = api.CreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (!program || err != CL_SUCCESS) {
        api.ReleaseMemObject(buf_a); api.ReleaseMemObject(buf_b); api.ReleaseMemObject(buf_c);
        api.ReleaseCommandQueue(queue); api.ReleaseContext(context);
        r.error = "clCreateProgramWithSource err=" + std::to_string(err); return r;
    }
    if (api.BuildProgram(program, 1, &device, /*opts*/ nullptr, nullptr, nullptr) != CL_SUCCESS) {
        r.error = "clBuildProgram failed: " + program_build_log(api, program, device);
        api.ReleaseProgram(program);
        api.ReleaseMemObject(buf_a); api.ReleaseMemObject(buf_b); api.ReleaseMemObject(buf_c);
        api.ReleaseCommandQueue(queue); api.ReleaseContext(context);
        return r;
    }

    cl_kernel kernel = api.CreateKernel(program, "vector_add", &err);
    if (!kernel) {
        api.ReleaseProgram(program);
        api.ReleaseMemObject(buf_a); api.ReleaseMemObject(buf_b); api.ReleaseMemObject(buf_c);
        api.ReleaseCommandQueue(queue); api.ReleaseContext(context);
        r.error = "clCreateKernel err=" + std::to_string(err); return r;
    }

    api.EnqueueWriteBuffer(queue, buf_a, CL_TRUE, 0, bytes, a.data(), 0, nullptr, nullptr);
    api.EnqueueWriteBuffer(queue, buf_b, CL_TRUE, 0, bytes, b.data(), 0, nullptr, nullptr);

    cl_uint n_arg = static_cast<cl_uint>(n);
    api.SetKernelArg(kernel, 0, sizeof(cl_mem),  &buf_a);
    api.SetKernelArg(kernel, 1, sizeof(cl_mem),  &buf_b);
    api.SetKernelArg(kernel, 2, sizeof(cl_mem),  &buf_c);
    api.SetKernelArg(kernel, 3, sizeof(cl_uint), &n_arg);

    const std::size_t global = ((n + kLocalSize - 1) / kLocalSize) * kLocalSize;
    const std::size_t local  = kLocalSize;
    if (api.EnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr) != CL_SUCCESS) {
        r.error = "clEnqueueNDRangeKernel failed";
    } else {
        api.EnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, bytes, c.data(), 0, nullptr, nullptr);
        api.Finish(queue);
    }

    auto t1 = std::chrono::steady_clock::now();
    r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    api.ReleaseKernel(kernel);
    api.ReleaseProgram(program);
    api.ReleaseMemObject(buf_a);
    api.ReleaseMemObject(buf_b);
    api.ReleaseMemObject(buf_c);
    api.ReleaseCommandQueue(queue);
    api.ReleaseContext(context);

    if (!r.error.empty()) return r;

    // Host-side verification.
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        const float diff     = std::fabs(c[i] - expected);
        if (diff > 1e-4f * std::fabs(expected) + 1e-5f) {
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
