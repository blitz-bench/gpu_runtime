// CUDA vector-add runner - single-file, no shared abstraction.
//
// Uses the CUDA Driver API exclusively. The kernel ships as inline PTX so
// the runner has zero build-time CUDA Toolkit dependency. NVIDIA's `cuda.h`
// is proprietary and not redistributable, so the small subset of types and
// entry points we need is declared here.

#include "cuda_runner.hpp"
#include "lib_loader.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace example {

namespace {

#if defined(_WIN32)
#define CUDA_STDCALL __stdcall
#else
#define CUDA_STDCALL
#endif

// PTX 7.0 / sm_50 is the lowest CC NVIDIA driver still accepts on modern
// builds. One thread per element, classic structure-of-arrays vector add.
constexpr const char* kVectorAddPtx = R"PTX(
.version 7.0
.target sm_50
.address_size 64

.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1,  [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
)PTX";

constexpr unsigned kBlockSize = 256;

// --- minimal Driver-API surface (license-clean hand-roll) ---

using CUresult       = int;
using CUdevice       = int;
using CUcontext      = void*;
using CUmodule       = void*;
using CUfunction     = void*;
using CUstream       = void*;
using CUdeviceptr    = std::uint64_t;
constexpr CUresult CUDA_SUCCESS = 0;

enum CUdevice_attribute : int {
    CU_ATTR_PCI_BUS_ID    = 33,
    CU_ATTR_PCI_DEVICE_ID = 34,
    CU_ATTR_PCI_DOMAIN_ID = 50,
};

using PFN_cuInit               = CUresult (CUDA_STDCALL*)(unsigned int);
using PFN_cuDeviceGetCount     = CUresult (CUDA_STDCALL*)(int*);
using PFN_cuDeviceGet          = CUresult (CUDA_STDCALL*)(CUdevice*, int);
using PFN_cuDeviceGetAttribute = CUresult (CUDA_STDCALL*)(int*, CUdevice_attribute, CUdevice);
using PFN_cuCtxCreate          = CUresult (CUDA_STDCALL*)(CUcontext*, unsigned int, CUdevice);
using PFN_cuCtxDestroy         = CUresult (CUDA_STDCALL*)(CUcontext);
using PFN_cuCtxSynchronize     = CUresult (CUDA_STDCALL*)(void);
using PFN_cuModuleLoadData     = CUresult (CUDA_STDCALL*)(CUmodule*, const void*);
using PFN_cuModuleUnload       = CUresult (CUDA_STDCALL*)(CUmodule);
using PFN_cuModuleGetFunction  = CUresult (CUDA_STDCALL*)(CUfunction*, CUmodule, const char*);
using PFN_cuMemAlloc           = CUresult (CUDA_STDCALL*)(CUdeviceptr*, std::size_t);
using PFN_cuMemFree            = CUresult (CUDA_STDCALL*)(CUdeviceptr);
using PFN_cuMemcpyHtoD         = CUresult (CUDA_STDCALL*)(CUdeviceptr, const void*, std::size_t);
using PFN_cuMemcpyDtoH         = CUresult (CUDA_STDCALL*)(void*, CUdeviceptr, std::size_t);
using PFN_cuLaunchKernel       = CUresult (CUDA_STDCALL*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                                          unsigned int, unsigned int, unsigned int,
                                                          unsigned int, CUstream,
                                                          void** /*params*/, void** /*extra*/);
using PFN_cuGetErrorString     = CUresult (CUDA_STDCALL*)(CUresult, const char**);

struct CuApi {
    PFN_cuInit               Init;
    PFN_cuDeviceGetCount     DeviceGetCount;
    PFN_cuDeviceGet          DeviceGet;
    PFN_cuDeviceGetAttribute DeviceGetAttribute;
    PFN_cuCtxCreate          CtxCreate;
    PFN_cuCtxDestroy         CtxDestroy;
    PFN_cuCtxSynchronize     CtxSynchronize;
    PFN_cuModuleLoadData     ModuleLoadData;
    PFN_cuModuleUnload       ModuleUnload;
    PFN_cuModuleGetFunction  ModuleGetFunction;
    PFN_cuMemAlloc           MemAlloc;
    PFN_cuMemFree            MemFree;
    PFN_cuMemcpyHtoD         MemcpyHtoD;
    PFN_cuMemcpyDtoH         MemcpyDtoH;
    PFN_cuLaunchKernel       LaunchKernel;
    PFN_cuGetErrorString     GetErrorString;
};

// Prefer the _v2 variants when available - they take size_t byte counts
// and are the current Driver-API entry points on every CUDA 4.0+ driver.
template <typename Fn>
Fn resolve_with_v2(const loader::Lib& lib, const char* base) {
    std::string v2 = std::string(base) + "_v2";
    if (auto f = lib.sym<Fn>(v2.c_str())) return f;
    return lib.sym<Fn>(base);
}

bool resolve(const loader::Lib& lib, CuApi& api) {
    api.Init               = lib.sym<PFN_cuInit>("cuInit");
    api.DeviceGetCount     = lib.sym<PFN_cuDeviceGetCount>("cuDeviceGetCount");
    api.DeviceGet          = lib.sym<PFN_cuDeviceGet>("cuDeviceGet");
    api.DeviceGetAttribute = lib.sym<PFN_cuDeviceGetAttribute>("cuDeviceGetAttribute");
    api.CtxCreate          = resolve_with_v2<PFN_cuCtxCreate>(lib, "cuCtxCreate");
    api.CtxDestroy         = resolve_with_v2<PFN_cuCtxDestroy>(lib, "cuCtxDestroy");
    api.CtxSynchronize     = lib.sym<PFN_cuCtxSynchronize>("cuCtxSynchronize");
    api.ModuleLoadData     = lib.sym<PFN_cuModuleLoadData>("cuModuleLoadData");
    api.ModuleUnload       = lib.sym<PFN_cuModuleUnload>("cuModuleUnload");
    api.ModuleGetFunction  = lib.sym<PFN_cuModuleGetFunction>("cuModuleGetFunction");
    api.MemAlloc           = resolve_with_v2<PFN_cuMemAlloc>(lib, "cuMemAlloc");
    api.MemFree            = resolve_with_v2<PFN_cuMemFree>(lib, "cuMemFree");
    api.MemcpyHtoD         = resolve_with_v2<PFN_cuMemcpyHtoD>(lib, "cuMemcpyHtoD");
    api.MemcpyDtoH         = resolve_with_v2<PFN_cuMemcpyDtoH>(lib, "cuMemcpyDtoH");
    api.LaunchKernel       = lib.sym<PFN_cuLaunchKernel>("cuLaunchKernel");
    api.GetErrorString     = lib.sym<PFN_cuGetErrorString>("cuGetErrorString");
    return api.Init && api.DeviceGetCount && api.DeviceGet && api.DeviceGetAttribute &&
           api.CtxCreate && api.CtxDestroy && api.CtxSynchronize &&
           api.ModuleLoadData && api.ModuleUnload && api.ModuleGetFunction &&
           api.MemAlloc && api.MemFree && api.MemcpyHtoD && api.MemcpyDtoH &&
           api.LaunchKernel;
}

std::string err_str(const CuApi& api, CUresult r) {
    if (api.GetErrorString) {
        const char* s = nullptr;
        if (api.GetErrorString(r, &s) == CUDA_SUCCESS && s) return s;
    }
    return "cuda error " + std::to_string(r);
}

// Build the same id string the gpgpu CUDA probe builds for a given device.
std::string device_bdf(const CuApi& api, CUdevice dev) {
    int dom = 0, bus = 0, devid = 0;
    api.DeviceGetAttribute(&dom,   CU_ATTR_PCI_DOMAIN_ID, dev);
    api.DeviceGetAttribute(&bus,   CU_ATTR_PCI_BUS_ID,    dev);
    api.DeviceGetAttribute(&devid, CU_ATTR_PCI_DEVICE_ID, dev);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "pci-%04x:%02x:%02x.0", dom, bus, devid);
    return buf;
}

CUdevice find_matching_device(const CuApi& api, const gpgpu::Device& target) {
    int n = 0;
    if (api.DeviceGetCount(&n) != CUDA_SUCCESS) return -1;
    for (int i = 0; i < n; ++i) {
        CUdevice d = 0;
        if (api.DeviceGet(&d, i) != CUDA_SUCCESS) continue;
        if (device_bdf(api, d) == target.id()) return d;
    }
    return -1;
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

    loader::Lib lib(setup.backend.path());
    if (!lib.ok()) { r.error = "dlopen(" + setup.backend.path() + ") failed"; return r; }

    CuApi api{};
    if (!resolve(lib, api)) { r.error = "CUDA Driver API symbol resolution failed"; return r; }

    CUresult rc = api.Init(0);
    if (rc != CUDA_SUCCESS) { r.error = "cuInit: " + err_str(api, rc); return r; }

    CUdevice device = find_matching_device(api, setup.device);
    if (device < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }

    auto t0 = std::chrono::steady_clock::now();

    CUcontext ctx = nullptr;
    rc = api.CtxCreate(&ctx, /*flags=*/0, device);
    if (rc != CUDA_SUCCESS) { r.error = "cuCtxCreate: " + err_str(api, rc); return r; }

    CUmodule mod = nullptr;
    rc = api.ModuleLoadData(&mod, kVectorAddPtx);
    if (rc != CUDA_SUCCESS) {
        r.error = "cuModuleLoadData (PTX): " + err_str(api, rc);
        api.CtxDestroy(ctx);
        return r;
    }

    CUfunction kernel = nullptr;
    rc = api.ModuleGetFunction(&kernel, mod, "vector_add");
    if (rc != CUDA_SUCCESS) {
        r.error = "cuModuleGetFunction: " + err_str(api, rc);
        api.ModuleUnload(mod); api.CtxDestroy(ctx);
        return r;
    }

    CUdeviceptr dA = 0, dB = 0, dC = 0;
    api.MemAlloc(&dA, bytes);
    api.MemAlloc(&dB, bytes);
    api.MemAlloc(&dC, bytes);

    api.MemcpyHtoD(dA, a.data(), bytes);
    api.MemcpyHtoD(dB, b.data(), bytes);

    std::uint32_t n_arg = static_cast<std::uint32_t>(n);
    void* params[] = { &dA, &dB, &dC, &n_arg };
    const unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
    rc = api.LaunchKernel(kernel,
                          grid, 1, 1,
                          kBlockSize, 1, 1,
                          /*sharedMemBytes=*/0,
                          /*hStream=*/nullptr,
                          params,
                          /*extra=*/nullptr);
    if (rc != CUDA_SUCCESS) {
        r.error = "cuLaunchKernel: " + err_str(api, rc);
    } else {
        api.CtxSynchronize();
        api.MemcpyDtoH(c.data(), dC, bytes);
    }

    auto t1 = std::chrono::steady_clock::now();
    r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    api.MemFree(dA); api.MemFree(dB); api.MemFree(dC);
    api.ModuleUnload(mod);
    api.CtxDestroy(ctx);

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
