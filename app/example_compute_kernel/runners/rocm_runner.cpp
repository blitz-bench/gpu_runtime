// HIP / ROCm vector-add runner - single-file, no shared abstraction.
//
// Uses HIP's Driver-style module API for execution and hipRTC for
// runtime kernel compilation. Two libraries are opened:
//   1. libamdhip64.so / amdhip64.dll      (HIP runtime)
//   2. libhiprtc.so   / hiprtc0xxx.dll    (HIP runtime compiler)
//
// The hipRTC library typically sits next to libamdhip64.so in the same
// directory; we infer its path from setup.backend.path().

#include "rocm_runner.hpp"
#include "lib_loader.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace example {

namespace {

#if defined(_WIN32)
#  define HIP_STDCALL
#else
#  define HIP_STDCALL
#endif

// Standard HIP source - same expression as the OpenCL kernel.
constexpr const char* kHipSource = R"HIP(
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
)HIP";

constexpr unsigned kBlockSize = 256;

// --- HIP runtime (libamdhip64) minimal subset ---

using hipError_t   = int;
using hipModule_t  = void*;
using hipFunction_t = void*;
using hipStream_t  = void*;
using hipDeviceptr_t = void*;
using hipEvent_t   = void*;
constexpr hipError_t hipSuccess = 0;

enum hipMemcpyKind : int {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
};

using PFN_hipInit                = hipError_t (HIP_STDCALL*)(unsigned int);
using PFN_hipGetDeviceCount      = hipError_t (HIP_STDCALL*)(int*);
using PFN_hipSetDevice           = hipError_t (HIP_STDCALL*)(int);
using PFN_hipDeviceGetPCIBusId   = hipError_t (HIP_STDCALL*)(char*, int, int);
using PFN_hipMalloc              = hipError_t (HIP_STDCALL*)(void**, std::size_t);
using PFN_hipFree                = hipError_t (HIP_STDCALL*)(void*);
using PFN_hipMemcpy              = hipError_t (HIP_STDCALL*)(void*, const void*, std::size_t, hipMemcpyKind);
using PFN_hipMemcpyAsync         = hipError_t (HIP_STDCALL*)(void*, const void*, std::size_t, hipMemcpyKind, hipStream_t);
using PFN_hipDeviceSynchronize   = hipError_t (HIP_STDCALL*)(void);
using PFN_hipModuleLoadData      = hipError_t (HIP_STDCALL*)(hipModule_t*, const void*);
using PFN_hipModuleUnload        = hipError_t (HIP_STDCALL*)(hipModule_t);
using PFN_hipModuleGetFunction   = hipError_t (HIP_STDCALL*)(hipFunction_t*, hipModule_t, const char*);
using PFN_hipModuleLaunchKernel  = hipError_t (HIP_STDCALL*)(hipFunction_t,
                                                             unsigned, unsigned, unsigned,
                                                             unsigned, unsigned, unsigned,
                                                             unsigned, hipStream_t,
                                                             void**, void**);
using PFN_hipGetErrorString      = const char* (HIP_STDCALL*)(hipError_t);
using PFN_hipEventCreate         = hipError_t (HIP_STDCALL*)(hipEvent_t*);
using PFN_hipEventDestroy        = hipError_t (HIP_STDCALL*)(hipEvent_t);
using PFN_hipEventRecord         = hipError_t (HIP_STDCALL*)(hipEvent_t, hipStream_t);
using PFN_hipEventSynchronize    = hipError_t (HIP_STDCALL*)(hipEvent_t);
using PFN_hipEventElapsedTime    = hipError_t (HIP_STDCALL*)(float* /*ms*/, hipEvent_t, hipEvent_t);

struct HipApi {
    PFN_hipInit               Init;
    PFN_hipGetDeviceCount     GetDeviceCount;
    PFN_hipSetDevice          SetDevice;
    PFN_hipDeviceGetPCIBusId  DeviceGetPCIBusId;
    PFN_hipMalloc             Malloc;
    PFN_hipFree               Free;
    PFN_hipMemcpy             Memcpy;
    PFN_hipMemcpyAsync        MemcpyAsync;
    PFN_hipDeviceSynchronize  DeviceSynchronize;
    PFN_hipModuleLoadData     ModuleLoadData;
    PFN_hipModuleUnload       ModuleUnload;
    PFN_hipModuleGetFunction  ModuleGetFunction;
    PFN_hipModuleLaunchKernel ModuleLaunchKernel;
    PFN_hipGetErrorString     GetErrorString;
    PFN_hipEventCreate        EventCreate;
    PFN_hipEventDestroy       EventDestroy;
    PFN_hipEventRecord        EventRecord;
    PFN_hipEventSynchronize   EventSynchronize;
    PFN_hipEventElapsedTime   EventElapsedTime;
};

bool resolve_hip(const loader::Lib& lib, HipApi& api) {
    api.Init               = lib.sym<PFN_hipInit>("hipInit");
    api.GetDeviceCount     = lib.sym<PFN_hipGetDeviceCount>("hipGetDeviceCount");
    api.SetDevice          = lib.sym<PFN_hipSetDevice>("hipSetDevice");
    api.DeviceGetPCIBusId  = lib.sym<PFN_hipDeviceGetPCIBusId>("hipDeviceGetPCIBusId");
    api.Malloc             = lib.sym<PFN_hipMalloc>("hipMalloc");
    api.Free               = lib.sym<PFN_hipFree>("hipFree");
    api.Memcpy             = lib.sym<PFN_hipMemcpy>("hipMemcpy");
    api.MemcpyAsync        = lib.sym<PFN_hipMemcpyAsync>("hipMemcpyAsync");
    api.DeviceSynchronize  = lib.sym<PFN_hipDeviceSynchronize>("hipDeviceSynchronize");
    api.ModuleLoadData     = lib.sym<PFN_hipModuleLoadData>("hipModuleLoadData");
    api.ModuleUnload       = lib.sym<PFN_hipModuleUnload>("hipModuleUnload");
    api.ModuleGetFunction  = lib.sym<PFN_hipModuleGetFunction>("hipModuleGetFunction");
    api.ModuleLaunchKernel = lib.sym<PFN_hipModuleLaunchKernel>("hipModuleLaunchKernel");
    api.GetErrorString     = lib.sym<PFN_hipGetErrorString>("hipGetErrorString");
    api.EventCreate        = lib.sym<PFN_hipEventCreate>("hipEventCreate");
    api.EventDestroy       = lib.sym<PFN_hipEventDestroy>("hipEventDestroy");
    api.EventRecord        = lib.sym<PFN_hipEventRecord>("hipEventRecord");
    api.EventSynchronize   = lib.sym<PFN_hipEventSynchronize>("hipEventSynchronize");
    api.EventElapsedTime   = lib.sym<PFN_hipEventElapsedTime>("hipEventElapsedTime");
    return api.Init && api.GetDeviceCount && api.SetDevice && api.DeviceGetPCIBusId &&
           api.Malloc && api.Free && api.Memcpy && api.MemcpyAsync &&
           api.DeviceSynchronize &&
           api.ModuleLoadData && api.ModuleUnload && api.ModuleGetFunction &&
           api.ModuleLaunchKernel &&
           api.EventCreate && api.EventDestroy && api.EventRecord &&
           api.EventSynchronize && api.EventElapsedTime;
}

std::string hip_err(const HipApi& api, hipError_t r) {
    if (api.GetErrorString) {
        const char* s = api.GetErrorString(r);
        if (s) return s;
    }
    return "hip error " + std::to_string(r);
}

// --- hipRTC (libhiprtc) minimal subset ---

using hiprtcResult  = int;
using hiprtcProgram = void*;
constexpr hiprtcResult HIPRTC_SUCCESS = 0;

using PFN_hiprtcCreateProgram     = hiprtcResult (HIP_STDCALL*)(hiprtcProgram*, const char*, const char*,
                                                                int, const char**, const char**);
using PFN_hiprtcCompileProgram    = hiprtcResult (HIP_STDCALL*)(hiprtcProgram, int, const char**);
using PFN_hiprtcGetCodeSize       = hiprtcResult (HIP_STDCALL*)(hiprtcProgram, std::size_t*);
using PFN_hiprtcGetCode           = hiprtcResult (HIP_STDCALL*)(hiprtcProgram, char*);
using PFN_hiprtcGetProgramLogSize = hiprtcResult (HIP_STDCALL*)(hiprtcProgram, std::size_t*);
using PFN_hiprtcGetProgramLog     = hiprtcResult (HIP_STDCALL*)(hiprtcProgram, char*);
using PFN_hiprtcDestroyProgram    = hiprtcResult (HIP_STDCALL*)(hiprtcProgram*);
using PFN_hiprtcGetErrorString    = const char* (HIP_STDCALL*)(hiprtcResult);

struct RtcApi {
    PFN_hiprtcCreateProgram     CreateProgram;
    PFN_hiprtcCompileProgram    CompileProgram;
    PFN_hiprtcGetCodeSize       GetCodeSize;
    PFN_hiprtcGetCode           GetCode;
    PFN_hiprtcGetProgramLogSize GetProgramLogSize;
    PFN_hiprtcGetProgramLog     GetProgramLog;
    PFN_hiprtcDestroyProgram    DestroyProgram;
    PFN_hiprtcGetErrorString    GetErrorString;
};

bool resolve_rtc(const loader::Lib& lib, RtcApi& api) {
    api.CreateProgram     = lib.sym<PFN_hiprtcCreateProgram>("hiprtcCreateProgram");
    api.CompileProgram    = lib.sym<PFN_hiprtcCompileProgram>("hiprtcCompileProgram");
    api.GetCodeSize       = lib.sym<PFN_hiprtcGetCodeSize>("hiprtcGetCodeSize");
    api.GetCode           = lib.sym<PFN_hiprtcGetCode>("hiprtcGetCode");
    api.GetProgramLogSize = lib.sym<PFN_hiprtcGetProgramLogSize>("hiprtcGetProgramLogSize");
    api.GetProgramLog     = lib.sym<PFN_hiprtcGetProgramLog>("hiprtcGetProgramLog");
    api.DestroyProgram    = lib.sym<PFN_hiprtcDestroyProgram>("hiprtcDestroyProgram");
    api.GetErrorString    = lib.sym<PFN_hiprtcGetErrorString>("hiprtcGetErrorString");
    return api.CreateProgram && api.CompileProgram && api.GetCodeSize && api.GetCode &&
           api.GetProgramLogSize && api.GetProgramLog && api.DestroyProgram;
}

// Derive libhiprtc.so / hiprtc*.dll path next to libamdhip64.so.
std::string hiprtc_path_from(const std::string& hip_path) {
    auto slash = hip_path.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? std::string{}
                                                   : hip_path.substr(0, slash + 1);
#if defined(_WIN32)
    return dir + "hiprtc0608.dll";   // ROCm 6.x stamps the soversion into the file name
#elif defined(__APPLE__)
    return dir + "libhiprtc.dylib";
#else
    return dir + "libhiprtc.so";
#endif
}

int find_matching_device(const HipApi& api, const gpgpu::Device& target) {
    int n = 0;
    if (api.GetDeviceCount(&n) != hipSuccess) return -1;
    for (int i = 0; i < n; ++i) {
        char bdf[32] = {0};
        if (api.DeviceGetPCIBusId(bdf, sizeof(bdf), i) != hipSuccess) continue;
        if (std::string("pci-") + bdf == target.id()) return i;
    }
    return -1;
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

    loader::Lib hip_lib(setup.backend.path());
    if (!hip_lib.ok()) { r.error = "dlopen(" + setup.backend.path() + ") failed"; return r; }

    HipApi hip{};
    if (!resolve_hip(hip_lib, hip)) { r.error = "HIP symbol resolution failed"; return r; }

    const std::string rtc_path = hiprtc_path_from(setup.backend.path());
    loader::Lib rtc_lib(rtc_path);
    if (!rtc_lib.ok()) { r.error = "dlopen(" + rtc_path + ") failed (hipRTC missing)"; return r; }

    RtcApi rtc{};
    if (!resolve_rtc(rtc_lib, rtc)) { r.error = "hipRTC symbol resolution failed"; return r; }

    hipError_t hrc = hip.Init(0);
    if (hrc != hipSuccess) { r.error = "hipInit: " + hip_err(hip, hrc); return r; }

    int dev = find_matching_device(hip, setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }

    hip.SetDevice(dev);

    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    // Compile HIP source -> code object via hipRTC.
    hiprtcProgram program = nullptr;
    if (rtc.CreateProgram(&program, kHipSource, "vector_add.hip",
                          /*numHeaders=*/0, nullptr, nullptr) != HIPRTC_SUCCESS) {
        r.error = "hiprtcCreateProgram failed"; return r;
    }
    hiprtcResult crc = rtc.CompileProgram(program, 0, nullptr);
    if (crc != HIPRTC_SUCCESS) {
        std::size_t log_size = 0; rtc.GetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) rtc.GetProgramLog(program, log.data());
        r.error = "hiprtcCompileProgram failed: " + log;
        rtc.DestroyProgram(&program);
        return r;
    }

    std::size_t code_size = 0;
    rtc.GetCodeSize(program, &code_size);
    std::vector<char> code(code_size);
    rtc.GetCode(program, code.data());
    rtc.DestroyProgram(&program);

    hipModule_t module = nullptr;
    if (hip.ModuleLoadData(&module, code.data()) != hipSuccess) {
        r.error = "hipModuleLoadData failed"; return r;
    }
    hipFunction_t kernel = nullptr;
    if (hip.ModuleGetFunction(&kernel, module, "vector_add") != hipSuccess) {
        r.error = "hipModuleGetFunction failed";
        hip.ModuleUnload(module);
        return r;
    }

    void* dA = nullptr; void* dB = nullptr; void* dC = nullptr;
    hip.Malloc(&dA, bytes);
    hip.Malloc(&dB, bytes);
    hip.Malloc(&dC, bytes);

    hipEvent_t ev_h2d_begin = nullptr, ev_h2d_end = nullptr;
    hipEvent_t ev_k_begin   = nullptr, ev_k_end   = nullptr;
    hipEvent_t ev_d2h_end   = nullptr;
    hip.EventCreate(&ev_h2d_begin); hip.EventCreate(&ev_h2d_end);
    hip.EventCreate(&ev_k_begin);   hip.EventCreate(&ev_k_end);
    hip.EventCreate(&ev_d2h_end);

    std::uint32_t n_arg = static_cast<std::uint32_t>(n);
    void* params[] = { &dA, &dB, &dC, &n_arg };
    const unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);

    const auto t_total_start = std::chrono::steady_clock::now();

    hip.EventRecord(ev_h2d_begin, nullptr);
    hip.MemcpyAsync(dA, a.data(), bytes, hipMemcpyHostToDevice, nullptr);
    hip.MemcpyAsync(dB, b.data(), bytes, hipMemcpyHostToDevice, nullptr);
    hip.EventRecord(ev_h2d_end, nullptr);

    hip.EventRecord(ev_k_begin, nullptr);
    const auto t_launch_start = std::chrono::steady_clock::now();
    hrc = hip.ModuleLaunchKernel(kernel, grid, 1, 1, kBlockSize, 1, 1,
                                 /*sharedMemBytes=*/0, /*stream=*/nullptr,
                                 params, nullptr);
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;
    hip.EventRecord(ev_k_end, nullptr);

    if (hrc != hipSuccess) {
        r.error = "hipModuleLaunchKernel: " + hip_err(hip, hrc);
    } else {
        hip.MemcpyAsync(c.data(), dC, bytes, hipMemcpyDeviceToHost, nullptr);
        hip.EventRecord(ev_d2h_end, nullptr);
        hip.EventSynchronize(ev_d2h_end);
    }

    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    auto ms_between = [&](hipEvent_t a, hipEvent_t b) -> std::chrono::duration<double> {
        float ms = 0.0f;
        if (hip.EventElapsedTime(&ms, a, b) != hipSuccess || ms < 0) return {};
        return std::chrono::duration<double>{ms / 1000.0};
    };
    r.timings.copy_h2d       = ms_between(ev_h2d_begin, ev_h2d_end);
    r.timings.kernel_compute = ms_between(ev_k_begin,   ev_k_end);
    r.timings.copy_d2h       = ms_between(ev_k_end,     ev_d2h_end);

    hip.EventDestroy(ev_h2d_begin); hip.EventDestroy(ev_h2d_end);
    hip.EventDestroy(ev_k_begin);   hip.EventDestroy(ev_k_end);
    hip.EventDestroy(ev_d2h_end);

    hip.Free(dA); hip.Free(dB); hip.Free(dC);
    hip.ModuleUnload(module);

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
