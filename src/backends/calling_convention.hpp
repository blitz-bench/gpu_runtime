#pragma once

// Calling-convention macro for the two backends whose entry points use
// __stdcall on Windows and where we cannot include the vendor headers:
//
//   - CUDA   (cuda.h: CUDAAPI = __stdcall) — NVIDIA license forbids
//            redistribution of the headers, so we hand-roll declarations.
//   - HIP    (hip_runtime_api.h on Windows uses __stdcall via HIPAPI) — the
//            header is MIT-licensed but transitively requires the CUDA / ROCm
//            SDK installation, so we hand-roll a minimal subset.
//
// OpenCL, Vulkan, and Level Zero get the right calling convention directly
// from their vendored headers (CL_API_CALL, VKAPI_PTR, ZE_APICALL).

#if defined(_WIN32)
#  define GPGPU_STDCALL __stdcall
#else
#  define GPGPU_STDCALL
#endif
