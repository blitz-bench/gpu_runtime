#pragma once

// Calling-convention macros for backends whose runtime libraries declare their
// entry points with __stdcall on Windows. Getting this wrong on 32-bit Windows
// corrupts the call stack and crashes the process; on x64 Windows the two
// conventions happen to be identical but the typedefs must still match the
// header declarations.
//
// Per the official headers:
//   - cl.h:      CL_API_CALL     = __stdcall  (Windows)
//   - cuda.h:    CUDAAPI         = __stdcall  (Windows)
//   - vk_platform.h: VKAPI_CALL  = __stdcall  (Windows)
//
// HIP and Level Zero stay on the default __cdecl on Windows.

#if defined(_WIN32)
#  define GPGPU_STDCALL __stdcall
#else
#  define GPGPU_STDCALL
#endif
