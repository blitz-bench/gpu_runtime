#include "search_paths.hpp"

#include <cstdlib>
#include <string>

namespace gpgpu::search {

namespace {

void push_env_dir(std::vector<std::string>& out, const char* env_var,
                  const char* relative_lib) {
    const char* val = std::getenv(env_var);
    if (!val || !*val) return;
    std::string path(val);
    if (path.back() != '/' && path.back() != '\\') path += '/';
    path += relative_lib;
    out.push_back(std::move(path));
}

void push_env_path_list(std::vector<std::string>& out, const char* env_var,
                        const char* lib_name) {
    const char* raw = std::getenv(env_var);
    if (!raw || !*raw) return;
    const std::string list(raw);
#if defined(_WIN32)
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif
    std::size_t start = 0;
    while (start <= list.size()) {
        std::size_t end = list.find(sep, start);
        if (end == std::string::npos) end = list.size();
        if (end > start) {
            std::string dir = list.substr(start, end - start);
            if (dir.back() != '/' && dir.back() != '\\') dir += '/';
            dir += lib_name;
            out.push_back(std::move(dir));
        }
        if (end == list.size()) break;
        start = end + 1;
    }
}

std::vector<std::string> cuda_candidates() {
    std::vector<std::string> out;
#if defined(_WIN32)
    push_env_dir(out, "CUDA_PATH", "bin\\nvcuda.dll");
    out.emplace_back("nvcuda.dll");
#elif defined(__APPLE__)
    // CUDA on macOS is end-of-life; still try, harmless if missing.
    out.emplace_back("/usr/local/cuda/lib/libcuda.dylib");
    out.emplace_back("libcuda.dylib");
#else
    out.emplace_back("/usr/lib/x86_64-linux-gnu/libcuda.so.1");
    out.emplace_back("/usr/lib64/libcuda.so.1");
    out.emplace_back("/usr/local/cuda/lib64/stubs/libcuda.so");
    out.emplace_back("/opt/cuda/lib64/libcuda.so.1");
    push_env_path_list(out, "LD_LIBRARY_PATH", "libcuda.so.1");
    out.emplace_back("libcuda.so.1");
    out.emplace_back("libcuda.so");
#endif
    return out;
}

std::vector<std::string> rocm_candidates() {
    std::vector<std::string> out;
#if defined(_WIN32)
    push_env_dir(out, "HIP_PATH", "bin\\amdhip64.dll");
    out.emplace_back("amdhip64.dll");
#elif defined(__APPLE__)
    // No ROCm on macOS.
#else
    out.emplace_back("/opt/rocm/lib/libamdhip64.so");
    out.emplace_back("/opt/rocm/hip/lib/libamdhip64.so");
    out.emplace_back("/usr/lib/x86_64-linux-gnu/libamdhip64.so");
    push_env_dir(out, "ROCM_PATH", "lib/libamdhip64.so");
    out.emplace_back("libamdhip64.so");
#endif
    return out;
}

std::vector<std::string> oneapi_candidates() {
    std::vector<std::string> out;
#if defined(_WIN32)
    out.emplace_back("ze_loader.dll");
#elif defined(__APPLE__)
    // Level Zero is not supported on macOS.
#else
    out.emplace_back("/usr/lib/x86_64-linux-gnu/libze_loader.so.1");
    out.emplace_back("/usr/lib64/libze_loader.so.1");
    out.emplace_back("/opt/intel/oneapi/compiler/latest/lib/libze_loader.so.1");
    out.emplace_back("libze_loader.so.1");
    out.emplace_back("libze_loader.so");
#endif
    return out;
}

std::vector<std::string> opencl_candidates() {
    std::vector<std::string> out;
#if defined(_WIN32)
    out.emplace_back("OpenCL.dll");
#elif defined(__APPLE__)
    out.emplace_back("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
    out.emplace_back("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1");
    out.emplace_back("/usr/lib64/libOpenCL.so.1");
    out.emplace_back("/usr/lib/libOpenCL.so.1");
    out.emplace_back("libOpenCL.so.1");
    out.emplace_back("libOpenCL.so");
#endif
    return out;
}

std::vector<std::string> vulkan_candidates() {
    std::vector<std::string> out;
#if defined(_WIN32)
    out.emplace_back("vulkan-1.dll");
#elif defined(__APPLE__)
    out.emplace_back("/usr/local/lib/libvulkan.1.dylib");
    out.emplace_back("/opt/homebrew/lib/libvulkan.1.dylib");
    out.emplace_back("libvulkan.1.dylib");
    out.emplace_back("libMoltenVK.dylib");
#else
    out.emplace_back("/usr/lib/x86_64-linux-gnu/libvulkan.so.1");
    out.emplace_back("/usr/lib64/libvulkan.so.1");
    out.emplace_back("libvulkan.so.1");
    out.emplace_back("libvulkan.so");
#endif
    return out;
}

std::vector<std::string> metal_candidates() {
    std::vector<std::string> out;
#if defined(__APPLE__)
    out.emplace_back("/System/Library/Frameworks/Metal.framework/Metal");
#endif
    return out;
}

} // namespace

std::vector<std::string> candidates(BackendId backend) {
    switch (backend) {
        case BackendId::CUDA: return cuda_candidates();
        case BackendId::ROCm: return rocm_candidates();
        case BackendId::OneAPI: return oneapi_candidates();
        case BackendId::OpenCL: return opencl_candidates();
        case BackendId::Vulkan: return vulkan_candidates();
        case BackendId::Metal: return metal_candidates();
    }
    return {};
}

} // namespace gpgpu::search
