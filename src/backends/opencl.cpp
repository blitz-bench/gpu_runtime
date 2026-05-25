// OpenCL ICD probe. We dlopen libOpenCL.so.1 (OpenCL.dll on Windows) and
// resolve only the ICD entry points we need. No vendor SDK is required at
// build time.

#include "probe.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"
#include "calling_convention.hpp"

namespace gpgpu::backends {

namespace {

using cl_int      = int;
using cl_uint     = unsigned int;
using cl_ulong    = unsigned long long;
using cl_bool     = unsigned int;
using cl_bitfield = unsigned long long;
using cl_platform_id = void*;
using cl_device_id   = void*;
using cl_device_type = cl_bitfield;
using cl_device_info = cl_uint;
using cl_platform_info = cl_uint;

constexpr cl_device_type CL_DEVICE_TYPE_ALL    = 0xFFFFFFFFULL;
constexpr cl_int          CL_SUCCESS            = 0;

constexpr cl_platform_info CL_PLATFORM_NAME      = 0x0902;
constexpr cl_platform_info CL_PLATFORM_VENDOR    = 0x0903;
constexpr cl_platform_info CL_PLATFORM_VERSION   = 0x0901;

constexpr cl_device_info CL_DEVICE_TYPE                    = 0x1000;
constexpr cl_device_info CL_DEVICE_VENDOR_ID               = 0x1001;
constexpr cl_device_info CL_DEVICE_MAX_COMPUTE_UNITS       = 0x1002;
constexpr cl_device_info CL_DEVICE_MAX_WORK_GROUP_SIZE     = 0x1004;
constexpr cl_device_info CL_DEVICE_MAX_CLOCK_FREQUENCY     = 0x100C;
constexpr cl_device_info CL_DEVICE_GLOBAL_MEM_CACHE_SIZE   = 0x101E;
constexpr cl_device_info CL_DEVICE_GLOBAL_MEM_SIZE         = 0x101F;
constexpr cl_device_info CL_DEVICE_LOCAL_MEM_SIZE          = 0x1023;
constexpr cl_device_info CL_DEVICE_NAME                    = 0x102B;
constexpr cl_device_info CL_DEVICE_VENDOR                  = 0x102C;
constexpr cl_device_info CL_DRIVER_VERSION                 = 0x102D;
constexpr cl_device_info CL_DEVICE_VERSION                 = 0x102F;
constexpr cl_device_info CL_DEVICE_EXTENSIONS              = 0x1030;
constexpr cl_device_info CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B;
constexpr cl_device_info CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF   = 0x103C;
constexpr cl_device_info CL_DEVICE_HOST_UNIFIED_MEMORY        = 0x1035; // CL 1.1 only
// NVIDIA + AMD PCI bus extensions.
constexpr cl_device_info CL_DEVICE_PCI_BUS_ID_NV           = 0x4008;
constexpr cl_device_info CL_DEVICE_PCI_SLOT_ID_NV          = 0x4009;
constexpr cl_device_info CL_DEVICE_TOPOLOGY_AMD            = 0x4037;
// KHR PCI bus info (OpenCL 3.0 + cl_khr_pci_bus_info).
constexpr cl_device_info CL_DEVICE_PCI_BUS_INFO_KHR        = 0x410F;

using PFN_clGetPlatformIDs   = cl_int (GPGPU_STDCALL*)(cl_uint, cl_platform_id*, cl_uint*);
using PFN_clGetPlatformInfo  = cl_int (GPGPU_STDCALL*)(cl_platform_id, cl_platform_info, std::size_t, void*, std::size_t*);
using PFN_clGetDeviceIDs     = cl_int (GPGPU_STDCALL*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using PFN_clGetDeviceInfo    = cl_int (GPGPU_STDCALL*)(cl_device_id, cl_device_info, std::size_t, void*, std::size_t*);

// Matches the cl_amd_device_topology union layout (24 bytes total).
// The pcie variant stores bus/device/function as signed chars at fixed
// offsets 21/22/23 after a 17-byte `unused` padding.
struct DeviceTopologyAmd {
    cl_uint type;
    char    unused[17];
    char    bus;
    char    device;
    char    function;
};

struct PciBusInfoKhr {
    cl_uint pci_domain;
    cl_uint pci_bus;
    cl_uint pci_device;
    cl_uint pci_function;
};

std::string info_string(PFN_clGetDeviceInfo fn, cl_device_id dev, cl_device_info key) {
    std::size_t size = 0;
    if (fn(dev, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size, '\0');
    if (fn(dev, key, size, out.data(), nullptr) != CL_SUCCESS) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

template <typename T>
bool info_scalar(PFN_clGetDeviceInfo fn, cl_device_id dev, cl_device_info key, T& out) {
    return fn(dev, key, sizeof(T), &out, nullptr) == CL_SUCCESS;
}

std::string platform_info_string(PFN_clGetPlatformInfo fn, cl_platform_id p, cl_platform_info key) {
    std::size_t size = 0;
    if (fn(p, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size, '\0');
    if (fn(p, key, size, out.data(), nullptr) != CL_SUCCESS) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

std::string format_pci_bdf(unsigned domain, unsigned bus, unsigned dev, unsigned func) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "pci-%04x:%02x:%02x.%x", domain, bus, dev, func);
    return buf;
}

// Trim leading/trailing whitespace.
std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}

} // namespace

ProbeResult probe_opencl() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::OpenCL), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::OpenCL, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto clGetPlatformIDs  = platform::resolve_as<PFN_clGetPlatformIDs>(lib, "clGetPlatformIDs");
    auto clGetPlatformInfo = platform::resolve_as<PFN_clGetPlatformInfo>(lib, "clGetPlatformInfo");
    auto clGetDeviceIDs    = platform::resolve_as<PFN_clGetDeviceIDs>(lib, "clGetDeviceIDs");
    auto clGetDeviceInfo   = platform::resolve_as<PFN_clGetDeviceInfo>(lib, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetDeviceIDs || !clGetDeviceInfo || !clGetPlatformInfo) {
        out.backend = Backend(BackendId::OpenCL, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("required OpenCL ICD symbols missing"));
        return out;
    }

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        out.backend = Backend(BackendId::OpenCL, /*version*/ {}, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }

    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), nullptr);

    std::string best_version; // pick the highest OpenCL version string seen.

    for (cl_platform_id p : platforms) {
        std::string pver = platform_info_string(clGetPlatformInfo, p, CL_PLATFORM_VERSION);
        // CL_PLATFORM_VERSION is shaped "OpenCL <major>.<minor> ...". Extract.
        const std::string prefix = "OpenCL ";
        std::string short_ver;
        if (pver.rfind(prefix, 0) == 0) {
            auto rest = pver.substr(prefix.size());
            auto sp = rest.find(' ');
            short_ver = sp == std::string::npos ? rest : rest.substr(0, sp);
        } else {
            short_ver = pver;
        }
        if (short_ver > best_version) best_version = short_ver;

        cl_uint n_devs = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS || n_devs == 0)
            continue;
        std::vector<cl_device_id> devs(n_devs);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);

        for (cl_device_id dev : devs) {
            Device d;
            d.set_name(trim(info_string(clGetDeviceInfo, dev, CL_DEVICE_NAME)));

            std::string vendor_str = trim(info_string(clGetDeviceInfo, dev, CL_DEVICE_VENDOR));
            d.set_vendor_string(vendor_str);

            cl_uint vendor_id = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_VENDOR_ID, vendor_id);
            Vendor v = classify_vendor_pci(vendor_id);
            if (v == Vendor::Unknown) v = classify_vendor(vendor_str);
            d.set_vendor(v);

            cl_ulong mem = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_GLOBAL_MEM_SIZE, mem) && mem > 0)
                d.set_memory(mem);

            cl_ulong l2 = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, l2) && l2 > 0)
                d.set_l2_cache_bytes(l2);

            cl_ulong local_mem = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_LOCAL_MEM_SIZE, local_mem) && local_mem > 0)
                d.set_shared_memory_per_block(static_cast<std::uint32_t>(local_mem));

            cl_uint cu = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_COMPUTE_UNITS, cu) && cu > 0)
                d.set_compute_units(cu);

            std::size_t max_wg = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, max_wg) && max_wg > 0)
                d.set_max_workgroup_size(static_cast<std::uint32_t>(max_wg));

            cl_uint freq = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, freq) && freq > 0)
                d.set_frequency(freq);

            cl_uint native_half = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, native_half);
            if (native_half > 0) d.set_fp16(true);

            cl_uint native_double = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, native_double);
            if (native_double > 0) d.set_fp64(true);

            cl_bool unified = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_HOST_UNIFIED_MEMORY, unified)) {
                d.set_integrated(unified != 0);
            }

            // Feature flags via extension string.
            std::string exts = info_string(clGetDeviceInfo, dev, CL_DEVICE_EXTENSIONS);
            if (exts.find("cl_khr_fp16") != std::string::npos) d.set_fp16(true);
            if (exts.find("cl_khr_fp64") != std::string::npos) d.set_fp64(true);
            if (exts.find("cl_khr_integer_dot_product") != std::string::npos) d.set_int8(true);

            d.set_driver_version(info_string(clGetDeviceInfo, dev, CL_DRIVER_VERSION));

            // Stable id: try KHR PCI bus info, NVIDIA ext, AMD topology, in that order.
            std::string id_str;
            PciBusInfoKhr khr{};
            if (clGetDeviceInfo(dev, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(khr), &khr, nullptr) == CL_SUCCESS) {
                id_str = format_pci_bdf(khr.pci_domain, khr.pci_bus, khr.pci_device, khr.pci_function);
            } else if (v == Vendor::Nvidia) {
                cl_uint bus = 0, slot = 0;
                bool have_bus = info_scalar(clGetDeviceInfo, dev, CL_DEVICE_PCI_BUS_ID_NV, bus);
                bool have_slot = info_scalar(clGetDeviceInfo, dev, CL_DEVICE_PCI_SLOT_ID_NV, slot);
                if (have_bus && have_slot) {
                    id_str = format_pci_bdf(0, bus, (slot >> 3) & 0x1F, slot & 0x7);
                }
            } else if (v == Vendor::AMD) {
                DeviceTopologyAmd topo{};
                if (clGetDeviceInfo(dev, CL_DEVICE_TOPOLOGY_AMD, sizeof(topo), &topo, nullptr) == CL_SUCCESS) {
                    id_str = format_pci_bdf(0,
                        static_cast<unsigned>(static_cast<unsigned char>(topo.bus)),
                        static_cast<unsigned>(static_cast<unsigned char>(topo.device)),
                        static_cast<unsigned>(static_cast<unsigned char>(topo.function)));
                }
            }
            if (id_str.empty()) {
                // Fallback id: vendor-id + name + uniqueness via pointer.
                char buf[64];
                std::snprintf(buf, sizeof(buf), "ocl-%04x-%p", vendor_id, dev);
                id_str = buf;
            }
            d.set_id(std::move(id_str));

            out.devices.push_back(std::move(d));
        }
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::OpenCL, std::move(best_version), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
