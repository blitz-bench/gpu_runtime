// CUDA Driver API probe. We dlopen libcuda.so.1 (nvcuda.dll on Windows) and
// resolve only the entry points we need. No vendor SDK is required at build
// time.
//
// Reference: CUDA Driver API stability guarantee — the symbols and the
// `CUdevice_attribute` enum values used below have been ABI-stable since
// CUDA 7. We deliberately stick to long-standing attributes.

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

using CUresult = int;
using CUdevice = int;

enum CUdevice_attribute : int {
    CU_ATTR_MAX_THREADS_PER_BLOCK   = 1,
    CU_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_ATTR_WARP_SIZE               = 10,
    CU_ATTR_CLOCK_RATE              = 13,   // kHz
    CU_ATTR_MULTIPROCESSOR_COUNT    = 16,
    CU_ATTR_INTEGRATED              = 18,
    CU_ATTR_PCI_BUS_ID              = 33,
    CU_ATTR_PCI_DEVICE_ID           = 34,
    CU_ATTR_MEMORY_CLOCK_RATE       = 36,   // kHz
    CU_ATTR_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_ATTR_L2_CACHE_SIZE           = 38,
    CU_ATTR_PCI_DOMAIN_ID           = 50,
    CU_ATTR_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_ATTR_COMPUTE_CAPABILITY_MINOR = 76,
};

using PFN_cuInit             = CUresult (GPGPU_STDCALL*)(unsigned int);
using PFN_cuDriverGetVersion = CUresult (GPGPU_STDCALL*)(int*);
using PFN_cuDeviceGetCount   = CUresult (GPGPU_STDCALL*)(int*);
using PFN_cuDeviceGet        = CUresult (GPGPU_STDCALL*)(CUdevice*, int);
using PFN_cuDeviceGetName    = CUresult (GPGPU_STDCALL*)(char*, int, CUdevice);
using PFN_cuDeviceTotalMem   = CUresult (GPGPU_STDCALL*)(std::size_t*, CUdevice);
using PFN_cuDeviceGetAttribute = CUresult (GPGPU_STDCALL*)(int*, CUdevice_attribute, CUdevice);
using PFN_cuDeviceGetUuid    = CUresult (GPGPU_STDCALL*)(unsigned char[16], CUdevice);

// Approximate "CUDA cores" from compute capability and SM count.
// Table is the well-known one published in the CUDA Programming Guide.
std::uint32_t cores_per_sm(int major, int minor) {
    if (major == 2) return 32;                     // Fermi
    if (major == 3) return 192;                    // Kepler
    if (major == 5) return 128;                    // Maxwell
    if (major == 6) {
        if (minor == 0) return 64;                 // P100
        return 128;                                 // P40/P4/MX-class
    }
    if (major == 7) {
        if (minor == 0 || minor == 2) return 64;   // V100, Xavier
        return 64;                                  // Turing 7.5 — 64 FP32 per SM
    }
    if (major == 8) {
        if (minor == 0) return 64;                 // A100
        if (minor == 6) return 128;                // Ampere consumer
        if (minor == 7) return 128;                // Orin
        if (minor == 9) return 128;                // Ada Lovelace
        return 128;
    }
    if (major == 9) return 128;                    // Hopper
    if (major == 10) return 128;                   // Blackwell-ish (best guess)
    return 0;                                       // unknown
}

std::string format_pci_bdf(int domain, int bus, int dev) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "pci-%04x:%02x:%02x.0", domain, bus, dev);
    return buf;
}

std::string format_uuid(const unsigned char u[16]) {
    char buf[48];
    std::snprintf(buf, sizeof(buf),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

std::string driver_version_string(int v) {
    // CUDA encodes "12.3" as 12030; some old drivers also use 4-digit form.
    int major = v / 1000;
    int minor = (v % 1000) / 10;
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d.%d", major, minor);
    return buf;
}

bool ok(CUresult r) { return r == 0; }

} // namespace

ProbeResult probe_cuda() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::CUDA), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::CUDA, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto cuInit              = platform::resolve_as<PFN_cuInit>(lib, "cuInit");
    auto cuDriverGetVersion  = platform::resolve_as<PFN_cuDriverGetVersion>(lib, "cuDriverGetVersion");
    auto cuDeviceGetCount    = platform::resolve_as<PFN_cuDeviceGetCount>(lib, "cuDeviceGetCount");
    auto cuDeviceGet         = platform::resolve_as<PFN_cuDeviceGet>(lib, "cuDeviceGet");
    auto cuDeviceGetName     = platform::resolve_as<PFN_cuDeviceGetName>(lib, "cuDeviceGetName");
    auto cuDeviceTotalMem    = platform::resolve_as<PFN_cuDeviceTotalMem>(lib, "cuDeviceTotalMem_v2");
    if (!cuDeviceTotalMem) {
        cuDeviceTotalMem = platform::resolve_as<PFN_cuDeviceTotalMem>(lib, "cuDeviceTotalMem");
    }
    auto cuDeviceGetAttribute = platform::resolve_as<PFN_cuDeviceGetAttribute>(lib, "cuDeviceGetAttribute");
    auto cuDeviceGetUuid     = platform::resolve_as<PFN_cuDeviceGetUuid>(lib, "cuDeviceGetUuid");

    if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetAttribute) {
        out.backend = Backend(BackendId::CUDA, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("required CUDA driver symbols missing"));
        return out;
    }

    CUresult init_r = cuInit(0);
    if (!ok(init_r)) {
        out.backend = Backend(BackendId::CUDA, /*version*/ {}, lib_path,
                              BackendStatus::MissingDriver,
                              "cuInit returned " + std::to_string(init_r));
        return out;
    }

    std::string version_str;
    if (cuDriverGetVersion) {
        int v = 0;
        if (ok(cuDriverGetVersion(&v))) version_str = driver_version_string(v);
    }

    int count = 0;
    if (!ok(cuDeviceGetCount(&count)) || count <= 0) {
        out.backend = Backend(BackendId::CUDA, version_str, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }

    for (int i = 0; i < count; ++i) {
        CUdevice dev = 0;
        if (!ok(cuDeviceGet(&dev, i))) continue;

        Device d;
        d.set_vendor(Vendor::Nvidia).set_vendor_string("NVIDIA Corporation");

        char name_buf[256] = {0};
        if (cuDeviceGetName && ok(cuDeviceGetName(name_buf, sizeof(name_buf), dev))) {
            d.set_name(name_buf);
        }

        if (cuDeviceTotalMem) {
            std::size_t bytes = 0;
            if (ok(cuDeviceTotalMem(&bytes, dev))) d.set_memory(bytes);
        }

        int sm_count = 0, cc_major = 0, cc_minor = 0;
        cuDeviceGetAttribute(&sm_count, CU_ATTR_MULTIPROCESSOR_COUNT, dev);
        cuDeviceGetAttribute(&cc_major, CU_ATTR_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&cc_minor, CU_ATTR_COMPUTE_CAPABILITY_MINOR, dev);
        if (sm_count > 0) d.set_compute_units(static_cast<std::uint32_t>(sm_count));
        if (sm_count > 0 && cc_major > 0) {
            auto per_sm = cores_per_sm(cc_major, cc_minor);
            if (per_sm > 0) d.set_cores(static_cast<std::uint32_t>(sm_count) * per_sm);
        }

        int warp = 0; if (ok(cuDeviceGetAttribute(&warp, CU_ATTR_WARP_SIZE, dev)) && warp > 0)
            d.set_subgroup_size(static_cast<std::uint32_t>(warp));
        int max_thread = 0; if (ok(cuDeviceGetAttribute(&max_thread, CU_ATTR_MAX_THREADS_PER_BLOCK, dev)) && max_thread > 0)
            d.set_max_workgroup_size(static_cast<std::uint32_t>(max_thread));
        int shared_mem = 0; if (ok(cuDeviceGetAttribute(&shared_mem, CU_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, dev)) && shared_mem > 0)
            d.set_shared_memory_per_block(static_cast<std::uint32_t>(shared_mem));

        int clock_khz = 0; if (ok(cuDeviceGetAttribute(&clock_khz, CU_ATTR_CLOCK_RATE, dev)) && clock_khz > 0)
            d.set_frequency(static_cast<std::uint32_t>(clock_khz / 1000));

        int bus_w = 0; if (ok(cuDeviceGetAttribute(&bus_w, CU_ATTR_GLOBAL_MEMORY_BUS_WIDTH, dev)) && bus_w > 0)
            d.set_memory_bus_width_bits(static_cast<std::uint32_t>(bus_w));
        int l2 = 0; if (ok(cuDeviceGetAttribute(&l2, CU_ATTR_L2_CACHE_SIZE, dev)) && l2 > 0)
            d.set_l2_cache_bytes(static_cast<std::uint64_t>(l2));

        int mem_clock_khz = 0;
        cuDeviceGetAttribute(&mem_clock_khz, CU_ATTR_MEMORY_CLOCK_RATE, dev);
        if (mem_clock_khz > 0 && bus_w > 0) {
            // GDDR/HBM transfer: clock (kHz) * bus_width (bits) * 2 (DDR) / 8 (B/b) / 1e6 -> GB/s.
            std::uint64_t gbps = static_cast<std::uint64_t>(mem_clock_khz)
                                 * static_cast<std::uint64_t>(bus_w) * 2ULL / 8ULL / 1000000ULL;
            if (gbps > 0) d.set_memory_bandwidth_gbps(gbps);
        }

        int integrated = 0;
        if (ok(cuDeviceGetAttribute(&integrated, CU_ATTR_INTEGRATED, dev)))
            d.set_integrated(integrated != 0);

        // Feature flags from compute capability.
        if (cc_major >= 6) d.set_fp16(true);   // sm_60+ has native FP16
        if (cc_major >= 1) d.set_fp64(true);   // CUDA always exposes FP64 (slow on consumer)
        if ((cc_major == 6 && cc_minor == 1) || cc_major >= 7) d.set_int8(true);
        if (cc_major >= 7) d.set_tensor_cores(true);

        if (!version_str.empty()) d.set_driver_version(version_str);

        // Stable cross-backend ID: prefer PCI BDF, fall back to UUID.
        int dom = 0, bus = 0, devid = 0;
        bool have_dom = ok(cuDeviceGetAttribute(&dom, CU_ATTR_PCI_DOMAIN_ID, dev));
        bool have_bus = ok(cuDeviceGetAttribute(&bus, CU_ATTR_PCI_BUS_ID, dev));
        bool have_dev = ok(cuDeviceGetAttribute(&devid, CU_ATTR_PCI_DEVICE_ID, dev));
        if (have_bus && have_dev) {
            d.set_id(format_pci_bdf(have_dom ? dom : 0, bus, devid));
        } else if (cuDeviceGetUuid) {
            unsigned char uuid[16] = {0};
            if (ok(cuDeviceGetUuid(uuid, dev))) d.set_id(format_uuid(uuid));
        }
        if (d.id().empty()) d.set_id("cuda-" + std::to_string(i));

        out.devices.push_back(std::move(d));
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::CUDA, std::move(version_str), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
