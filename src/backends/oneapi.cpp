// Intel OneAPI / Level Zero probe. Uses the vendored Level Zero header
// (third_party/level-zero/ze_api.h) for type definitions and constants. The
// runtime is dlopen'd; no link-time vendor SDK is required.

#include "probe.hpp"

#include <ze_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"

namespace gpgpu::backends {

namespace {

// ze_api.h declares the API functions but does not provide
// function-pointer typedefs for them (only callback typedefs for the
// tracing layer). We declare the small subset we resolve via dlopen.
// ZE_APICALL comes from the vendored header so the calling convention
// stays correct on every platform.
using PFN_zeInit                       = ze_result_t (ZE_APICALL*)(ze_init_flags_t);
using PFN_zeDriverGet                  = ze_result_t (ZE_APICALL*)(uint32_t*, ze_driver_handle_t*);
using PFN_zeDriverGetApiVersion        = ze_result_t (ZE_APICALL*)(ze_driver_handle_t, ze_api_version_t*);
using PFN_zeDeviceGet                  = ze_result_t (ZE_APICALL*)(ze_driver_handle_t, uint32_t*, ze_device_handle_t*);
using PFN_zeDeviceGetProperties        = ze_result_t (ZE_APICALL*)(ze_device_handle_t, ze_device_properties_t*);
using PFN_zeDeviceGetComputeProperties = ze_result_t (ZE_APICALL*)(ze_device_handle_t, ze_device_compute_properties_t*);
using PFN_zeDeviceGetMemoryProperties  = ze_result_t (ZE_APICALL*)(ze_device_handle_t, uint32_t*, ze_device_memory_properties_t*);

bool ok(ze_result_t r) { return r == ZE_RESULT_SUCCESS; }

std::string format_api_version(ze_api_version_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%u.%u",
                  ZE_MAJOR_VERSION(v), ZE_MINOR_VERSION(v));
    return buf;
}

std::string format_uuid(const std::uint8_t u[ZE_MAX_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

} // namespace

ProbeResult probe_oneapi() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::OneAPI), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::OneAPI, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto zeInit                = platform::resolve_as<PFN_zeInit>(lib, "zeInit");
    auto zeDriverGet           = platform::resolve_as<PFN_zeDriverGet>(lib, "zeDriverGet");
    auto zeDriverGetApiVersion = platform::resolve_as<PFN_zeDriverGetApiVersion>(lib, "zeDriverGetApiVersion");
    auto zeDeviceGet           = platform::resolve_as<PFN_zeDeviceGet>(lib, "zeDeviceGet");
    auto zeDeviceGetProperties = platform::resolve_as<PFN_zeDeviceGetProperties>(lib, "zeDeviceGetProperties");
    auto zeDeviceGetCompute    = platform::resolve_as<PFN_zeDeviceGetComputeProperties>(lib, "zeDeviceGetComputeProperties");
    auto zeDeviceGetMemory     = platform::resolve_as<PFN_zeDeviceGetMemoryProperties>(lib, "zeDeviceGetMemoryProperties");

    if (!zeInit || !zeDriverGet || !zeDeviceGet || !zeDeviceGetProperties) {
        out.backend = Backend(BackendId::OneAPI, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("required Level Zero symbols missing"));
        return out;
    }

    if (!ok(zeInit(ZE_INIT_FLAG_GPU_ONLY))) {
        if (!ok(zeInit(0))) {
            out.backend = Backend(BackendId::OneAPI, /*version*/ {}, lib_path,
                                  BackendStatus::MissingDriver,
                                  std::string("zeInit failed"));
            return out;
        }
    }

    std::uint32_t n_drivers = 0;
    zeDriverGet(&n_drivers, nullptr);
    if (n_drivers == 0) {
        out.backend = Backend(BackendId::OneAPI, /*version*/ {}, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }
    std::vector<ze_driver_handle_t> drivers(n_drivers);
    zeDriverGet(&n_drivers, drivers.data());

    std::string best_api_version;
    for (auto drv : drivers) {
        if (zeDriverGetApiVersion) {
            ze_api_version_t api = ZE_API_VERSION_1_0;
            if (ok(zeDriverGetApiVersion(drv, &api))) {
                auto s = format_api_version(api);
                if (s > best_api_version) best_api_version = s;
            }
        }

        std::uint32_t n_devs = 0;
        if (!ok(zeDeviceGet(drv, &n_devs, nullptr)) || n_devs == 0) continue;
        std::vector<ze_device_handle_t> devs(n_devs);
        zeDeviceGet(drv, &n_devs, devs.data());

        for (auto h : devs) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (!ok(zeDeviceGetProperties(h, &props))) continue;

            Device d;
            d.set_name(std::string(props.name,
                                   ::strnlen(props.name, ZE_MAX_DEVICE_NAME)));
            Vendor v = classify_vendor_pci(props.vendorId);
            d.set_vendor(v);
            d.set_vendor_string(std::string(to_string(v)));

            if (props.coreClockRate > 0) d.set_frequency(props.coreClockRate);
            d.set_integrated((props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0);

            if (zeDeviceGetCompute) {
                ze_device_compute_properties_t cp{};
                cp.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
                if (ok(zeDeviceGetCompute(h, &cp))) {
                    if (cp.maxTotalGroupSize > 0) d.set_max_workgroup_size(cp.maxTotalGroupSize);
                    if (cp.maxSharedLocalMemory > 0) d.set_shared_memory_per_block(cp.maxSharedLocalMemory);
                    if (cp.numSubGroupSizes > 0)
                        d.set_subgroup_size(cp.subGroupSizes[cp.numSubGroupSizes - 1]);
                }
            }

            if (zeDeviceGetMemory) {
                std::uint32_t n_mem = 0;
                zeDeviceGetMemory(h, &n_mem, nullptr);
                if (n_mem > 0) {
                    std::vector<ze_device_memory_properties_t> mem(n_mem);
                    for (auto& m : mem) m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                    if (ok(zeDeviceGetMemory(h, &n_mem, mem.data()))) {
                        std::uint64_t total = 0;
                        std::uint32_t widest = 0;
                        std::uint32_t fastest = 0;
                        for (const auto& m : mem) {
                            total += m.totalSize;
                            if (m.maxBusWidth > widest) widest = m.maxBusWidth;
                            if (m.maxClockRate > fastest) fastest = m.maxClockRate;
                        }
                        if (total > 0) d.set_memory(total);
                        if (widest > 0) d.set_memory_bus_width_bits(widest);
                        if (widest > 0 && fastest > 0) {
                            std::uint64_t gbps = static_cast<std::uint64_t>(fastest)
                                                 * static_cast<std::uint64_t>(widest) * 2ULL / 8ULL / 1000ULL;
                            if (gbps > 0) d.set_memory_bandwidth_gbps(gbps);
                        }
                    }
                }
            }

            if (props.numSlices && props.numSubslicesPerSlice && props.numEUsPerSubslice) {
                std::uint32_t eus = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;
                d.set_compute_units(eus);
                if (props.physicalEUSimdWidth > 0) {
                    d.set_cores(eus * props.physicalEUSimdWidth);
                }
            }

            if (!best_api_version.empty()) d.set_driver_version(best_api_version);

            d.set_fp16(true);
            d.set_int8(true);

            char id[80];
            std::snprintf(id, sizeof(id), "ze-%04x:%04x-%s", props.vendorId, props.deviceId,
                          format_uuid(props.uuid.id).c_str());
            d.set_id(id);

            out.devices.push_back(std::move(d));
        }
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::OneAPI, std::move(best_api_version), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
