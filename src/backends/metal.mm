// Metal probe (macOS only). Enumerates all Metal-capable GPUs and fills in
// device characteristics that the Metal API exposes.

#if defined(__APPLE__)

#include "probe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace gpgpu::backends {

namespace {

std::string nsstr(NSString* s) {
    if (!s) return {};
    const char* c = [s UTF8String];
    return c ? std::string(c) : std::string();
}

Vendor classify_metal_device(id<MTLDevice> d) {
    std::string n = nsstr([d name]);
    Vendor v = classify_vendor(n);
    if (v != Vendor::Unknown) return v;
    // Apple-silicon GPUs always identify as Apple.
    if ([d respondsToSelector:@selector(supportsFamily:)] &&
        [d supportsFamily:MTLGPUFamilyApple1]) {
        return Vendor::Apple;
    }
    return Vendor::Unknown;
}

} // namespace

ProbeResult probe_metal() {
    ProbeResult out;

    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    if (!devs || [devs count] == 0) {
        out.backend = Backend(BackendId::Metal, /*version*/ {},
                              "/System/Library/Frameworks/Metal.framework/Metal",
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }

    NSOperatingSystemVersion ver = [[NSProcessInfo processInfo] operatingSystemVersion];
    char version_buf[32];
    std::snprintf(version_buf, sizeof(version_buf), "%ld.%ld.%ld",
                  (long)ver.majorVersion, (long)ver.minorVersion, (long)ver.patchVersion);

    for (id<MTLDevice> dev in devs) {
        Device d;
        d.set_name(nsstr([dev name]));
        Vendor v = classify_metal_device(dev);
        d.set_vendor(v);
        d.set_vendor_string(std::string(to_string(v)));

        if ([dev respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            std::uint64_t mem = (std::uint64_t)[dev recommendedMaxWorkingSetSize];
            if (mem > 0) d.set_memory(mem);
        }
        if ([dev respondsToSelector:@selector(hasUnifiedMemory)]) {
            d.set_integrated([dev hasUnifiedMemory] ? true : false);
        }
        if ([dev respondsToSelector:@selector(maxThreadsPerThreadgroup)]) {
            MTLSize tg = [dev maxThreadsPerThreadgroup];
            std::uint64_t total = (std::uint64_t)tg.width * tg.height * tg.depth;
            if (total > 0) d.set_max_workgroup_size((std::uint32_t)total);
        }
        if ([dev respondsToSelector:@selector(maxThreadgroupMemoryLength)]) {
            std::uint64_t sm = (std::uint64_t)[dev maxThreadgroupMemoryLength];
            if (sm > 0) d.set_shared_memory_per_block((std::uint32_t)sm);
        }

        // Metal exposes a stable registryID per device.
        if ([dev respondsToSelector:@selector(registryID)]) {
            char id_buf[40];
            std::snprintf(id_buf, sizeof(id_buf), "metal-%llx",
                          (unsigned long long)[dev registryID]);
            d.set_id(id_buf);
        }
        if (d.id().empty()) d.set_id("metal-" + std::to_string(out.devices.size()));

        d.set_driver_version(version_buf);

        d.set_fp16(true);
        d.set_fp64(false); // Apple GPUs don't expose hardware FP64.
        d.set_int8(true);

        out.devices.push_back(std::move(d));
    }

    out.backend = Backend(BackendId::Metal, std::string(version_buf),
                          "/System/Library/Frameworks/Metal.framework/Metal",
                          BackendStatus::Available, std::nullopt);
    return out;
}

} // namespace gpgpu::backends

#endif // __APPLE__
