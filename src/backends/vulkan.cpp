// Vulkan compute probe. Uses the vendored Khronos Vulkan-Headers
// (third_party/Vulkan-Headers/) for type definitions and constants. The
// runtime is dlopen'd; CMake defines VK_NO_PROTOTYPES so the header emits
// types only and we resolve every entry point via vkGetInstanceProcAddr.

#include "probe.hpp"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"

namespace gpgpu::backends {

namespace {

std::string format_uuid(const std::uint8_t u[VK_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

std::string format_api_version(std::uint32_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u",
                  VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
    return buf;
}

} // namespace

ProbeResult probe_vulkan() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::Vulkan), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::Vulkan, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto vkGetInstanceProcAddr = platform::resolve_as<PFN_vkGetInstanceProcAddr>(
        lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) {
        out.backend = Backend(BackendId::Vulkan, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("vkGetInstanceProcAddr not exported"));
        return out;
    }

    auto get_global = [&](const char* name) {
        return vkGetInstanceProcAddr(VK_NULL_HANDLE, name);
    };

    auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(get_global("vkCreateInstance"));
    auto vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        get_global("vkEnumerateInstanceVersion"));

    if (!vkCreateInstance) {
        out.backend = Backend(BackendId::Vulkan, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("vkCreateInstance not available"));
        return out;
    }

    std::uint32_t instance_api = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) {
        vkEnumerateInstanceVersion(&instance_api);
    }
    const std::string instance_version_str = format_api_version(instance_api);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gpu_runtime probe";
    app.applicationVersion = 1;
    app.pEngineName = "gpu_runtime";
    app.engineVersion = 1;
    app.apiVersion = instance_api;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &instance);
    if (r != VK_SUCCESS || !instance) {
        out.backend = Backend(BackendId::Vulkan, instance_version_str, lib_path,
                              BackendStatus::InitFailed,
                              "vkCreateInstance returned " + std::to_string(r));
        return out;
    }

    auto resolve_instance = [&](const char* name) {
        return vkGetInstanceProcAddr(instance, name);
    };

    auto vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        resolve_instance("vkDestroyInstance"));
    auto vkEnumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        resolve_instance("vkEnumeratePhysicalDevices"));
    auto vkGetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        resolve_instance("vkGetPhysicalDeviceProperties"));
    auto vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        resolve_instance("vkGetPhysicalDeviceMemoryProperties"));

    if (!vkEnumeratePhysicalDevices || !vkGetPhysicalDeviceProperties) {
        if (vkDestroyInstance) vkDestroyInstance(instance, nullptr);
        out.backend = Backend(BackendId::Vulkan, instance_version_str, lib_path,
                              BackendStatus::InitFailed,
                              std::string("required Vulkan device enumeration symbols missing"));
        return out;
    }

    std::uint32_t n_devs = 0;
    vkEnumeratePhysicalDevices(instance, &n_devs, nullptr);
    if (n_devs == 0) {
        if (vkDestroyInstance) vkDestroyInstance(instance, nullptr);
        out.backend = Backend(BackendId::Vulkan, instance_version_str, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }
    std::vector<VkPhysicalDevice> devs(n_devs);
    vkEnumeratePhysicalDevices(instance, &n_devs, devs.data());

    for (VkPhysicalDevice pd : devs) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);

        Device d;
        d.set_name(std::string(props.deviceName,
                               ::strnlen(props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE)));
        Vendor v = classify_vendor_pci(props.vendorID);
        d.set_vendor(v);
        d.set_vendor_string(std::string(to_string(v)));
        d.set_driver_version(format_api_version(props.driverVersion));
        d.set_integrated(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

        if (vkGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(pd, &mp);
            std::uint64_t device_local = 0;
            for (std::uint32_t i = 0; i < mp.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
                if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    device_local += mp.memoryHeaps[i].size;
                }
            }
            if (device_local > 0) d.set_memory(device_local);
        }

        // Stable id: vendor:device + pipelineCacheUUID (uniquely identifies a
        // driver+device pair across boots).
        char id[80];
        std::snprintf(id, sizeof(id), "vk-%04x:%04x-%s", props.vendorID, props.deviceID,
                      format_uuid(props.pipelineCacheUUID).c_str());
        d.set_id(id);

        out.devices.push_back(std::move(d));
    }

    if (vkDestroyInstance) vkDestroyInstance(instance, nullptr);

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::Vulkan, instance_version_str, lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
