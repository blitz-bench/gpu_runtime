// Vulkan compute probe. We dlopen libvulkan.so.1 (vulkan-1.dll), pull entry
// points via vkGetInstanceProcAddr, create a minimal instance and enumerate
// physical devices.
//
// We deliberately avoid Properties2/extensions to keep the implementation
// portable across drivers and platforms.

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

constexpr std::size_t VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
constexpr std::size_t VK_UUID_SIZE = 16;
constexpr std::size_t VK_MAX_MEMORY_TYPES = 32;
constexpr std::size_t VK_MAX_MEMORY_HEAPS = 16;

using VkResult           = int;
using VkInstance         = void*;
using VkPhysicalDevice   = void*;
using VkStructureType    = std::uint32_t;
using VkFlags            = std::uint32_t;
using VkDeviceSize       = std::uint64_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;

constexpr std::uint32_t VK_API_VERSION_1_0 = (1u << 22);

constexpr std::uint32_t VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1;

enum VkPhysicalDeviceType : std::uint32_t {
    VK_PHYSICAL_DEVICE_TYPE_OTHER          = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU            = 4,
};

struct VkApplicationInfo {
    VkStructureType sType;
    const void*     pNext;
    const char*     pApplicationName;
    std::uint32_t   applicationVersion;
    const char*     pEngineName;
    std::uint32_t   engineVersion;
    std::uint32_t   apiVersion;
};

struct VkInstanceCreateInfo {
    VkStructureType            sType;
    const void*                pNext;
    VkFlags                    flags;
    const VkApplicationInfo*   pApplicationInfo;
    std::uint32_t              enabledLayerCount;
    const char* const*         ppEnabledLayerNames;
    std::uint32_t              enabledExtensionCount;
    const char* const*         ppEnabledExtensionNames;
};

// VkPhysicalDeviceLimits is large; we don't need its fields but its size must
// match. From the Vulkan headers it is 504 bytes on every platform we target.
struct VkPhysicalDeviceLimits {
    std::uint8_t opaque[504];
};

struct VkPhysicalDeviceSparseProperties {
    std::uint32_t residencyStandard2DBlockShape;
    std::uint32_t residencyStandard2DMultisampleBlockShape;
    std::uint32_t residencyStandard3DBlockShape;
    std::uint32_t residencyAlignedMipSize;
    std::uint32_t residencyNonResidentStrict;
};

struct VkPhysicalDeviceProperties {
    std::uint32_t                    apiVersion;
    std::uint32_t                    driverVersion;
    std::uint32_t                    vendorID;
    std::uint32_t                    deviceID;
    VkPhysicalDeviceType             deviceType;
    char                             deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    std::uint8_t                     pipelineCacheUUID[VK_UUID_SIZE];
    VkPhysicalDeviceLimits           limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};

struct VkMemoryType {
    VkFlags       propertyFlags;
    std::uint32_t heapIndex;
};

struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags      flags;
};

struct VkPhysicalDeviceMemoryProperties {
    std::uint32_t memoryTypeCount;
    VkMemoryType  memoryTypes[VK_MAX_MEMORY_TYPES];
    std::uint32_t memoryHeapCount;
    VkMemoryHeap  memoryHeaps[VK_MAX_MEMORY_HEAPS];
};

using PFN_vkVoidFunction = void (GPGPU_STDCALL*)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (GPGPU_STDCALL*)(VkInstance, const char*);
using PFN_vkCreateInstance = VkResult (GPGPU_STDCALL*)(const VkInstanceCreateInfo*, const void*, VkInstance*);
using PFN_vkDestroyInstance = void (GPGPU_STDCALL*)(VkInstance, const void*);
using PFN_vkEnumeratePhysicalDevices = VkResult (GPGPU_STDCALL*)(VkInstance, std::uint32_t*, VkPhysicalDevice*);
using PFN_vkGetPhysicalDeviceProperties = void (GPGPU_STDCALL*)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
using PFN_vkGetPhysicalDeviceMemoryProperties = void (GPGPU_STDCALL*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
using PFN_vkEnumerateInstanceVersion = VkResult (GPGPU_STDCALL*)(std::uint32_t*);

std::string format_pci_uuid(const std::uint8_t u[VK_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

std::string format_api_version(std::uint32_t v) {
    // VK_VERSION_MAJOR/MINOR/PATCH macros: major=v>>22, minor=(v>>12)&0x3FF, patch=v&0xFFF
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u",
                  (v >> 22) & 0x7Fu, (v >> 12) & 0x3FFu, v & 0xFFFu);
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
        return vkGetInstanceProcAddr(nullptr, name);
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

    VkInstance instance = nullptr;
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
                      format_pci_uuid(props.pipelineCacheUUID).c_str());
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
