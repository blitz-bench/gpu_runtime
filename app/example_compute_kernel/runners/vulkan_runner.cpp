// Vulkan compute vector-add runner - single-file, no shared abstraction.
//
// Walks the entire compute path: instance -> physical-device pick ->
// logical device with one compute queue -> host-coherent buffers ->
// descriptor set layout / pipeline layout / compute pipeline from SPIR-V ->
// command buffer with push constant + dispatch -> submit + wait -> read
// back -> teardown.
//
// SPIR-V is included from shaders/vector_add.spv.inl (committed binary).

#include "vulkan_runner.hpp"
#include "lib_loader.hpp"

#include <vulkan/vulkan_core.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../shaders/vector_add.spv.inl"

namespace example {

namespace {

constexpr std::uint32_t kLocalSize = 256;

// --- function pointer storage --------------------------------------------

#define VKAPI(name) PFN_vk##name name = nullptr
struct GlobalApi {
    VKAPI(GetInstanceProcAddr);
    VKAPI(CreateInstance);
    VKAPI(EnumerateInstanceVersion);
};
struct InstanceApi {
    VKAPI(DestroyInstance);
    VKAPI(EnumeratePhysicalDevices);
    VKAPI(GetPhysicalDeviceProperties);
    VKAPI(GetPhysicalDeviceMemoryProperties);
    VKAPI(GetPhysicalDeviceQueueFamilyProperties);
    VKAPI(CreateDevice);
    VKAPI(GetDeviceProcAddr);
};
struct DeviceApi {
    VKAPI(DestroyDevice);
    VKAPI(GetDeviceQueue);
    VKAPI(AllocateMemory);
    VKAPI(FreeMemory);
    VKAPI(MapMemory);
    VKAPI(UnmapMemory);
    VKAPI(CreateBuffer);
    VKAPI(DestroyBuffer);
    VKAPI(GetBufferMemoryRequirements);
    VKAPI(BindBufferMemory);
    VKAPI(CreateShaderModule);
    VKAPI(DestroyShaderModule);
    VKAPI(CreateDescriptorSetLayout);
    VKAPI(DestroyDescriptorSetLayout);
    VKAPI(CreatePipelineLayout);
    VKAPI(DestroyPipelineLayout);
    VKAPI(CreateComputePipelines);
    VKAPI(DestroyPipeline);
    VKAPI(CreateDescriptorPool);
    VKAPI(DestroyDescriptorPool);
    VKAPI(AllocateDescriptorSets);
    VKAPI(UpdateDescriptorSets);
    VKAPI(CreateCommandPool);
    VKAPI(DestroyCommandPool);
    VKAPI(AllocateCommandBuffers);
    VKAPI(BeginCommandBuffer);
    VKAPI(EndCommandBuffer);
    VKAPI(CmdBindPipeline);
    VKAPI(CmdBindDescriptorSets);
    VKAPI(CmdPushConstants);
    VKAPI(CmdDispatch);
    VKAPI(QueueSubmit);
    VKAPI(QueueWaitIdle);
};
#undef VKAPI

#define LOAD_GLOBAL(api, name) (api).name = reinterpret_cast<PFN_vk##name>(g.GetInstanceProcAddr(nullptr, "vk" #name))
#define LOAD_INSTANCE(api, name, inst) (api).name = reinterpret_cast<PFN_vk##name>(g.GetInstanceProcAddr(inst, "vk" #name))
#define LOAD_DEVICE(api, name, dev) (api).name = reinterpret_cast<PFN_vk##name>(i.GetDeviceProcAddr(dev, "vk" #name))

bool load_global(loader::Lib& lib, GlobalApi& g) {
    g.GetInstanceProcAddr = lib.sym<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (!g.GetInstanceProcAddr) return false;
    LOAD_GLOBAL(g, CreateInstance);
    LOAD_GLOBAL(g, EnumerateInstanceVersion);
    return g.CreateInstance != nullptr;
}

void load_instance(const GlobalApi& g, VkInstance inst, InstanceApi& a) {
    LOAD_INSTANCE(a, DestroyInstance, inst);
    LOAD_INSTANCE(a, EnumeratePhysicalDevices, inst);
    LOAD_INSTANCE(a, GetPhysicalDeviceProperties, inst);
    LOAD_INSTANCE(a, GetPhysicalDeviceMemoryProperties, inst);
    LOAD_INSTANCE(a, GetPhysicalDeviceQueueFamilyProperties, inst);
    LOAD_INSTANCE(a, CreateDevice, inst);
    LOAD_INSTANCE(a, GetDeviceProcAddr, inst);
}

void load_device(const InstanceApi& i, VkDevice dev, DeviceApi& a) {
    LOAD_DEVICE(a, DestroyDevice, dev);
    LOAD_DEVICE(a, GetDeviceQueue, dev);
    LOAD_DEVICE(a, AllocateMemory, dev);
    LOAD_DEVICE(a, FreeMemory, dev);
    LOAD_DEVICE(a, MapMemory, dev);
    LOAD_DEVICE(a, UnmapMemory, dev);
    LOAD_DEVICE(a, CreateBuffer, dev);
    LOAD_DEVICE(a, DestroyBuffer, dev);
    LOAD_DEVICE(a, GetBufferMemoryRequirements, dev);
    LOAD_DEVICE(a, BindBufferMemory, dev);
    LOAD_DEVICE(a, CreateShaderModule, dev);
    LOAD_DEVICE(a, DestroyShaderModule, dev);
    LOAD_DEVICE(a, CreateDescriptorSetLayout, dev);
    LOAD_DEVICE(a, DestroyDescriptorSetLayout, dev);
    LOAD_DEVICE(a, CreatePipelineLayout, dev);
    LOAD_DEVICE(a, DestroyPipelineLayout, dev);
    LOAD_DEVICE(a, CreateComputePipelines, dev);
    LOAD_DEVICE(a, DestroyPipeline, dev);
    LOAD_DEVICE(a, CreateDescriptorPool, dev);
    LOAD_DEVICE(a, DestroyDescriptorPool, dev);
    LOAD_DEVICE(a, AllocateDescriptorSets, dev);
    LOAD_DEVICE(a, UpdateDescriptorSets, dev);
    LOAD_DEVICE(a, CreateCommandPool, dev);
    LOAD_DEVICE(a, DestroyCommandPool, dev);
    LOAD_DEVICE(a, AllocateCommandBuffers, dev);
    LOAD_DEVICE(a, BeginCommandBuffer, dev);
    LOAD_DEVICE(a, EndCommandBuffer, dev);
    LOAD_DEVICE(a, CmdBindPipeline, dev);
    LOAD_DEVICE(a, CmdBindDescriptorSets, dev);
    LOAD_DEVICE(a, CmdPushConstants, dev);
    LOAD_DEVICE(a, CmdDispatch, dev);
    LOAD_DEVICE(a, QueueSubmit, dev);
    LOAD_DEVICE(a, QueueWaitIdle, dev);
}

#undef LOAD_GLOBAL
#undef LOAD_INSTANCE
#undef LOAD_DEVICE

// --- helpers --------------------------------------------------------------

// Format a stable id matching the gpgpu Vulkan probe:
//   "vk-<vendor>:<device>-<pipelineCacheUUID>"
std::string format_id(const VkPhysicalDeviceProperties& p) {
    const auto& u = p.pipelineCacheUUID;
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "vk-%04x:%04x-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  p.vendorID, p.deviceID,
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

VkPhysicalDevice find_matching_device(const InstanceApi& i, VkInstance inst,
                                      const gpgpu::Device& target) {
    std::uint32_t n = 0;
    i.EnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> phys(n);
    i.EnumeratePhysicalDevices(inst, &n, phys.data());
    for (auto pd : phys) {
        VkPhysicalDeviceProperties props{};
        i.GetPhysicalDeviceProperties(pd, &props);
        if (format_id(props) == target.id()) return pd;
    }
    return VK_NULL_HANDLE;
}

std::uint32_t find_compute_queue_family(const InstanceApi& i, VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    i.GetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    if (n == 0) return UINT32_MAX;
    std::vector<VkQueueFamilyProperties> fams(n);
    i.GetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    // Prefer a queue family that supports compute (graphics queues usually do).
    for (std::uint32_t k = 0; k < n; ++k) {
        if (fams[k].queueFlags & VK_QUEUE_COMPUTE_BIT) return k;
    }
    return UINT32_MAX;
}

std::uint32_t find_host_coherent_memory_type(const VkPhysicalDeviceMemoryProperties& mp,
                                             std::uint32_t allowed_types) {
    const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t k = 0; k < mp.memoryTypeCount; ++k) {
        if ((allowed_types & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & wanted) == wanted) {
            return k;
        }
    }
    return UINT32_MAX;
}

struct BufferAlloc {
    VkBuffer       buf{VK_NULL_HANDLE};
    VkDeviceMemory mem{VK_NULL_HANDLE};
};

bool create_buffer(const DeviceApi& d, VkDevice dev,
                   const VkPhysicalDeviceMemoryProperties& mp,
                   VkDeviceSize bytes, BufferAlloc& out) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.CreateBuffer(dev, &bi, nullptr, &out.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    d.GetBufferMemoryRequirements(dev, out.buf, &req);
    const std::uint32_t mtype = find_host_coherent_memory_type(mp, req.memoryTypeBits);
    if (mtype == UINT32_MAX) return false;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mtype;
    if (d.AllocateMemory(dev, &ai, nullptr, &out.mem) != VK_SUCCESS) return false;
    if (d.BindBufferMemory(dev, out.buf, out.mem, 0) != VK_SUCCESS) return false;
    return true;
}

void destroy_buffer(const DeviceApi& d, VkDevice dev, BufferAlloc& b) {
    if (b.buf) { d.DestroyBuffer(dev, b.buf, nullptr); b.buf = VK_NULL_HANDLE; }
    if (b.mem) { d.FreeMemory(dev, b.mem, nullptr);    b.mem = VK_NULL_HANDLE; }
}

bool upload(const DeviceApi& d, VkDevice dev, VkDeviceMemory mem,
            const void* src, std::size_t bytes) {
    void* mapped = nullptr;
    if (d.MapMemory(dev, mem, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, src, bytes);
    d.UnmapMemory(dev, mem);
    return true;
}

bool download(const DeviceApi& d, VkDevice dev, VkDeviceMemory mem,
              void* dst, std::size_t bytes) {
    void* mapped = nullptr;
    if (d.MapMemory(dev, mem, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(dst, mapped, bytes);
    d.UnmapMemory(dev, mem);
    return true;
}

} // namespace

RunResult run_vector_add_vulkan(const gpgpu::Setup&    setup,
                                std::span<const float> a,
                                std::span<const float> b,
                                std::span<float>       c) {
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    if (kVectorAddSpvBytesLen == 0) {
        r.error = "SPIR-V is empty - regenerate shaders/vector_add.spv.inl";
        return r;
    }
    const std::size_t   n     = a.size();
    const VkDeviceSize  bytes = n * sizeof(float);

    loader::Lib lib(setup.backend.path());
    if (!lib.ok()) { r.error = "dlopen(" + setup.backend.path() + ") failed"; return r; }

    GlobalApi g{};
    if (!load_global(lib, g)) { r.error = "vkGetInstanceProcAddr / vkCreateInstance missing"; return r; }

    std::uint32_t api_version = VK_API_VERSION_1_0;
    if (g.EnumerateInstanceVersion) g.EnumerateInstanceVersion(&api_version);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "example_compute_kernel";
    app.applicationVersion = 1;
    app.pEngineName = "example";
    app.engineVersion = 1;
    app.apiVersion = api_version;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    if (g.CreateInstance(&ici, nullptr, &inst) != VK_SUCCESS || !inst) {
        r.error = "vkCreateInstance failed"; return r;
    }

    InstanceApi i{};
    load_instance(g, inst, i);

    auto t0 = std::chrono::steady_clock::now();

    VkPhysicalDevice pd = find_matching_device(i, inst, setup.device);
    if (!pd) {
        i.DestroyInstance(inst, nullptr);
        r.error = "no Vulkan device matched " + setup.device.id();
        return r;
    }

    const std::uint32_t qfam = find_compute_queue_family(i, pd);
    if (qfam == UINT32_MAX) {
        i.DestroyInstance(inst, nullptr);
        r.error = "no compute queue family";
        return r;
    }

    VkPhysicalDeviceMemoryProperties mp{};
    i.GetPhysicalDeviceMemoryProperties(pd, &mp);

    const float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qprio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;

    VkDevice dev = VK_NULL_HANDLE;
    if (i.CreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS || !dev) {
        i.DestroyInstance(inst, nullptr);
        r.error = "vkCreateDevice failed"; return r;
    }

    DeviceApi d{};
    load_device(i, dev, d);

    VkQueue queue = VK_NULL_HANDLE;
    d.GetDeviceQueue(dev, qfam, 0, &queue);

    // --- buffers ---
    BufferAlloc bA{}, bB{}, bC{};
    bool buf_ok = create_buffer(d, dev, mp, bytes, bA) &&
                  create_buffer(d, dev, mp, bytes, bB) &&
                  create_buffer(d, dev, mp, bytes, bC);
    if (!buf_ok) { r.error = "buffer alloc failed"; goto cleanup_dev; }
    if (!upload(d, dev, bA.mem, a.data(), bytes) ||
        !upload(d, dev, bB.mem, b.data(), bytes)) {
        r.error = "MapMemory upload failed"; goto cleanup_bufs;
    }

    // --- shader module ---
    {
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kVectorAddSpvBytesLen;
    smci.pCode    = reinterpret_cast<const std::uint32_t*>(kVectorAddSpvBytes);
    VkShaderModule shader = VK_NULL_HANDLE;
    if (d.CreateShaderModule(dev, &smci, nullptr, &shader) != VK_SUCCESS) {
        r.error = "vkCreateShaderModule failed";
        goto cleanup_bufs;
    }

    // --- descriptor set layout: 3 storage buffers at bindings 0..2 ---
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (int k = 0; k < 3; ++k) {
        bindings[k].binding         = static_cast<std::uint32_t>(k);
        bindings[k].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[k].descriptorCount = 1;
        bindings[k].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    d.CreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    // --- pipeline layout with one push constant (uint n) ---
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size   = sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    d.CreatePipelineLayout(dev, &plci, nullptr, &pl);

    // --- compute pipeline ---
    VkPipelineShaderStageCreateInfo ssci{};
    ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = shader;
    ssci.pName  = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = ssci;
    cpci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    d.CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);

    // --- descriptor pool + set ---
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    d.CreateDescriptorPool(dev, &dpci, nullptr, &dpool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    d.AllocateDescriptorSets(dev, &dsai, &dset);

    VkDescriptorBufferInfo dbi[3] = {};
    dbi[0].buffer = bA.buf; dbi[0].range = VK_WHOLE_SIZE;
    dbi[1].buffer = bB.buf; dbi[1].range = VK_WHOLE_SIZE;
    dbi[2].buffer = bC.buf; dbi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet writes[3] = {};
    for (int k = 0; k < 3; ++k) {
        writes[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstSet          = dset;
        writes[k].dstBinding      = static_cast<std::uint32_t>(k);
        writes[k].descriptorCount = 1;
        writes[k].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[k].pBufferInfo     = &dbi[k];
    }
    d.UpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    // --- command pool + buffer ---
    VkCommandPoolCreateInfo cpoolci{};
    cpoolci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpoolci.queueFamilyIndex = qfam;
    cpoolci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool cpool = VK_NULL_HANDLE;
    d.CreateCommandPool(dev, &cpoolci, nullptr, &cpool);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = cpool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    d.AllocateCommandBuffers(dev, &cbai, &cb);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d.BeginCommandBuffer(cb, &bi);

    d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    d.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                            0, 1, &dset, 0, nullptr);
    std::uint32_t n_pc = static_cast<std::uint32_t>(n);
    d.CmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n_pc), &n_pc);
    const std::uint32_t groups = static_cast<std::uint32_t>(
        (n + kLocalSize - 1) / kLocalSize);
    d.CmdDispatch(cb, groups, 1, 1);
    d.EndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    if (d.QueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        r.error = "vkQueueSubmit failed";
    } else {
        d.QueueWaitIdle(queue);
        if (!download(d, dev, bC.mem, c.data(), bytes)) {
            r.error = "MapMemory download failed";
        }
    }

    d.DestroyCommandPool(dev, cpool, nullptr);
    d.DestroyDescriptorPool(dev, dpool, nullptr);
    d.DestroyPipeline(dev, pipeline, nullptr);
    d.DestroyPipelineLayout(dev, pl, nullptr);
    d.DestroyDescriptorSetLayout(dev, dsl, nullptr);
    d.DestroyShaderModule(dev, shader, nullptr);
    }

cleanup_bufs:
    destroy_buffer(d, dev, bA);
    destroy_buffer(d, dev, bB);
    destroy_buffer(d, dev, bC);
cleanup_dev:
    if (d.DestroyDevice) d.DestroyDevice(dev, nullptr);
    i.DestroyInstance(inst, nullptr);

    auto t1 = std::chrono::steady_clock::now();
    r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    if (!r.error.empty()) return r;

    for (std::size_t k = 0; k < n; ++k) {
        const float expected = a[k] + b[k];
        if (std::fabs(c[k] - expected) > 1e-4f * std::fabs(expected) + 1e-5f) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "first mismatch at i=%zu: c=%g expected=%g", k, c[k], expected);
            r.error = buf;
            r.correct = false;
            return r;
        }
    }
    r.correct = true;
    return r;
}

} // namespace example
