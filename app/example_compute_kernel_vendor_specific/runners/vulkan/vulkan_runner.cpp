// Vulkan compute vector-add runner — SDK-required variant.
//
// Direct calls into <vulkan/vulkan.h>. The shared shader build emits
// vector_add.spv.inl, which we include for the SPIR-V bytes.

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include "vector_add.spv.inl"  // k_vector_add_spv_bytes / _len

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace example {

namespace {

constexpr std::uint32_t kLocalSize = 256;

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

VkPhysicalDevice find_matching_device(VkInstance inst, const gpgpu::Device& target) {
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(inst, &n, phys.data());
    for (auto pd : phys) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (format_id(props) == target.id()) return pd;
    }
    return VK_NULL_HANDLE;
}

std::uint32_t find_compute_queue_family(VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    if (n == 0) return UINT32_MAX;
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    for (std::uint32_t k = 0; k < n; ++k) {
        if (fams[k].queueFlags & VK_QUEUE_COMPUTE_BIT) return k;
    }
    return UINT32_MAX;
}

std::uint32_t find_host_coherent_memory_type(const VkPhysicalDeviceMemoryProperties& mp,
                                             std::uint32_t allowed) {
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t k = 0; k < mp.memoryTypeCount; ++k) {
        if ((allowed & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & wanted) == wanted) return k;
    }
    return UINT32_MAX;
}

struct Buffer {
    VkBuffer       buf{VK_NULL_HANDLE};
    VkDeviceMemory mem{VK_NULL_HANDLE};
};

bool make_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp,
                 VkDeviceSize bytes, Buffer& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &out.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, out.buf, &req);
    const std::uint32_t mt = find_host_coherent_memory_type(mp, req.memoryTypeBits);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(dev, &ai, nullptr, &out.mem) != VK_SUCCESS) return false;
    return vkBindBufferMemory(dev, out.buf, out.mem, 0) == VK_SUCCESS;
}

void free_buffer(VkDevice dev, Buffer& b) {
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
    b = {};
}

} // namespace

RunResult run_vector_add_vulkan(const gpgpu::Setup&    setup,
                                std::span<const float> a,
                                std::span<const float> b,
                                std::span<float>       c) {
    using namespace example_vs_vulkan_shader;

    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t  n     = a.size();
    const VkDeviceSize bytes = n * sizeof(float);
    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "example_compute_kernel_vendor_specific";
    app.apiVersion       = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        r.error = "vkCreateInstance failed"; return r;
    }

    VkPhysicalDevice pd = find_matching_device(inst, setup.device);
    if (!pd) { vkDestroyInstance(inst, nullptr); r.error = "no Vulkan device matched " + setup.device.id(); return r; }

    VkPhysicalDeviceProperties pd_props{};
    vkGetPhysicalDeviceProperties(pd, &pd_props);
    const float ts_period_ns = pd_props.limits.timestampPeriod;
    const bool  ts_ok        = pd_props.limits.timestampComputeAndGraphics != 0;

    const std::uint32_t qfam = find_compute_queue_family(pd);
    if (qfam == UINT32_MAX) { vkDestroyInstance(inst, nullptr); r.error = "no compute queue family"; return r; }

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) {
        vkDestroyInstance(inst, nullptr); r.error = "vkCreateDevice failed"; return r;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    Buffer bA{}, bB{}, bC{};
    if (!make_buffer(dev, mp, bytes, bA) ||
        !make_buffer(dev, mp, bytes, bB) ||
        !make_buffer(dev, mp, bytes, bC)) {
        r.error = "buffer alloc failed";
        free_buffer(dev, bA); free_buffer(dev, bB); free_buffer(dev, bC);
        vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr); return r;
    }

    // Pipeline.
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = k_vector_add_spv_bytes_len;
    smci.pCode    = reinterpret_cast<const std::uint32_t*>(k_vector_add_spv_bytes);
    VkShaderModule shader = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &smci, nullptr, &shader);

    VkDescriptorSetLayoutBinding binds[3] = {};
    for (int k = 0; k < 3; ++k) {
        binds[k].binding         = std::uint32_t(k);
        binds[k].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[k].descriptorCount = 1;
        binds[k].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = shader;
    ssci.pName  = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci; cpci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &dset);

    VkDescriptorBufferInfo dbi[3] = {
        {bA.buf, 0, VK_WHOLE_SIZE},
        {bB.buf, 0, VK_WHOLE_SIZE},
        {bC.buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3] = {};
    for (int k = 0; k < 3; ++k) {
        writes[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstSet          = dset;
        writes[k].dstBinding      = std::uint32_t(k);
        writes[k].descriptorCount = 1;
        writes[k].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[k].pBufferInfo     = &dbi[k];
    }
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    // Timestamp query pool for kernel_compute.
    VkQueryPool qpool = VK_NULL_HANDLE;
    if (ts_ok) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
        vkCreateQueryPool(dev, &qpci, nullptr, &qpool);
    }

    VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpoolci.queueFamilyIndex = qfam;
    cpoolci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool cpool = VK_NULL_HANDLE;
    vkCreateCommandPool(dev, &cpoolci, nullptr, &cpool);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev, &cbai, &cb);

    const auto t_total_start = std::chrono::steady_clock::now();

    // Host-coherent upload via map+memcpy.
    {
        const auto t_h2d_start = std::chrono::steady_clock::now();
        void* mapped = nullptr;
        vkMapMemory(dev, bA.mem, 0, bytes, 0, &mapped); std::memcpy(mapped, a.data(), bytes); vkUnmapMemory(dev, bA.mem);
        vkMapMemory(dev, bB.mem, 0, bytes, 0, &mapped); std::memcpy(mapped, b.data(), bytes); vkUnmapMemory(dev, bB.mem);
        r.timings.copy_h2d = std::chrono::steady_clock::now() - t_h2d_start;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
        vkCmdResetQueryPool(cb, qpool, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    std::uint32_t n_pc = static_cast<std::uint32_t>(n);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n_pc), &n_pc);
    const std::uint32_t groups = static_cast<std::uint32_t>((n + kLocalSize - 1) / kLocalSize);
    vkCmdDispatch(cb, groups, 1, 1);
    if (qpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;

    const auto t_launch_start = std::chrono::steady_clock::now();
    VkResult sub_rc = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;

    if (sub_rc != VK_SUCCESS) {
        r.error = "vkQueueSubmit failed";
    } else {
        vkQueueWaitIdle(queue);
        if (qpool) {
            std::uint64_t ts[2] = {0, 0};
            if (vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(std::uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS
                && ts[1] > ts[0]) {
                r.timings.kernel_compute = std::chrono::duration<double>{
                    (ts[1] - ts[0]) * ts_period_ns / 1.0e9};
            }
        }
        const auto t_d2h_start = std::chrono::steady_clock::now();
        void* mapped = nullptr;
        vkMapMemory(dev, bC.mem, 0, bytes, 0, &mapped);
        std::memcpy(c.data(), mapped, bytes);
        vkUnmapMemory(dev, bC.mem);
        r.timings.copy_d2h = std::chrono::steady_clock::now() - t_d2h_start;
    }

    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    free_buffer(dev, bA); free_buffer(dev, bB); free_buffer(dev, bC);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);

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
