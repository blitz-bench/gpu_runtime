// Level Zero / oneAPI vector-add runner — SDK-required variant.
//
// Direct calls into <ze_api.h>; linked against ze_loader via pkg-config or
// find_library. SPIR-V is consumed from the shared build-generated
// vector_add.spv.inl (compiled by the shaders/ subdir at build time).

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include "vector_add.spv.inl"  // shared shader build; defines k_vector_add_spv_bytes / _len

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace example {

namespace {

constexpr std::uint32_t kGroupSize = 256;

std::string format_uuid(const std::uint8_t u[ZE_MAX_DEVICE_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

bool find_matching_device(const gpgpu::Device& target,
                          ze_driver_handle_t&  out_driver,
                          ze_device_handle_t&  out_device) {
    std::uint32_t n_drv = 0;
    zeDriverGet(&n_drv, nullptr);
    if (n_drv == 0) return false;
    std::vector<ze_driver_handle_t> drivers(n_drv);
    zeDriverGet(&n_drv, drivers.data());
    for (auto drv : drivers) {
        std::uint32_t n_dev = 0;
        zeDeviceGet(drv, &n_dev, nullptr);
        if (n_dev == 0) continue;
        std::vector<ze_device_handle_t> devs(n_dev);
        zeDeviceGet(drv, &n_dev, devs.data());
        for (auto d : devs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(d, &p) != ZE_RESULT_SUCCESS) continue;
            char id[80];
            std::snprintf(id, sizeof(id), "ze-%04x:%04x-%s", p.vendorId, p.deviceId,
                          format_uuid(p.uuid.id).c_str());
            if (target.id() == id) {
                out_driver = drv;
                out_device = d;
                return true;
            }
        }
    }
    return false;
}

} // namespace

RunResult run_vector_add_oneapi(const gpgpu::Setup&    setup,
                                std::span<const float> a,
                                std::span<const float> b,
                                std::span<float>       c) {
    using namespace example_vs_vulkan_shader; // brings in k_vector_add_spv_bytes / _len

    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);
    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS && zeInit(0) != ZE_RESULT_SUCCESS) {
        r.error = "zeInit failed"; return r;
    }

    ze_driver_handle_t drv = nullptr;
    ze_device_handle_t dev = nullptr;
    if (!find_matching_device(setup.device, drv, dev)) {
        r.error = "no Level Zero device matched " + setup.device.id(); return r;
    }

    ze_device_properties_t dev_props{};
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(dev, &dev_props);
    const std::uint64_t timer_res_ns = dev_props.timerResolution;

    ze_context_desc_t ctx_desc{};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx = nullptr;
    if (zeContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
        r.error = "zeContextCreate failed"; return r;
    }

    ze_command_queue_desc_t cqd{};
    cqd.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqd.ordinal  = 0;
    cqd.mode     = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t queue = nullptr;
    zeCommandQueueCreate(ctx, dev, &cqd, &queue);

    ze_command_list_desc_t cld{};
    cld.stype                    = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cld.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t list = nullptr;
    zeCommandListCreate(ctx, dev, &cld, &list);

    ze_module_desc_t mdesc{};
    mdesc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mdesc.format       = ZE_MODULE_FORMAT_IL_SPIRV;
    mdesc.inputSize    = k_vector_add_spv_bytes_len;
    mdesc.pInputModule = k_vector_add_spv_bytes;
    mdesc.pBuildFlags  = "";
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t buildLog = nullptr;
    if (zeModuleCreate(ctx, dev, &mdesc, &module, &buildLog) != ZE_RESULT_SUCCESS) {
        std::string log;
        if (buildLog) {
            std::size_t lsz = 0;
            zeModuleBuildLogGetString(buildLog, &lsz, nullptr);
            log.resize(lsz);
            if (lsz > 0) zeModuleBuildLogGetString(buildLog, &lsz, log.data());
            zeModuleBuildLogDestroy(buildLog);
        }
        r.error = "zeModuleCreate failed: " + log;
        zeCommandListDestroy(list); zeCommandQueueDestroy(queue); zeContextDestroy(ctx);
        return r;
    }
    if (buildLog) zeModuleBuildLogDestroy(buildLog);

    ze_kernel_desc_t kdesc{};
    kdesc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = "vector_add";
    ze_kernel_handle_t kernel = nullptr;
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        kdesc.pKernelName = "main";
        if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
            r.error = "zeKernelCreate: no vector_add / main entry";
            zeModuleDestroy(module);
            zeCommandListDestroy(list); zeCommandQueueDestroy(queue); zeContextDestroy(ctx);
            return r;
        }
    }
    zeKernelSetGroupSize(kernel, kGroupSize, 1, 1);

    ze_device_mem_alloc_desc_t mad{};
    mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    void* dA = nullptr; void* dB = nullptr; void* dC = nullptr;
    zeMemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dA);
    zeMemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dB);
    zeMemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dC);

    std::uint32_t n_arg = static_cast<std::uint32_t>(n);
    zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dA);
    zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &dB);
    zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &dC);
    zeKernelSetArgumentValue(kernel, 3, sizeof(std::uint32_t), &n_arg);

    ze_event_pool_desc_t epd{};
    epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    epd.count = 4;
    ze_event_pool_handle_t epool = nullptr;
    zeEventPoolCreate(ctx, &epd, 1, &dev, &epool);

    auto make_event = [&](std::uint32_t i) {
        ze_event_desc_t ed{};
        ed.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
        ed.index  = i;
        ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        ze_event_handle_t e = nullptr;
        zeEventCreate(epool, &ed, &e);
        return e;
    };
    ze_event_handle_t ev_h2d_a = make_event(0);
    ze_event_handle_t ev_h2d_b = make_event(1);
    ze_event_handle_t ev_k     = make_event(2);
    ze_event_handle_t ev_d2h   = make_event(3);

    ze_group_count_t groups{};
    groups.groupCountX = static_cast<std::uint32_t>((n + kGroupSize - 1) / kGroupSize);
    groups.groupCountY = 1;
    groups.groupCountZ = 1;

    zeCommandListAppendMemoryCopy(list, dA, a.data(), bytes, ev_h2d_a, 0, nullptr);
    zeCommandListAppendMemoryCopy(list, dB, b.data(), bytes, ev_h2d_b, 0, nullptr);

    const auto t_launch_start = std::chrono::steady_clock::now();
    zeCommandListAppendLaunchKernel(list, kernel, &groups, ev_k, 0, nullptr);
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;

    zeCommandListAppendMemoryCopy(list, c.data(), dC, bytes, ev_d2h, 0, nullptr);
    zeCommandListClose(list);

    const auto t_total_start = std::chrono::steady_clock::now();
    if (zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr) != ZE_RESULT_SUCCESS) {
        r.error = "zeCommandQueueExecuteCommandLists failed";
    } else {
        zeCommandQueueSynchronize(queue, UINT64_MAX);
    }
    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    auto kts = [&](ze_event_handle_t e) -> std::pair<std::uint64_t, std::uint64_t> {
        ze_kernel_timestamp_result_t ts{};
        if (zeEventQueryKernelTimestamp(e, &ts) != ZE_RESULT_SUCCESS) return {0, 0};
        return {ts.global.kernelStart, ts.global.kernelEnd};
    };
    auto to_seconds = [&](std::uint64_t ticks) {
        return std::chrono::duration<double>{ticks * static_cast<double>(timer_res_ns) / 1.0e9};
    };
    auto [h2da_s, h2da_e] = kts(ev_h2d_a);
    auto [h2db_s, h2db_e] = kts(ev_h2d_b);
    auto [k_s,    k_e]    = kts(ev_k);
    auto [d2h_s,  d2h_e]  = kts(ev_d2h);
    if (h2db_e > h2da_s)  r.timings.copy_h2d       = to_seconds(h2db_e - h2da_s);
    if (k_e    > k_s)     r.timings.kernel_compute = to_seconds(k_e    - k_s);
    if (d2h_e  > d2h_s)   r.timings.copy_d2h       = to_seconds(d2h_e  - d2h_s);

    zeEventDestroy(ev_h2d_a); zeEventDestroy(ev_h2d_b);
    zeEventDestroy(ev_k);     zeEventDestroy(ev_d2h);
    zeEventPoolDestroy(epool);

    zeMemFree(ctx, dA); zeMemFree(ctx, dB); zeMemFree(ctx, dC);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeCommandListDestroy(list);
    zeCommandQueueDestroy(queue);
    zeContextDestroy(ctx);

    if (!r.error.empty()) return r;

    for (std::size_t i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-4f * std::fabs(expected) + 1e-5f) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "first mismatch at i=%zu: c=%g expected=%g", i, c[i], expected);
            r.error = buf;
            r.correct = false;
            return r;
        }
    }
    r.correct = true;
    return r;
}

} // namespace example
