// OneAPI / Level Zero vector-add runner - single-file, no shared abstraction.
//
// Walks the L0 compute path: zeInit -> find driver+device matching
// setup.device.id() -> context -> command queue + immediate command list ->
// SPIR-V module -> kernel -> device-allocated buffers -> append memcpy +
// launch + memcpy -> close + execute + sync.

#include "oneapi_runner.hpp"
#include "lib_loader.hpp"

#include <ze_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../shaders/vector_add.spv.inl"

namespace example {

namespace {

constexpr std::uint32_t kGroupSize = 256;

// --- function pointer subset (ze_api.h omits PFN typedefs) ---
using PFN_zeInit                       = ze_result_t (ZE_APICALL*)(ze_init_flags_t);
using PFN_zeDriverGet                  = ze_result_t (ZE_APICALL*)(uint32_t*, ze_driver_handle_t*);
using PFN_zeDeviceGet                  = ze_result_t (ZE_APICALL*)(ze_driver_handle_t, uint32_t*, ze_device_handle_t*);
using PFN_zeDeviceGetProperties        = ze_result_t (ZE_APICALL*)(ze_device_handle_t, ze_device_properties_t*);
using PFN_zeContextCreate              = ze_result_t (ZE_APICALL*)(ze_driver_handle_t, const ze_context_desc_t*, ze_context_handle_t*);
using PFN_zeContextDestroy             = ze_result_t (ZE_APICALL*)(ze_context_handle_t);
using PFN_zeCommandQueueCreate         = ze_result_t (ZE_APICALL*)(ze_context_handle_t, ze_device_handle_t,
                                                                   const ze_command_queue_desc_t*, ze_command_queue_handle_t*);
using PFN_zeCommandQueueDestroy        = ze_result_t (ZE_APICALL*)(ze_command_queue_handle_t);
using PFN_zeCommandQueueExecuteCommandLists =
    ze_result_t (ZE_APICALL*)(ze_command_queue_handle_t, uint32_t, ze_command_list_handle_t*, ze_fence_handle_t);
using PFN_zeCommandQueueSynchronize    = ze_result_t (ZE_APICALL*)(ze_command_queue_handle_t, uint64_t);
using PFN_zeCommandListCreate          = ze_result_t (ZE_APICALL*)(ze_context_handle_t, ze_device_handle_t,
                                                                   const ze_command_list_desc_t*, ze_command_list_handle_t*);
using PFN_zeCommandListDestroy         = ze_result_t (ZE_APICALL*)(ze_command_list_handle_t);
using PFN_zeCommandListClose           = ze_result_t (ZE_APICALL*)(ze_command_list_handle_t);
using PFN_zeCommandListAppendMemoryCopy =
    ze_result_t (ZE_APICALL*)(ze_command_list_handle_t, void*, const void*, size_t,
                              ze_event_handle_t, uint32_t, ze_event_handle_t*);
using PFN_zeCommandListAppendLaunchKernel =
    ze_result_t (ZE_APICALL*)(ze_command_list_handle_t, ze_kernel_handle_t,
                              const ze_group_count_t*, ze_event_handle_t, uint32_t, ze_event_handle_t*);
using PFN_zeModuleCreate               = ze_result_t (ZE_APICALL*)(ze_context_handle_t, ze_device_handle_t,
                                                                   const ze_module_desc_t*, ze_module_handle_t*,
                                                                   ze_module_build_log_handle_t*);
using PFN_zeModuleDestroy              = ze_result_t (ZE_APICALL*)(ze_module_handle_t);
using PFN_zeModuleBuildLogGetString    = ze_result_t (ZE_APICALL*)(ze_module_build_log_handle_t, size_t*, char*);
using PFN_zeModuleBuildLogDestroy      = ze_result_t (ZE_APICALL*)(ze_module_build_log_handle_t);
using PFN_zeKernelCreate               = ze_result_t (ZE_APICALL*)(ze_module_handle_t, const ze_kernel_desc_t*, ze_kernel_handle_t*);
using PFN_zeKernelDestroy              = ze_result_t (ZE_APICALL*)(ze_kernel_handle_t);
using PFN_zeKernelSetGroupSize         = ze_result_t (ZE_APICALL*)(ze_kernel_handle_t, uint32_t, uint32_t, uint32_t);
using PFN_zeKernelSetArgumentValue     = ze_result_t (ZE_APICALL*)(ze_kernel_handle_t, uint32_t, size_t, const void*);
using PFN_zeMemAllocDevice             = ze_result_t (ZE_APICALL*)(ze_context_handle_t, const ze_device_mem_alloc_desc_t*,
                                                                   size_t, size_t, ze_device_handle_t, void**);
using PFN_zeMemFree                    = ze_result_t (ZE_APICALL*)(ze_context_handle_t, void*);
using PFN_zeEventPoolCreate            = ze_result_t (ZE_APICALL*)(ze_context_handle_t, const ze_event_pool_desc_t*,
                                                                   uint32_t, ze_device_handle_t*, ze_event_pool_handle_t*);
using PFN_zeEventPoolDestroy           = ze_result_t (ZE_APICALL*)(ze_event_pool_handle_t);
using PFN_zeEventCreate                = ze_result_t (ZE_APICALL*)(ze_event_pool_handle_t, const ze_event_desc_t*, ze_event_handle_t*);
using PFN_zeEventDestroy               = ze_result_t (ZE_APICALL*)(ze_event_handle_t);
using PFN_zeEventQueryKernelTimestamp  = ze_result_t (ZE_APICALL*)(ze_event_handle_t, ze_kernel_timestamp_result_t*);

struct ZeApi {
    PFN_zeInit                            Init;
    PFN_zeDriverGet                       DriverGet;
    PFN_zeDeviceGet                       DeviceGet;
    PFN_zeDeviceGetProperties             DeviceGetProperties;
    PFN_zeContextCreate                   ContextCreate;
    PFN_zeContextDestroy                  ContextDestroy;
    PFN_zeCommandQueueCreate              CommandQueueCreate;
    PFN_zeCommandQueueDestroy             CommandQueueDestroy;
    PFN_zeCommandQueueExecuteCommandLists CommandQueueExecuteCommandLists;
    PFN_zeCommandQueueSynchronize         CommandQueueSynchronize;
    PFN_zeCommandListCreate               CommandListCreate;
    PFN_zeCommandListDestroy              CommandListDestroy;
    PFN_zeCommandListClose                CommandListClose;
    PFN_zeCommandListAppendMemoryCopy     CommandListAppendMemoryCopy;
    PFN_zeCommandListAppendLaunchKernel   CommandListAppendLaunchKernel;
    PFN_zeModuleCreate                    ModuleCreate;
    PFN_zeModuleDestroy                   ModuleDestroy;
    PFN_zeModuleBuildLogGetString         ModuleBuildLogGetString;
    PFN_zeModuleBuildLogDestroy           ModuleBuildLogDestroy;
    PFN_zeKernelCreate                    KernelCreate;
    PFN_zeKernelDestroy                   KernelDestroy;
    PFN_zeKernelSetGroupSize              KernelSetGroupSize;
    PFN_zeKernelSetArgumentValue          KernelSetArgumentValue;
    PFN_zeMemAllocDevice                  MemAllocDevice;
    PFN_zeMemFree                         MemFree;
    PFN_zeEventPoolCreate                 EventPoolCreate;
    PFN_zeEventPoolDestroy                EventPoolDestroy;
    PFN_zeEventCreate                     EventCreate;
    PFN_zeEventDestroy                    EventDestroy;
    PFN_zeEventQueryKernelTimestamp       EventQueryKernelTimestamp;
};

bool resolve(const loader::Lib& lib, ZeApi& api) {
    api.Init                            = lib.sym<PFN_zeInit>("zeInit");
    api.DriverGet                       = lib.sym<PFN_zeDriverGet>("zeDriverGet");
    api.DeviceGet                       = lib.sym<PFN_zeDeviceGet>("zeDeviceGet");
    api.DeviceGetProperties             = lib.sym<PFN_zeDeviceGetProperties>("zeDeviceGetProperties");
    api.ContextCreate                   = lib.sym<PFN_zeContextCreate>("zeContextCreate");
    api.ContextDestroy                  = lib.sym<PFN_zeContextDestroy>("zeContextDestroy");
    api.CommandQueueCreate              = lib.sym<PFN_zeCommandQueueCreate>("zeCommandQueueCreate");
    api.CommandQueueDestroy             = lib.sym<PFN_zeCommandQueueDestroy>("zeCommandQueueDestroy");
    api.CommandQueueExecuteCommandLists = lib.sym<PFN_zeCommandQueueExecuteCommandLists>("zeCommandQueueExecuteCommandLists");
    api.CommandQueueSynchronize         = lib.sym<PFN_zeCommandQueueSynchronize>("zeCommandQueueSynchronize");
    api.CommandListCreate               = lib.sym<PFN_zeCommandListCreate>("zeCommandListCreate");
    api.CommandListDestroy              = lib.sym<PFN_zeCommandListDestroy>("zeCommandListDestroy");
    api.CommandListClose                = lib.sym<PFN_zeCommandListClose>("zeCommandListClose");
    api.CommandListAppendMemoryCopy     = lib.sym<PFN_zeCommandListAppendMemoryCopy>("zeCommandListAppendMemoryCopy");
    api.CommandListAppendLaunchKernel   = lib.sym<PFN_zeCommandListAppendLaunchKernel>("zeCommandListAppendLaunchKernel");
    api.ModuleCreate                    = lib.sym<PFN_zeModuleCreate>("zeModuleCreate");
    api.ModuleDestroy                   = lib.sym<PFN_zeModuleDestroy>("zeModuleDestroy");
    api.ModuleBuildLogGetString         = lib.sym<PFN_zeModuleBuildLogGetString>("zeModuleBuildLogGetString");
    api.ModuleBuildLogDestroy           = lib.sym<PFN_zeModuleBuildLogDestroy>("zeModuleBuildLogDestroy");
    api.KernelCreate                    = lib.sym<PFN_zeKernelCreate>("zeKernelCreate");
    api.KernelDestroy                   = lib.sym<PFN_zeKernelDestroy>("zeKernelDestroy");
    api.KernelSetGroupSize              = lib.sym<PFN_zeKernelSetGroupSize>("zeKernelSetGroupSize");
    api.KernelSetArgumentValue          = lib.sym<PFN_zeKernelSetArgumentValue>("zeKernelSetArgumentValue");
    api.MemAllocDevice                  = lib.sym<PFN_zeMemAllocDevice>("zeMemAllocDevice");
    api.MemFree                         = lib.sym<PFN_zeMemFree>("zeMemFree");
    api.EventPoolCreate                 = lib.sym<PFN_zeEventPoolCreate>("zeEventPoolCreate");
    api.EventPoolDestroy                = lib.sym<PFN_zeEventPoolDestroy>("zeEventPoolDestroy");
    api.EventCreate                     = lib.sym<PFN_zeEventCreate>("zeEventCreate");
    api.EventDestroy                    = lib.sym<PFN_zeEventDestroy>("zeEventDestroy");
    api.EventQueryKernelTimestamp       = lib.sym<PFN_zeEventQueryKernelTimestamp>("zeEventQueryKernelTimestamp");
    return api.Init && api.DriverGet && api.DeviceGet && api.DeviceGetProperties &&
           api.ContextCreate && api.ContextDestroy &&
           api.CommandQueueCreate && api.CommandQueueDestroy &&
           api.CommandQueueExecuteCommandLists && api.CommandQueueSynchronize &&
           api.CommandListCreate && api.CommandListDestroy && api.CommandListClose &&
           api.CommandListAppendMemoryCopy && api.CommandListAppendLaunchKernel &&
           api.ModuleCreate && api.ModuleDestroy &&
           api.KernelCreate && api.KernelDestroy &&
           api.KernelSetGroupSize && api.KernelSetArgumentValue &&
           api.MemAllocDevice && api.MemFree &&
           api.EventPoolCreate && api.EventPoolDestroy &&
           api.EventCreate && api.EventDestroy && api.EventQueryKernelTimestamp;
}

std::string format_uuid(const std::uint8_t u[ZE_MAX_DEVICE_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

bool find_matching_device(const ZeApi& api, const gpgpu::Device& target,
                          ze_driver_handle_t& out_driver,
                          ze_device_handle_t& out_device) {
    std::uint32_t n_drv = 0;
    api.DriverGet(&n_drv, nullptr);
    if (n_drv == 0) return false;
    std::vector<ze_driver_handle_t> drivers(n_drv);
    api.DriverGet(&n_drv, drivers.data());
    for (auto drv : drivers) {
        std::uint32_t n_dev = 0;
        api.DeviceGet(drv, &n_dev, nullptr);
        if (n_dev == 0) continue;
        std::vector<ze_device_handle_t> devs(n_dev);
        api.DeviceGet(drv, &n_dev, devs.data());
        for (auto d : devs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (api.DeviceGetProperties(d, &p) != ZE_RESULT_SUCCESS) continue;
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
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    if (kVectorAddSpvBytesLen == 0) {
        r.error = "SPIR-V is empty - regenerate shaders/vector_add.spv.inl";
        return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);

    loader::Lib lib(setup.backend.path());
    if (!lib.ok()) { r.error = "dlopen(" + setup.backend.path() + ") failed"; return r; }

    ZeApi api{};
    if (!resolve(lib, api)) { r.error = "Level Zero symbol resolution failed"; return r; }

    if (api.Init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS && api.Init(0) != ZE_RESULT_SUCCESS) {
        r.error = "zeInit failed"; return r;
    }

    ze_driver_handle_t drv = nullptr;
    ze_device_handle_t dev = nullptr;
    if (!find_matching_device(api, setup.device, drv, dev)) {
        r.error = "no Level Zero device matched " + setup.device.id();
        return r;
    }

    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    // Capture timerResolution so we can convert kernel-timestamp ticks to ns.
    ze_device_properties_t dev_props{};
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    api.DeviceGetProperties(dev, &dev_props);
    const std::uint64_t timer_res_ns = dev_props.timerResolution; // ns per tick

    ze_context_desc_t ctx_desc{};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx = nullptr;
    if (api.ContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
        r.error = "zeContextCreate failed"; return r;
    }

    ze_command_queue_desc_t cqd{};
    cqd.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqd.ordinal  = 0;
    cqd.index    = 0;
    cqd.mode     = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t queue = nullptr;
    if (api.CommandQueueCreate(ctx, dev, &cqd, &queue) != ZE_RESULT_SUCCESS) {
        api.ContextDestroy(ctx);
        r.error = "zeCommandQueueCreate failed"; return r;
    }

    ze_command_list_desc_t cld{};
    cld.stype                          = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cld.commandQueueGroupOrdinal       = 0;
    ze_command_list_handle_t list = nullptr;
    if (api.CommandListCreate(ctx, dev, &cld, &list) != ZE_RESULT_SUCCESS) {
        api.CommandQueueDestroy(queue); api.ContextDestroy(ctx);
        r.error = "zeCommandListCreate failed"; return r;
    }

    // Module from SPIR-V.
    ze_module_desc_t mdesc{};
    mdesc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mdesc.format       = ZE_MODULE_FORMAT_IL_SPIRV;
    mdesc.inputSize    = kVectorAddSpvBytesLen;
    mdesc.pInputModule = kVectorAddSpvBytes;
    mdesc.pBuildFlags  = "";
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t buildLog = nullptr;
    ze_result_t zrc = api.ModuleCreate(ctx, dev, &mdesc, &module, &buildLog);
    if (zrc != ZE_RESULT_SUCCESS) {
        std::string log;
        if (buildLog) {
            std::size_t lsz = 0;
            api.ModuleBuildLogGetString(buildLog, &lsz, nullptr);
            log.resize(lsz);
            if (lsz > 0) api.ModuleBuildLogGetString(buildLog, &lsz, log.data());
            api.ModuleBuildLogDestroy(buildLog);
        }
        r.error = "zeModuleCreate failed: " + log;
        api.CommandListDestroy(list); api.CommandQueueDestroy(queue); api.ContextDestroy(ctx);
        return r;
    }
    if (buildLog) api.ModuleBuildLogDestroy(buildLog);

    ze_kernel_desc_t kdesc{};
    kdesc.stype        = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName  = "vector_add";   // GLSL shader's entry point
    ze_kernel_handle_t kernel = nullptr;
    if (api.KernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        // Many SPIR-V producers name the entry "main"; try that as a fallback.
        kdesc.pKernelName = "main";
        if (api.KernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
            r.error = "zeKernelCreate: no vector_add / main entry";
            api.ModuleDestroy(module);
            api.CommandListDestroy(list); api.CommandQueueDestroy(queue); api.ContextDestroy(ctx);
            return r;
        }
    }

    api.KernelSetGroupSize(kernel, kGroupSize, 1, 1);

    // Device-resident buffers.
    ze_device_mem_alloc_desc_t mad{};
    mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    void* dA = nullptr; void* dB = nullptr; void* dC = nullptr;
    api.MemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dA);
    api.MemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dB);
    api.MemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &dC);

    std::uint32_t n_arg = static_cast<std::uint32_t>(n);
    api.KernelSetArgumentValue(kernel, 0, sizeof(void*),      &dA);
    api.KernelSetArgumentValue(kernel, 1, sizeof(void*),      &dB);
    api.KernelSetArgumentValue(kernel, 2, sizeof(void*),      &dC);
    api.KernelSetArgumentValue(kernel, 3, sizeof(std::uint32_t), &n_arg);

    // Kernel-timestamp event pool: 4 events bracket the 4 commands.
    ze_event_pool_desc_t epd{};
    epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    epd.count = 4;
    ze_event_pool_handle_t epool = nullptr;
    api.EventPoolCreate(ctx, &epd, 1, &dev, &epool);

    auto make_event = [&](std::uint32_t index) {
        ze_event_desc_t ed{};
        ed.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
        ed.index  = index;
        ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        ze_event_handle_t e = nullptr;
        api.EventCreate(epool, &ed, &e);
        return e;
    };
    ze_event_handle_t ev_h2d_a = make_event(0);
    ze_event_handle_t ev_h2d_b = make_event(1);
    ze_event_handle_t ev_kernel = make_event(2);
    ze_event_handle_t ev_d2h   = make_event(3);

    ze_group_count_t groups{};
    groups.groupCountX = static_cast<std::uint32_t>((n + kGroupSize - 1) / kGroupSize);
    groups.groupCountY = 1;
    groups.groupCountZ = 1;

    api.CommandListAppendMemoryCopy(list, dA, a.data(), bytes, ev_h2d_a, 0, nullptr);
    api.CommandListAppendMemoryCopy(list, dB, b.data(), bytes, ev_h2d_b, 0, nullptr);

    const auto t_launch_start = std::chrono::steady_clock::now();
    api.CommandListAppendLaunchKernel(list, kernel, &groups, ev_kernel, 0, nullptr);
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;

    api.CommandListAppendMemoryCopy(list, c.data(), dC, bytes, ev_d2h, 0, nullptr);
    api.CommandListClose(list);

    const auto t_total_start = std::chrono::steady_clock::now();
    if (api.CommandQueueExecuteCommandLists(queue, 1, &list, nullptr) != ZE_RESULT_SUCCESS) {
        r.error = "zeCommandQueueExecuteCommandLists failed";
    } else {
        api.CommandQueueSynchronize(queue, UINT64_MAX);
    }
    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    auto kts = [&](ze_event_handle_t e) -> std::pair<std::uint64_t, std::uint64_t> {
        ze_kernel_timestamp_result_t ts{};
        if (api.EventQueryKernelTimestamp(e, &ts) != ZE_RESULT_SUCCESS) return {0, 0};
        return {ts.global.kernelStart, ts.global.kernelEnd};
    };
    auto to_seconds = [&](std::uint64_t ticks) {
        return std::chrono::duration<double>{ticks * static_cast<double>(timer_res_ns) / 1.0e9};
    };
    auto [h2da_s, h2da_e] = kts(ev_h2d_a);
    auto [h2db_s, h2db_e] = kts(ev_h2d_b);
    auto [k_s, k_e]       = kts(ev_kernel);
    auto [d2h_s, d2h_e]   = kts(ev_d2h);
    if (h2db_e > h2da_s)  r.timings.copy_h2d       = to_seconds(h2db_e - h2da_s);
    if (k_e    > k_s)     r.timings.kernel_compute = to_seconds(k_e    - k_s);
    if (d2h_e  > d2h_s)   r.timings.copy_d2h       = to_seconds(d2h_e  - d2h_s);

    api.EventDestroy(ev_h2d_a); api.EventDestroy(ev_h2d_b);
    api.EventDestroy(ev_kernel); api.EventDestroy(ev_d2h);
    api.EventPoolDestroy(epool);

    api.MemFree(ctx, dA);
    api.MemFree(ctx, dB);
    api.MemFree(ctx, dC);
    api.KernelDestroy(kernel);
    api.ModuleDestroy(module);
    api.CommandListDestroy(list);
    api.CommandQueueDestroy(queue);
    api.ContextDestroy(ctx);

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
