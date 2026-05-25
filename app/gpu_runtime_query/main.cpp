// Sample app: print everything gpgpu::Runtime::query() discovered.

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include <gpgpu/runtime.hpp>
#include <gpgpu/version.hpp>

namespace {

std::string fmt_bytes(std::uint64_t bytes) {
    constexpr std::uint64_t KiB = 1024ULL;
    constexpr std::uint64_t MiB = KiB * 1024ULL;
    constexpr std::uint64_t GiB = MiB * 1024ULL;
    char buf[32];
    if (bytes >= GiB)      std::snprintf(buf, sizeof(buf), "%.2f GiB", (double)bytes / GiB);
    else if (bytes >= MiB) std::snprintf(buf, sizeof(buf), "%.2f MiB", (double)bytes / MiB);
    else if (bytes >= KiB) std::snprintf(buf, sizeof(buf), "%.2f KiB", (double)bytes / KiB);
    else                   std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    return buf;
}

template <typename T>
std::string fmt_opt(const std::optional<T>& v) {
    if (!v) return "n/a";
    return std::to_string(*v);
}

std::string fmt_opt_bytes(const std::optional<std::uint64_t>& v) {
    return v ? fmt_bytes(*v) : "n/a";
}

std::string fmt_opt_str(const std::optional<std::string>& v) {
    return v ? *v : std::string("n/a");
}

std::string yn(bool b) { return b ? "yes" : "no"; }

} // namespace

int main() {
    auto v = gpgpu::version();
    std::printf("gpu_runtime %u.%u.%u\n\n", v.major, v.minor, v.patch);

    auto report = gpgpu::Runtime::query();

    std::printf("== Backends ==\n");
    for (const auto& b : report.backends()) {
        std::printf("  %-8s %-10s status=%-13s",
                    std::string(b.name()).c_str(),
                    b.version().empty() ? "-" : b.version().c_str(),
                    std::string(gpgpu::to_string(b.status())).c_str());
        if (!b.path().empty()) std::printf(" path=%s", b.path().c_str());
        std::printf("\n");
        if (b.last_error() && !b.last_error()->empty()) {
            std::printf("           error: %s\n", b.last_error()->c_str());
        }
    }

    std::printf("\n== Devices ==\n");
    if (report.empty()) {
        std::printf("  (no GPGPU devices found)\n");
        return 0;
    }

    for (const auto& dev : report.devices()) {
        std::printf("  [%s] %s\n", dev.id().c_str(), dev.name().c_str());
        std::printf("      vendor=%s (%s)\n",
                    std::string(gpgpu::to_string(dev.vendor())).c_str(),
                    dev.vendor_string().c_str());
        std::printf("      memory=%s  bandwidth=%s GB/s  bus_width=%s bits  L2=%s\n",
                    fmt_opt_bytes(dev.memory()).c_str(),
                    fmt_opt(dev.memory_bandwidth_gbps()).c_str(),
                    fmt_opt(dev.memory_bus_width_bits()).c_str(),
                    fmt_opt_bytes(dev.l2_cache_bytes()).c_str());
        std::printf("      cores=%s  compute_units=%s  subgroup=%s  max_wg=%s  shared_mem=%s\n",
                    fmt_opt(dev.cores()).c_str(),
                    fmt_opt(dev.compute_units()).c_str(),
                    fmt_opt(dev.subgroup_size()).c_str(),
                    fmt_opt(dev.max_workgroup_size()).c_str(),
                    fmt_opt(dev.shared_memory_per_block()).c_str());
        std::printf("      clock=%s MHz  integrated=%s  driver=%s\n",
                    fmt_opt(dev.frequency()).c_str(),
                    dev.integrated() ? yn(*dev.integrated()).c_str() : "n/a",
                    fmt_opt_str(dev.driver_version()).c_str());
        std::printf("      fp16=%s fp64=%s int8=%s tensor_cores=%s\n",
                    yn(dev.supports_fp16()).c_str(),
                    yn(dev.supports_fp64()).c_str(),
                    yn(dev.supports_int8()).c_str(),
                    yn(dev.supports_tensor_cores()).c_str());

        std::printf("      Setups:\n");
        for (const auto& s : report.setups_for(dev.id())) {
            std::printf("        %-8s %-10s preferred=%s\n",
                        std::string(s.backend.name()).c_str(),
                        s.backend.version().empty() ? "-" : s.backend.version().c_str(),
                        std::string(gpgpu::to_string(s.preferred)).c_str());
        }
    }

    return 0;
}
