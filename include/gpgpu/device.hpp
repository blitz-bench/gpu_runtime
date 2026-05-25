#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gpgpu {

enum class Vendor : std::uint8_t {
    Nvidia,
    AMD,
    Intel,
    Apple,
    Qualcomm,
    ARM,
    Unknown,
};

std::string_view to_string(Vendor v) noexcept;

// Best-effort vendor classification from a raw vendor string or PCI vendor ID.
Vendor classify_vendor(std::string_view raw) noexcept;
Vendor classify_vendor_pci(std::uint32_t pci_vendor_id) noexcept;

class Device {
public:
    Device() = default;

    // --- stable identity ---
    [[nodiscard]] const std::string& id() const noexcept { return id_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] Vendor vendor() const noexcept { return vendor_; }
    [[nodiscard]] const std::string& vendor_string() const noexcept { return vendor_string_; }

    // --- memory ---
    [[nodiscard]] std::optional<std::uint64_t> memory() const noexcept { return memory_; }
    [[nodiscard]] std::optional<std::uint64_t> memory_bandwidth_gbps() const noexcept { return memory_bandwidth_gbps_; }
    [[nodiscard]] std::optional<std::uint64_t> l2_cache_bytes() const noexcept { return l2_cache_bytes_; }
    [[nodiscard]] std::optional<std::uint32_t> memory_bus_width_bits() const noexcept { return memory_bus_width_bits_; }

    // --- compute units ---
    [[nodiscard]] std::optional<std::uint32_t> cores() const noexcept { return cores_; }
    [[nodiscard]] std::optional<std::uint32_t> compute_units() const noexcept { return compute_units_; }
    [[nodiscard]] std::optional<std::uint32_t> subgroup_size() const noexcept { return subgroup_size_; }
    [[nodiscard]] std::optional<std::uint32_t> max_workgroup_size() const noexcept { return max_workgroup_size_; }
    [[nodiscard]] std::optional<std::uint32_t> shared_memory_per_block() const noexcept { return shared_memory_per_block_; }

    // --- clocks ---
    [[nodiscard]] std::optional<std::uint32_t> frequency() const noexcept { return frequency_; }

    // --- topology / form ---
    [[nodiscard]] std::optional<bool> integrated() const noexcept { return integrated_; }
    [[nodiscard]] std::optional<std::uint32_t> pcie_gen() const noexcept { return pcie_gen_; }
    [[nodiscard]] std::optional<std::uint32_t> pcie_width() const noexcept { return pcie_width_; }

    // --- driver ---
    [[nodiscard]] const std::optional<std::string>& driver_version() const noexcept { return driver_version_; }

    // --- feature flags ---
    [[nodiscard]] bool supports_fp16() const noexcept { return fp16_; }
    [[nodiscard]] bool supports_fp64() const noexcept { return fp64_; }
    [[nodiscard]] bool supports_int8() const noexcept { return int8_; }
    [[nodiscard]] bool supports_tensor_cores() const noexcept { return tensor_cores_; }

    // Builder methods are public so each backend can populate the value type
    // without forcing a deep friend graph. The Device is only mutated by its
    // producing backend during the query() call.
    Device& set_id(std::string v) { id_ = std::move(v); return *this; }
    Device& set_name(std::string v) { name_ = std::move(v); return *this; }
    Device& set_vendor(Vendor v) { vendor_ = v; return *this; }
    Device& set_vendor_string(std::string v) { vendor_string_ = std::move(v); return *this; }
    Device& set_memory(std::uint64_t v) { memory_ = v; return *this; }
    Device& set_memory_bandwidth_gbps(std::uint64_t v) { memory_bandwidth_gbps_ = v; return *this; }
    Device& set_l2_cache_bytes(std::uint64_t v) { l2_cache_bytes_ = v; return *this; }
    Device& set_memory_bus_width_bits(std::uint32_t v) { memory_bus_width_bits_ = v; return *this; }
    Device& set_cores(std::uint32_t v) { cores_ = v; return *this; }
    Device& set_compute_units(std::uint32_t v) { compute_units_ = v; return *this; }
    Device& set_subgroup_size(std::uint32_t v) { subgroup_size_ = v; return *this; }
    Device& set_max_workgroup_size(std::uint32_t v) { max_workgroup_size_ = v; return *this; }
    Device& set_shared_memory_per_block(std::uint32_t v) { shared_memory_per_block_ = v; return *this; }
    Device& set_frequency(std::uint32_t v) { frequency_ = v; return *this; }
    Device& set_integrated(bool v) { integrated_ = v; return *this; }
    Device& set_pcie_gen(std::uint32_t v) { pcie_gen_ = v; return *this; }
    Device& set_pcie_width(std::uint32_t v) { pcie_width_ = v; return *this; }
    Device& set_driver_version(std::string v) { driver_version_ = std::move(v); return *this; }
    Device& set_fp16(bool v) { fp16_ = v; return *this; }
    Device& set_fp64(bool v) { fp64_ = v; return *this; }
    Device& set_int8(bool v) { int8_ = v; return *this; }
    Device& set_tensor_cores(bool v) { tensor_cores_ = v; return *this; }

private:
    std::string id_;
    std::string name_;
    Vendor vendor_{Vendor::Unknown};
    std::string vendor_string_;

    std::optional<std::uint64_t> memory_;
    std::optional<std::uint64_t> memory_bandwidth_gbps_;
    std::optional<std::uint64_t> l2_cache_bytes_;
    std::optional<std::uint32_t> memory_bus_width_bits_;

    std::optional<std::uint32_t> cores_;
    std::optional<std::uint32_t> compute_units_;
    std::optional<std::uint32_t> subgroup_size_;
    std::optional<std::uint32_t> max_workgroup_size_;
    std::optional<std::uint32_t> shared_memory_per_block_;

    std::optional<std::uint32_t> frequency_;

    std::optional<bool> integrated_;
    std::optional<std::uint32_t> pcie_gen_;
    std::optional<std::uint32_t> pcie_width_;

    std::optional<std::string> driver_version_;

    bool fp16_{false};
    bool fp64_{false};
    bool int8_{false};
    bool tensor_cores_{false};
};

} // namespace gpgpu
