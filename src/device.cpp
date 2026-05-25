#include "gpgpu/device.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace gpgpu {

std::string_view to_string(Vendor v) noexcept {
    switch (v) {
        case Vendor::Nvidia:   return "NVIDIA";
        case Vendor::AMD:      return "AMD";
        case Vendor::Intel:    return "Intel";
        case Vendor::Apple:    return "Apple";
        case Vendor::Qualcomm: return "Qualcomm";
        case Vendor::ARM:      return "ARM";
        case Vendor::Unknown:  return "Unknown";
    }
    return "Unknown";
}

Vendor classify_vendor(std::string_view raw) noexcept {
    std::string lower;
    lower.reserve(raw.size());
    for (char c : raw) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    auto contains = [&](std::string_view needle) {
        return lower.find(needle) != std::string::npos;
    };

    if (contains("nvidia") || contains("nvcorp")) return Vendor::Nvidia;
    if (contains("advanced micro") || contains("amd") || contains("ati ") || contains("radeon")) return Vendor::AMD;
    if (contains("intel")) return Vendor::Intel;
    if (contains("apple")) return Vendor::Apple;
    if (contains("qualcomm") || contains("adreno")) return Vendor::Qualcomm;
    if (contains("arm ") || contains("mali")) return Vendor::ARM;
    return Vendor::Unknown;
}

Vendor classify_vendor_pci(std::uint32_t pci_vendor_id) noexcept {
    switch (pci_vendor_id) {
        case 0x10DE: return Vendor::Nvidia;
        case 0x1002: return Vendor::AMD;
        case 0x1022: return Vendor::AMD; // AMD bridge IDs occasionally show up
        case 0x8086: return Vendor::Intel;
        case 0x106B: return Vendor::Apple;
        case 0x5143: return Vendor::Qualcomm;
        case 0x13B5: return Vendor::ARM;
        default:     return Vendor::Unknown;
    }
}

} // namespace gpgpu
