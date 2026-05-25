#pragma once

#include <string>
#include <vector>

#include "gpgpu/backend.hpp"

namespace gpgpu::search {

// Standard install paths for the runtime shared library of the given backend,
// in priority order. Each entry is either an absolute path or a bare library
// filename (resolved by the OS loader's default search). Returned list is
// host-OS specific.
std::vector<std::string> candidates(BackendId backend);

} // namespace gpgpu::search
