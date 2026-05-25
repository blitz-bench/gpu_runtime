#include "probe.hpp"

namespace gpgpu::backends {

Backend make_missing(BackendId id, std::string error) {
    return Backend(id, /*version*/ {}, /*path*/ {},
                   BackendStatus::MissingLib, std::move(error));
}

} // namespace gpgpu::backends
