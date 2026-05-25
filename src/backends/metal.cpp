// Metal stub for non-Apple targets. On Apple platforms the real probe lives in
// metal.mm and this translation unit contributes nothing.

#if !defined(__APPLE__)

#include "probe.hpp"

namespace gpgpu::backends {

ProbeResult probe_metal() {
    ProbeResult out;
    out.backend = make_missing(BackendId::Metal,
                               std::string("Metal is only available on Apple platforms"));
    return out;
}

} // namespace gpgpu::backends

#endif // !__APPLE__
