// Non-Apple stub for the Metal runner symbol.
#include "metal_runner.hpp"

namespace example {
RunResult run_vector_add_metal(const gpgpu::Setup&,
                               std::span<const float>,
                               std::span<const float>,
                               std::span<float>) {
    RunResult r;
    r.error = "Metal is only available on Apple platforms";
    return r;
}
} // namespace example
