# gpu_runtime

A C++ library that detects available (GP)GPU backends and loads them in
runtime. It also provides Device information for detected GPUs and their
backends.

## Building

```bash
# After clone, fetch the third-party header submodules:
git submodule update --init --recursive

cmake -B build -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/app/gpu_runtime_query/gpu_runtime_query
```

## Third-party headers

Three GPGPU API header sets are pulled in as git submodules under `external/`.
None of the vendor runtimes is required at build time — the libraries are
`dlopen`'d at runtime.

| Submodule | Upstream | License |
|---|---|---|
| `external/OpenCL-Headers` | [KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers) | Apache-2.0 |
| `external/Vulkan-Headers` | [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | Apache-2.0 (a few files MIT) |
| `external/level-zero`     | [oneapi-src/level-zero](https://github.com/oneapi-src/level-zero) | MIT (a few files Apache-2.0) |

Each submodule ships its own `LICENSE`. To pin a different upstream revision,
`cd` into the submodule, `git checkout <ref>`, then commit the submodule SHA
update in this repository.

### Not vendored

- **NVIDIA CUDA** headers: proprietary license, redistribution not permitted.
  The CUDA backend (`src/backends/cuda.cpp`) declares the small Driver-API
  subset it uses inline.
- **AMD HIP / ROCm** headers: MIT-licensed, but `hip_runtime_api.h` is 10 K+
  lines and transitively pulls in `nvidia_detail/` and `amd_detail/` paths
  that require full CUDA / ROCm SDK installations. The HIP backend
  (`src/backends/rocm.cpp`) declares the small Runtime-API subset it uses
  inline.
- **Apple Metal**: `<Metal/Metal.h>` is part of the macOS system SDK and is
  `#import`ed directly when building on Apple platforms.
