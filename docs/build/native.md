# Native build (Arch Linux + CUDA)

!!! abstract "TL;DR"
    Reference platform is Arch Linux with CUDA 13.x and an RTX 5090
    (Blackwell, sm_120). `scripts/install_native.sh` is the supported
    one-shot installer; this page documents what it does so you can
    reproduce it by hand or adapt to a different distribution.

## System packages

`scripts/install_native.sh:45` invokes:

```bash
sudo pacman -S --needed --noconfirm \
    cmake opencv protobuf grpc jsoncpp openssl c-ares nginx curl libjpeg-turbo
```

On Debian/Ubuntu the equivalents include `libcurl4-openssl-dev` and
`libturbojpeg0-dev` — both are hard link dependencies of the GPU build (the
VLM table/formula clients use libcurl even though they are runtime-opt-in,
and inline page-image export uses libjpeg-turbo as the CPU fallback encoder).
The GPU build additionally needs a **CUDA-enabled** ONNX Runtime (with
`libonnxruntime_providers_cuda.so`) in `third_party/onnxruntime/{include,lib}`
or `/usr/local`; the plain `onnxruntime-linux-<arch>-<ver>.tgz` release asset
is CPU-only and is rejected at configure time with an explanatory error.

!!! warning "CUDA prerequisite"
    The installer refuses to proceed if `nvcc --version` reports a
    release below 13.0 (`install_native.sh:33`). On Arch the package is
    `cuda`; ensure `nvcc` is on `$PATH` before running the installer.

## Drogon HTTP framework

Pinned to **v1.9.12**, built from source — there is no Arch package.
`install_native.sh:48-66` clones the upstream repo into a tempdir,
configures with every optional ORM / Redis / SQLite backend disabled,
and `sudo cmake --install build`s into `/usr/local`.

```bash
cmake -B build \
      -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
      -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
      -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF
cmake --build build -j"$(nproc)"
sudo cmake --install build
```

## TensorRT

Version is mapped to the host CUDA version
(`install_native.sh:82-99`):

| CUDA   | TensorRT       | Tar suffix    |
|--------|----------------|---------------|
| 13.0   | 10.14.1.16     | `cuda-13.0`   |
| 13.1   | 10.15.1.29     | `cuda-13.1`   |
| 13.2   | 10.16.0.72     | `cuda-13.2`   |

The tarball is fetched from
`https://developer.download.nvidia.com/compute/machine-learning/tensorrt/...`,
extracted to `/usr/local/`, and symlinked at `/usr/local/tensorrt`. The
installer appends

```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}
```

to your shell rc so `turboocr-server` finds `libnvinfer.so.10` at run
time.

!!! danger "LD_LIBRARY_PATH is mandatory"
    Without `/usr/local/tensorrt/lib` on `LD_LIBRARY_PATH` the server
    will fail to dlopen the TRT runtime at startup. The installer wires
    this into `~/.bashrc` / `~/.zshrc` automatically; if you bypass the
    installer, set it yourself.

!!! tip "sm_120 builder lib"
    TensorRT 10.15's `Builder` library has a hard sm_120 (Blackwell)
    dependency when JIT-building engines from ONNX on first launch. On
    older GPUs CUDA omits the kernels silently and you'll see
    `kernels not found` at engine-build time — drop
    `-DCMAKE_CUDA_ARCHITECTURES=120` to your host's compute capability
    (e.g. `89` for Ada).

## Configure & build

Direct CMake invocation, matching what `install_native.sh:160-163`
runs:

```bash
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTENSORRT_DIR=$HOME/TensorRT-10.15.1.29 \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc) --target turboocr-server
```

`TENSORRT_DIR` can also point to `/usr/local/tensorrt` (the installer's
default symlink). The default `target` builds everything including the
unit-test binary; restricting to `turboocr-server` shaves ~30 s on a
clean build.

## Smoke test

```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
./build/turboocr-server &      # native build binds port 8080 (the 8000→8080 nginx hop is Docker-only)
curl -fsS http://localhost:8080/health/ready
curl -X POST http://localhost:8080/ocr/raw \
     --data-binary @tests/test_data/png/receipt.png \
     -H 'Content-Type: image/png'
```

!!! note "First-start cost"
    First start spends ~90 s building TensorRT engines from the bundled
    ONNX files; engines are cached under `~/.cache/turbo-ocr/` so
    subsequent runs are instant. `/health/ready` returns `503 NOT_READY`
    during the build — the `curl -fsS` above fails fast on that 503, so retry
    it (or poll in a loop) until it returns 200.

## Build output

| Path | What |
|---|---|
| `build/turboocr-server` | HTTP + gRPC server (GPU build) |
| `build/turbo_ocr_tests` | Catch2 unit suite (always built) |
| `build/proto_gen/` | Generated `ocr.{pb,grpc.pb}.{h,cc}` stubs |
| `build/turbo_ocr_common.a`, `build/turbo_ocr_gpu.a` | Internal libs |

## CPU-only variant

For a no-GPU build pass `-DUSE_CPU_ONLY=ON` and target
`turboocr-cpu-server`; that build does **not** need TensorRT (only
ONNX Runtime 1.22.0, which CMake fetches on demand if not already in
`/usr/local/lib`).

```bash
cmake -B build_cpu -DUSE_CPU_ONLY=ON
cmake --build build_cpu -j$(nproc) --target turboocr-cpu-server
```

!!! info "See also"
    - [Build → Docker](docker.md) — production-ready images with the
      same dependency pins.
    - [Build → Models](models.md) — what `fetch_release_models.sh`
      lays down.
    - [Dev → Testing](../dev/testing.md) — how to exercise the binary
      after build.
