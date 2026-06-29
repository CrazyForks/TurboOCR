# Non-canonical experiment images

These Dockerfiles are **not** buildable from a clean checkout. Each chains
`FROM turboocr:vlm` or `FROM turboocr:pool` — prebuilt base images that have no
in-repo Dockerfile (there is no `Dockerfile.vlm`). They are incremental
patch artifacts that re-copy a handful of changed sources/headers onto an
already-built base and relink only `turboocr-server`. They exist to iterate
fast on a running experiment, not to produce a reproducible image.

| File | Base | What it patches |
|---|---|---|
| `Dockerfile.pool` | `turboocr:vlm` | global async VLM crop pool |
| `Dockerfile.parallel` | `turboocr:vlm` | formula\|\|table parallel dispatch + CJK VLM text re-rec |
| `Dockerfile.async` | `turboocr:pool` | async VLM dispatch + router threshold env vars |
| `Dockerfile.router` | `turboocr:pool` | env-tunable router confidence thresholds (`router_patch/`) |
| `Dockerfile.profile` | `turboocr:vlm` | `VLM_PROFILE=1` timing instrumentation (`vlm_*_instrumented.cpp`) |

The COPY paths in `Dockerfile.router` / `Dockerfile.profile` reference
`docker/experiments/...`, so any build must use the repo root as the build
context (`docker build -f docker/experiments/Dockerfile.router .`).

## Canonical images

The only images intended to build from a clean checkout and ship are:

- `../Dockerfile.gpu` — GPU (TensorRT) HTTP + gRPC server.
- `../Dockerfile.cpu` — CPU-only (ONNX Runtime) server.

Use `../docker-compose.yml` to run the GPU image.
