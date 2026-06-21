# Production-Readiness Hardening Plan — paddle-highspeed-cpp

**Branch:** `prod-readiness-hardening` (off `feat/v3-ppocrv6` @ `56b13412`)
**Policy:** all work commits to this branch; **never push**; no force/amend; no AI attribution in messages.
**Source of findings:** 9-dimension parallel review (request-handling, build/deploy, ops/observability,
correctness/tests, code-quality, endpoint verification) + 3 architecture reviews (modules/god-classes,
API/contract, concurrency/resource) + clang-tidy/cppcheck/lizard/cloc.

---

## 0. Constraints & validation strategy

| Constraint | Implication for this plan |
|---|---|
| 1× RTX 5090, **~29.6 GB held by a vLLM**, 2.5 GB free | GPU server (TRT engines + pool) **cannot run** until VRAM frees. CPU build is the validation workhorse this session. |
| CPU build (`paddle_cpu_server`) boots & serves all endpoints | Every transport/contract/config fix is CPU-validatable end-to-end now. |
| GPU-only paths (nvjpeg decode, TRT, CUDA streams, fork-after-CUDA) | Validate by **build + unit test + code review** now; gate full GPU runtime validation on a freed VRAM window (Phase G). |
| Shared machine | Check `nvidia-smi` + `uptime` before any GPU benchmark; never assume the GPU is idle. |

**Definition of "production-ready" (exit criteria):**
1. All 5 Critical blockers closed and validated.
2. CI runs build + C++ tests + CPU-server smoke + image build on every push; green = real coverage.
3. All High items closed or explicitly deferred with a written risk acceptance.
4. GPU runtime validation (Phase G) passes once VRAM is available: FUNSD F1 ≥ 0.90, sustained ≥ 10 pages/s.
5. Static-analysis baseline recorded (clang-tidy clean of cert-/bugprone- in app code; cppcheck dead-code pass complete excluding `third_party`).

### HARD CONSTRAINT (added after review feedback)

**The service is already deployed — no behavioral breaking changes.** Defaults, bind
address, response shapes, ports, and existing env semantics must keep working as-is. New
behavior is opt-in (additive env/flags) only.

**This is a vLLM-class inference server — auth, TLS, API versioning, and exposure are the
fronting gateway/proxy's job, NOT the server's.** Therefore:
- **C2 reduced** to non-breaking edge hardening only: keep default bind `0.0.0.0`; add
  `BIND_HOST` as an *optional* alias to restrict; add slow-loris timeouts in the bundled
  nginx; `/metrics` restriction is left to the fronting gateway (documented, not enforced).
  No in-server auth token, no refuse-to-boot gate. (Done; corrected in commit after Batch 1.)
- **H6 (API versioning) — DROPPED.** Versioning/routing belongs at the gateway.
- Re-audit every remaining item for breaking behavior before landing.

**Effort key:** S ≤ 0.5 day · M ≈ 1–2 days · L ≈ 3–5 days · XL > 1 week.
**Each task lands as its own commit** with a `[phase-id]` prefix so the branch history maps to this plan.

---

## Phase 1 — Critical ship-blockers (must close before any deploy)

### C1 — Remove `-march=native` from shipped images  · effort S · risk Low
- **Problem:** `CMakeLists.txt:43` defaults `NATIVE_ARCH=ON`; `Dockerfile.gpu:115` builds with defaults → image is tuned to the *build host* CPU. Deploying on a different/older CPU SIGILLs at startup. Hand-written AVX2 in `ctc_decode.cpp` compounds it.
- **Fix:**
  - Default `NATIVE_ARCH=OFF` in `CMakeLists.txt`.
  - Set an explicit portable baseline for release builds: `-march=x86-64-v3` (AVX2-era, safe for any 2015+ server) for x86, `-mcpu=neoverse-n1` (or `armv8.2-a`) for aarch64.
  - Guard the hand-written AVX2 in `ctc_decode.cpp` behind a runtime CPU-feature check **or** compile it as a separate TU with a documented minimum ISA and a scalar fallback.
  - Keep `NATIVE_ARCH=ON` as an opt-in for self-hosted single-machine builds.
- **Validate:** `objdump -d build/... | grep -c vfmadd` shows no AVX-512; build on one host, run on a VM with `-cpu Haswell` (or `qemu`/`taskset` ISA mask) without SIGILL. Add a CI job that builds with the baseline and runs `turbo_ocr_tests`.

### C2 — Lock down network exposure (auth + bind) · effort M · risk Medium
- **Problem:** No authn/authz on any endpoint; default bind `0.0.0.0` (`server_config.h:37`), gRPC `InsecureServerCredentials` (`grpc_service.h:1237`), `/metrics` leaks VRAM/topology. nginx only proxies — it does not bind-restrict 8080/50051.
- **Fix (defense in depth):**
  1. Default `host` to `127.0.0.1`; require an explicit `BIND_HOST=0.0.0.0` opt-in with a startup WARN that documents the trusted-network assumption.
  2. Entrypoint binds the app to loopback by default so only nginx (8000) is public.
  3. Add optional bearer-token / mTLS auth middleware, controlled by env (`API_AUTH_TOKEN`, `GRPC_TLS_CERT/KEY`). Off by default for backward compat but documented as **required for any non-loopback bind**; refuse to start with `BIND_HOST=0.0.0.0` and no auth unless `ALLOW_INSECURE_PUBLIC_BIND=1`.
  4. Move `/metrics` to an internal-only listener **or** add nginx `location /metrics { allow <cidr>; deny all; }` and document it.
- **Validate:** CPU server: `curl` from loopback works, from the host IP refuses by default; with auth on, missing/blank token → 401; `/metrics` not reachable through the public proxy. Integration test in `tests/integration`.

### C3 — Wire CI/CD and make the C++ test suite actually run · effort M · risk Low
- **Problem:** No `.github/`. No `enable_testing()`/`add_test()` anywhere → `run_all.py --suite cpp` runs `ctest`, finds nothing, **exits 0** (green while testing nothing). `tests/cpp/CMakeLists.txt` is stale and omits every migration test; the real target is the root `CMakeLists.txt:190` `turbo_ocr_tests`.
- **Fix:**
  - Add `include(CTest)` / `enable_testing()` and register `turbo_ocr_tests` via Catch2 `catch_discover_tests()` (or one `add_test`) in the root `CMakeLists.txt`.
  - Delete or sync `tests/cpp/CMakeLists.txt` (it builds a suite missing migration tests — a trap).
  - Add `.github/workflows/ci.yml`:
    - **build-cpu** job: configure with `-DNATIVE_ARCH=OFF -DBUILD_GPU=OFF`, build, `ctest --output-on-failure`.
    - **cpu-smoke** job: boot `paddle_cpu_server` on a free port with the tiny model, `curl` the endpoint matrix (reuse `tests/docker_endpoint_matrix.py`), assert 200s + error codes.
    - **docker-cpu** job: build `Dockerfile.cpu`.
    - **static** job: clang-tidy on changed files + cppcheck (excluding `third_party`).
  - Gate the model-bundle release (Phase H3) on CI green.
- **Validate:** CI red when a test fails (introduce a temporary failing assert to prove it); `ctest` lists > 0 tests locally.

### C4 — Per-request timeout / cancellation + slot recycling · effort M · risk Medium
- **Problem:** Every `dispatcher.submit().get()` (`pipeline_dispatcher.h:71`, `main.cpp:264`, `image_routes.cpp:124/339`, gRPC) blocks unbounded. A wedged `enqueueV3` permanently loses that pipeline slot; at pool 1–2 the server deadlocks while the queue fills to 503.
- **Fix:**
  - Add a per-submit deadline: `future.wait_for(REQUEST_TIMEOUT_MS)` at every `.get()` site → return 504 `INFERENCE_TIMEOUT` and free the WorkPool thread.
  - On timeout, mark the `GpuPipelineEntry` suspect; a watchdog rebuilds (`OcrPipeline` + stream) a wedged entry rather than leaking the slot forever.
  - `REQUEST_TIMEOUT_MS` env (default e.g. 30000), surfaced in `log_effective()`.
- **Validate:** unit test with an injected sleeping pipeline → request returns 504 within the deadline and the slot is reusable on the next request (no permanent capacity loss). CPU-validatable with a mock slow stage.

### C5 — Fix fork-after-CUDA-init in the PDF renderer · effort M · risk High
- **Problem:** `PdfRenderer` is constructed at `main.cpp:219`, **after** `make_pipeline_dispatcher` (`:206`) initializes CUDA/TRT/streams. `pdf_renderer.cpp:254` forks a process with a live CUDA context — undefined behavior; survives only because the child `execl`s `fastpdf2png` immediately (`:266`). Any allocator lock held across fork can deadlock the child.
- **Fix:**
  - Construct `PdfRenderer` (fork all daemons) **before** the first CUDA call — move it above `make_pipeline_dispatcher` in both `main.cpp` and `cpu_main.cpp`.
  - Alternatively/additionally switch the spawn to `posix_spawn` (no fork-time copy of the parent address space / locks).
  - Add a comment documenting the fork-before-CUDA invariant so it isn't reordered later.
- **Validate:** order-of-init review + CI build; on a freed-GPU window, stress the PDF path under concurrency (Phase G) and confirm no daemon hangs. ASAN/TSAN build of the CPU path through the renderer.

---

## Phase 2 — High (close before GA; the gRPC items share files, do them together)

### H1 — Coordinated gRPC graceful shutdown · effort S · risk Medium
- `main.cpp:375` calls `server->Shutdown()` with no deadline, *after* HTTP drains → inflight gRPC RPCs cut mid-response, no admission stop.
- **Fix:** trigger `grpc_handle.server->Shutdown(now + shutdown_grace_seconds)` inside `begin_graceful_shutdown`, in parallel with the WorkPool drain; stop admitting new RPCs first. Drain order: stop-accept (both protocols) → drain WorkPool + gRPC → join dispatcher → tear down renderer.
- **Validate:** send a long PDF RPC, SIGTERM mid-flight → RPC completes (or returns UNAVAILABLE cleanly) within grace; no truncated response.

### H2 — Offload the gRPC Health probe · effort S · risk Low
- `grpc_service.h:224` runs the real 48×48 GPU inference inline on the CQ poller thread (HTTP offloads it to the WorkPool). Probe storm → poller starvation.
- **Fix:** offload the gRPC probe to the WorkPool like HTTP, or have gRPC Health return only the 5s-cached verdict (never force a fresh GPU pass on the poller).
- **Validate:** hammer gRPC Health concurrently while serving OCR; pollers stay responsive.

### H3 — Remove gRPC per-page thread amplification · effort M · risk Medium
- gRPC PDF spawns one `std::async(launch::async)` OS thread per page (gated 64; `grpc_service.h:742`), each then calling `dispatcher.submit().get()` — thread churn + a backpressure model divergent from the clean HTTP path (`pdf_routes.cpp:649` enqueues futures directly).
- **Fix:** make the gRPC PDF path mirror HTTP — push page work onto the bounded dispatcher queue directly; delete the per-page `std::async` + `counting_semaphore`.
- **Validate:** thread count stays flat under a multi-page gRPC PDF burst (compare `ls /proc/<pid>/task | wc -l` before/after); backpressure → RESOURCE_EXHAUSTED, not unbounded threads.

### H4 — Quarantine sticky CUDA faults · effort M · risk Medium
- `trt_engine.cpp:223` maps `enqueueV3==false` to a plain error; a sticky fault (`cudaErrorIllegalAddress`, ECC) poisons the whole process context and subsequent requests serve corrupt results.
- **Fix:** after any `enqueueV3==false` / `CudaError`, check `cudaGetLastError()` for sticky codes; on a sticky fault, fail-fast the process (log + exit non-zero) so k8s restarts it, rather than serving garbage. Non-sticky → 5xx + slot recycle (ties into C4).
- **Validate:** unit test injecting a sticky-error code path → process exits; non-sticky → recoverable 5xx.

### H5 — Supply-chain: model bundle + PDFium pinning · effort M · risk Medium
- Whole deploy depends on one un-mirrored GitHub release (`fetch_release_models.sh:21`); cold-start self-heal breaks if it vanishes. PDFium pulled from `latest` moving tag, checksum optional (`install_pdfium.sh:43`). aarch64 ORT checksum skipped (`CMakeLists.txt:249`).
- **Fix:** script the model-release publish + `SHA256SUMS.txt` generation; document a `MODELS_RELEASE_URL` mirror env; pin a default `PDFIUM_RELEASE=chromium/NNNN` + ship its SHA256; pin the aarch64 ORT hash. Add `third_party/VERSIONS.md` (nlohmann 3.12.0, CLI11 2.4.2, ORT 1.22.0, clipper/wuffs/simdutf commits + URLs).
- **Validate:** clean-checkout build with no network except the pinned URLs; checksum mismatch → hard fail.

### H6 — API versioning · DROPPED
Versioning/routing is the fronting gateway's responsibility for a vLLM-class server; adding
`/v1` in-server provides no value the gateway can't, and aliasing risks breaking deployed
clients. Not implemented by design.

### H7 — Unify the response contract across HTTP/gRPC · effort L · risk Medium
- gRPC `json_bytes` mode (default, `server_config.h:71`) leaves `results[].text` empty — response meaning depends on a server env var the client can't see. PDF shapes drift (autorotate/inline-images only on HTTP); text-layer helpers duplicated (`pdf_routes.cpp:151` vs `grpc_service.h:634`).
- **Fix:** extract ONE serialization module both transports call; make the active gRPC response mode discoverable (return it in `HealthResponse`/metadata); document in-proto that `results` is empty under `json_bytes`; bring PDF feature parity (or explicitly declare gRPC's reduced PDF surface via a `/capabilities`).
- **Validate:** golden-file tests asserting HTTP JSON == gRPC `json_response` for the same input; capability list reflects reality across builds.

### H8 — Engine abstraction `IEngine` + collapse CPU/GPU twins · effort XL · risk High
- No `IEngine`; `TrtEngine`/`CpuEngine` unrelated, held concretely in detectors → a parallel `cpu_*` class tree (~1,500 LoC twin code) and triplicated `main` (`cpu_main` CCN 74).
- **Fix:** introduce `IEngine { infer(spans) -> spans }`; detectors/recognizers depend on the interface; collapse `cpu_*`/GPU twins behind one backend-parameterized class; extract a shared `ServerBootstrap` so the three `main`s become ~20-line shims.
- **Validate:** both builds produce identical OCR output on the fixture set; LoC drop measured; new unit tests drive the pipeline through a **mock** `IEngine` with no GPU.
- **Note:** large refactor — schedule after Critical/High behavioral fixes so it rebases onto a stable contract. Land incrementally (interface first, one stage at a time).

### H9 — Split god-function `RecognizePDF` · effort L · risk Medium
- CCN 87, 405 NLOC, header-only (`grpc_service.h`) → untestable + recompile bomb.
- **Fix:** move `grpc_service` impl to a `.cpp`; extract a transport-agnostic `PdfJob` orchestrator in the pipeline layer that both gRPC and HTTP `pdf_routes` call (also removes H7/H3 duplication).
- **Validate:** lizard CCN of the new units < 15; recompile time of includers drops; PdfJob unit-tested headless.

---

## Phase 3 — Medium (operability, hygiene, robustness)

### M1 — Saturation metrics · effort S
Export `turbo_ocr_workpool_queue_depth`, `_inflight`, dispatcher queue depth, and per-stage (det/rec/cls/layout) timing histograms. WorkPool already tracks `inflight_`/`queue_.size()` (`work_pool.h:114`) — just expose them. Validate: scrape `/metrics`, depth rises under load before 503s.

### M2 — Distinguish liveness vs readiness · effort S
gRPC Health maps only to readiness (GPU). Add a GPU-free liveness signal; gate only readiness on inference; add a consecutive-failure threshold / longer readiness cache so a *busy* GPU doesn't make k8s evict a healthy pod. Validate: saturate GPU → liveness stays up, readiness flaps without eviction.

### M3 — Validating env-config parser · effort S
46 unchecked `atoi`/`atof` (cert-err34): `GPU_CCL=foo` silently becomes 0. Route all env parsing through one helper that validates and WARN/refuses on malformed input (extend `env_int_strict`). Validate: bad env → clear startup error, not silent degradation.

### M4 — `crop_pool.cpp` leak + hung-future fix · effort S
Raw `new`/`delete` + unchecked `curl_multi_add_handle` (`:281`) → ctx leak + caller future never resolves. Wrap `HandleCtx` in `unique_ptr`; check the return; on failure `set_exception` + delete. Validate: inject add-handle failure → caller gets an error promptly, no leak (ASAN).

### M5 — nginx hardening · effort S
gRPC body path uncapped; slow-loris exposure (`proxy_request_buffering off`, no `client_body_timeout`). Add `client_body_timeout`/`send_timeout`, cap the gRPC path, reconsider request buffering for the bounded body sizes. Validate: slow-trickle client times out instead of holding a worker.

### M6 — Cross-build behavioral parity · effort M
`/profile` CPU-only, `/ocr/raw` two impls, `mode=auto_verified`→`auto` silently on CPU, default PDF DPI differs (100 vs `cfg.pdf_dpi`). Make route registration one shared backend-parameterized function; expose a `/capabilities` endpoint; align defaults or document divergence. (Folds into H7/H8.)

### M7 — Docker hygiene · effort M
Delete/move chained `turboocr:*` Dockerfiles (async/parallel/pool/router — unbuildable from clean checkout, no `Dockerfile.vlm`); fix `docker-compose.yml:62` (removed `paddle_grpc_server`); add `HEALTHCHECK` to GPU/CPU Dockerfiles; document `Dockerfile.gpu`/`.cpu` as the only canonical images. Validate: `docker build` each canonical image from a clean checkout succeeds; compose `up` healthy.

### M8 — Error-code registry · effort S
Codes duplicated as string literals across HTTP/gRPC/proto. Single `error_codes.h` table: code → (HTTP status, gRPC StatusCode, default message); both `error_response` and `grpc_error` consume it. Validate: rename a code in one place, both surfaces update.

### M9 — VRAM-aware pool sizing + daemon-crash recovery · effort M
Pool auto-size (`main.cpp:184`) keys off *total* VRAM, not measured per-pipeline footprint → OOM risk at warmup on smaller cards. Measure one warmed pipeline's resident VRAM, then `pool = min(cap, free/footprint × safety)`. Add daemon re-fork on `send_cmd` failure (`pdf_renderer.cpp:342`) guarded by the per-daemon mutex. Validate: small-VRAM sim sizes down; killing a daemon mid-traffic → it re-forks, capacity restored.

### M10 — Decouple GPU from transport · effort M
Routes construct raw `GpuImage` + thread `cudaStream` (`image_routes.cpp:95`). Hand bytes to a job submitter; keep GPU residency inside the pipeline. (Folds into H8/H9.)

### M11 — Repo-relative script paths · effort S
Hardcoded `/home/user/...` in `_build_int8_common.sh:16`, `_build_fp8_common.sh:21`, `build_rec_int8_qdq.sh:30`. Derive from repo-relative `.venv*` or require the env var. Validate: scripts run for another user/CI.

---

## Phase 4 — Low / polish

- L1 `/metrics` `cudaMemGetInfo` is device-global — add a HELP-string caveat (or per-process accounting) to avoid misleading leak hunts on shared GPUs.
- L2 Wire **ASAN + TSAN** builds into a periodic (nightly) CI job — highest-value safety investment given the concurrency; options already exist (`CMakeLists.txt:106`).
- L3 Strict-mode option to reject unknown query params / extra JSON fields (today a typo'd `?reading_oder=1` returns 200 with wrong results); document the default ignore-unknown policy as contract.
- L4 cert-err33: handle the ignored `dup2`/`write` returns in the forked child (`pdf_renderer.cpp:258`).
- L5 De-duplicate magic numbers (DPI bounds 50–600 at `pdf_routes.cpp:891/1104`; probe TTL 5000; dummy 48) into named constants.
- L6 Replace predictable request-id PRNG seeding if IDs ever gate anything (currently observability-only — fine).
- L7 Entrypoint: forward SIGTERM to the background nginx on shutdown (`entrypoint.sh:97`).

---

## Phase G — GPU runtime validation (gated on a freed VRAM window)

Cannot run now (vLLM holds 29.6/32 GB). When a window opens (vLLM stopped or scaled down — **operator action, not automated here**):
1. Build GPU target portable (`-DNATIVE_ARCH=OFF`), boot `paddle_highspeed_cpp` with the tiny/medium model.
2. Endpoint matrix over the **GPU** path (nvjpeg decode + TRT) — the one leg unverified this session.
3. Accuracy gate: FUNSD F1 ≥ 0.90 (per prior v6 parity).
4. Throughput gate: sustained ≥ 10 pages/s (hard floor); record pages/s at pool 5.
5. Soak: C4 timeout + C5 fork-order + H3 thread-count under a concurrent PDF burst; watch for slot leaks / daemon hangs / VRAM growth.
6. ASAN/TSAN pass over the CPU path through the renderer in the meantime (no GPU needed).

---

## Phase H — Static-analysis baseline & gates

- H1 Re-run **cppcheck** dead-code pass with `-i third_party` (and a longer budget) for a clean `unusedFunction` verdict — this session's run timed out and was drowned by vendored `wuffs` `v_*` noise.
- H2 Run **scan-build** (clang static analyzer) over a CPU build for path-sensitive null-deref/leak findings; triage into this plan.
- H3 Run **valgrind** memcheck + helgrind against `turbo_ocr_tests` and a short `paddle_cpu_server` run with a CUDA/ORT suppression file.
- H4 Make clang-tidy a CI gate scoped to app headers (`HeaderFilterRegex` already set) — fail on new cert-/bugprone- findings only (the 99% stylistic noise stays non-blocking).

---

## Sequencing & rough timeline

| Wave | Items | Gated by | Effort |
|---|---|---|---|
| **Wave 1 (week 1)** | C1, C2, C3, C4, C5 + M1, M3, M4, M11 | none — all CPU-validatable | ~1.5 wk |
| **Wave 2 (week 2)** | H1–H7, M2, M5, M7, M8 | Wave 1 (C3 CI gates the rest) | ~2 wk |
| **Wave 3 (week 3+)** | H8, H9, M6, M9, M10 (architecture) | stable contract from Wave 2 | ~2–3 wk |
| **Wave 4 (ongoing)** | Phase 4 Low, Phase H static gates | parallel | ~1 wk |
| **Phase G** | GPU runtime validation | **freed VRAM window** | ~0.5 wk when unblocked |

**Open decision (blocks nothing, but confirm early):** `kDefaultModel` now ships **"tiny"** (`model_catalog.h:56`); the original migration intent was **"medium"**, and tiny's accuracy gate is report-only. Confirm tiny is the intended v3 default (throughput) vs a regression of the accuracy intent — affects the Phase G accuracy gate target.

---

## Decision log / risk acceptances
_(append as we go — each deferred High needs a one-line written acceptance here)_

- **H8 scoped to ServerBootstrap; cpu_*/GPU twin-collapse + bare IEngine REJECTED (perf).**
  Investigation showed the GPU and CPU detector/recognizer/layout classes are
  *different algorithms*, not one algorithm over two engine backends: GPU
  `PaddleDet` is ~628 LoC of CUDA CCL kernels + device buffers + streams; CPU
  `CpuPaddleDet` is ~112 LoC of OpenCV `findContours`. `TrtEngine`
  (device-buffer/stream/graph-capture API) and `CpuEngine` (host float-buffer
  API) likewise don't share a non-leaky interface. Merging the twins behind an
  `IEngine` would force the GPU hot path through host round-trips / CPU contours
  and regress the hand-won >400 pages/s + FUNSD F1 — a breaking change to a
  deployed perf-critical server. So H8 delivers its safe, real value
  (ServerBootstrap: de-triplicate the 3 mains' startup, the CCN-74 cpu_main the
  review flagged) and the twin-collapse is documented-rejected here rather than
  shipped as a regression. M10 (move GpuImage/stream construction out of the GPU
  routes) is likewise left as-is: the route-level `GpuImage` is a deliberate
  zero-copy optimization, not a defect, and relocating it risks the hot path for
  no functional gain.
- **REQUEST_TIMEOUT_MS defaults to 0 (DISABLED) — opt-in.** Review Round 1 flagged
  a 30000ms default as an opt-OUT behavior change (previously-unbounded waits would
  start returning 504). To stay strictly non-breaking on the deployed server, the
  default is 0 (unbounded, the pre-hardening behavior); operators opt in with
  `REQUEST_TIMEOUT_MS=30000` for wedged-slot recovery. PDF pages are unaffected.
- **Phase H static gates run + triaged.** cppcheck is CI-wired (+ ASAN/TSAN nightly).
  scan-build (H2) + valgrind memcheck (H3) were run manually this pass: valgrind
  found + fixed one real OOB read in layout flatten_descendants (now 0 errors/0
  leaks); scan-build reported 0 real production bugs (60 findings = dead-stores /
  vendored simdutf / Catch2-REQUIRE FPs); cppcheck 0 error-severity. A nightly
  scan-build/valgrind CI job (with a CUDA/ORT suppression file) is the follow-up.
- **M1 per-stage histogram removed (dead code).** Built + serialized but never
  populated; accurate GPU per-stage timing needs CUDA events (out of scope), so the
  dead machinery was removed. The M1 saturation gauges (workpool queue depth /
  inflight / dispatcher queue depth) are wired and live — that is M1's real value.
