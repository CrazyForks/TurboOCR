# Testing

!!! abstract "TL;DR"
    Tests live in two places: **Catch2 C++ unit tests**
    (`build/turbo_ocr_tests`, built unconditionally) and a **Python
    harness** under `tests/` (driven by `python tests/run_all.py`).
    Performance work goes through the `scripts/bench_*.sh` family;
    CUA-router regressions are caught by `scripts/bench_cua_loop.sh`.

See also `tests/TESTING_GUIDE.md` for the canonical pre-merge
checklist.

## C++ unit tests (Catch2)

A single binary, `turbo_ocr_tests`, is built unconditionally alongside
the server. Sources are listed in `CMakeLists.txt:93-108`:

| Test | Covers |
|---|---|
| `test_box`, `test_perspective`, `test_xy_cut` | Geometry primitives |
| `test_serialization` | JSON envelope shape, escape rules, blocks |
| `test_ctc_decode`, `test_det_postprocess` | Recognizer / detector postprocess |
| `test_image_dims` | Pre-decode header sniff and dim guard |
| `test_logger_ratelimit` | Structured logger throttle |
| `test_encoding` | UTF-8 round-trip via simdutf |
| `test_language_paths`, `test_server_config` | Config / CLI plumbing |
| `test_pdf_renderer_liveness` | fastpdf2png renderer smoke |

### Run

```bash
cmake --build build --target turbo_ocr_tests
./build/turbo_ocr_tests
```

Catch2 tag filters work as usual:

```bash
./build/turbo_ocr_tests "[serialization]"
./build/turbo_ocr_tests --list-tests
```

## Extra C++ test sources

!!! note "Not wired by default"
    A handful of additional C++ tests live under
    `tests/{router,table,formula,lang_cls,pipeline}/` and are
    intentionally **not** linked into `turbo_ocr_tests` — they pull in
    the larger GPU stack (TensorRT, CUDA) or rely on baked test
    fixtures. Add them to the `add_executable(turbo_ocr_tests ...)`
    block when you need to run them, or build a one-off binary linking
    `turbo_ocr_common` / `turbo_ocr_router` / `turbo_ocr_table` /
    `turbo_ocr_formula` as appropriate.

Sources:

- `tests/router/test_cua_router.cpp`
- `tests/table/test_html_reconstruct.cpp` (+ `_with_formulas`,
  `_aabb`, `_cell_matcher`)
- `tests/table/test_slanext_dict.cpp`,
  `test_slanext_postprocess.cpp`, `test_slanext_postprocess_aabb.cpp`,
  `test_cell_matcher.cpp`, `test_html_reconstruct.cpp`, `test_otsl_to_html.cpp`
- `tests/formula/test_formula_tokenizer.cpp`,
  `test_formulanet_ar_loop.cpp`
- `tests/lang_cls/test_script_id.cpp`
- `tests/pipeline/test_multi_rec.cpp`

## Standalone CPU-only Catch2 build

`tests/cpp/CMakeLists.txt` is a self-contained project that builds the
same Catch2 suite **without** CUDA / TensorRT — useful on CI runners
that don't have a GPU:

```bash
cmake -S tests/cpp -B tests/cpp/build
cmake --build tests/cpp/build -j
./tests/cpp/build/turbo_ocr_tests
```

## Python suites

`python tests/run_all.py` is the master driver.

| Suite | Path | Default? |
|---|---|---|
| `unit` | `tests/unit/` | yes |
| `integration` | `tests/integration/` | yes (needs `OCR_SERVER_URL`) |
| `regression` | `tests/regression/` | yes |
| `accuracy` | `tests/accuracy/` | yes |
| `cpp` | `tests/cpp/` | opt-in |
| `stress` | `tests/stress/` | opt-in (60 s soak) |
| `benchmark` | `tests/benchmark/` | opt-in |

### Run

=== "default"

    ```bash
    # unit + integration + regression + accuracy
    python tests/run_all.py
    ```

=== "one suite"

    ```bash
    python tests/run_all.py --suite unit
    python tests/run_all.py --suite benchmark
    ```

=== "all"

    ```bash
    python tests/run_all.py --suite all
    ```

### Environment

| Var | Default | What |
|---|---|---|
| `OCR_SERVER_URL` | `http://localhost:8000` | HTTP base used by integration / accuracy / regression. |
| `OCR_GRPC_TARGET` | `localhost:50051` | gRPC target for the same suites. |

`pytest.ini` defines markers `stress`, `benchmark`, `accuracy`,
`layout`. The `layout` marker auto-skips when the server reports
layout-disabled.

## Bench scripts

Standalone shell + Python harnesses for performance work:

| Script | Purpose |
|---|---|
| `scripts/bench_latency.sh`     | `hey -n 200 -c 1` sequential latency p50/p95/p99 over `/ocr/raw`. Image arg defaults to `tests/test_data/png/receipt.png`. |
| `scripts/bench_throughput.sh`  | `hey` at configurable `-c` for req/s sweeps. |
| `scripts/bench_full.sh`        | Five-concurrency sweep (`c=1,4,8,16,32`) across the fixture set; the canonical "post-change full run". |
| `scripts/bench_cua_loop.sh`    | Periodic CUA-router benchmark: health-probes `/health/ready`, then invokes `tests/benchmark/bench_cua_router.py` against the three scenarios (`text_only`, `formula_heavy`, `table_heavy`). |
| `tests/benchmark/bench_cua_router.py` | Orchestrator for the three scenarios; writes a schema-versioned JSON report under `$CUA_BENCH_OUT_DIR` (default `/tmp/cua_bench`). |
| `tests/benchmark/bench_latency.py` | Python equivalent of the shell sweep, used inside `run_all.py --suite benchmark`. |
| `tests/benchmark/bench_matrix.py` | Cross-product of fixtures × concurrency — primary regression input. |

!!! tip "bench_cua_loop exit codes"
    Per `.claude/plans/06_benchmark_harness.md §6`:
    `0 = PASS`, `1 = INFRA`, `2 = server-down`,
    `3 = ALERT`, `4 = HALT`. Anything non-zero blocks the merge.

!!! warning "hey on PATH"
    All `bench_*.sh` scripts expect `hey` on `$PATH` (or at
    `~/go/bin/hey`). Install with `go install
    github.com/rakyll/hey@latest`.

## CI gating

The pre-merge gate runs three stages:

1. `cmake --build build --target turbo_ocr_tests && ./build/turbo_ocr_tests`
2. `python tests/run_all.py` (the four default suites).
3. `scripts/bench_cua_loop.sh` once against the running server — the
   regression detector exits non-zero if any of the three scenarios
   regresses past the rolling-baseline threshold.

!!! info "See also"
    - [Dev → Plan history](plan-history.md) — `06_benchmark_harness.md`
      and the diary that drives the bench cadence.
    - [Benchmarks → Latency](../benchmarks/latency.md) — what the
      shell scripts measure.
    - [API → HTTP](../api/http.md) — the surface most tests hit.
