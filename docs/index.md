# TurboOCR

!!! abstract "TL;DR"
    Multi-stream OCR pipeline that holds **6 ms p50** on RTX 5090 for the text-only short-circuit path, with opt-in layout / table / formula stages that fan out on dedicated CUDA streams.

<div class="grid phc-stats">
<div class="phc-stat">
  <div class="phc-stat-num">6.0 ms</div>
  <div class="phc-stat-label">Text-only p50 latency</div>
  <div class="phc-stat-foot">RTX 5090, TRT 10.15.1.29</div>
</div>
<div class="phc-stat">
  <div class="phc-stat-num">&asymp; 49.0</div>
  <div class="phc-stat-label">OmniDocBench v1.7 composite</div>
  <div class="phc-stat-foot">full 1651 pages, 2026-05-17 — pre formula-fix, re-measure pending</div>
</div>
<div class="phc-stat">
  <div class="phc-stat-num">0.568</div>
  <div class="phc-stat-label">Table TEDS (all)</div>
  <div class="phc-stat-foot">full 1651 pages; +0.541 vs May 14 baseline</div>
</div>
<div class="phc-stat">
  <div class="phc-stat-num">0.079</div>
  <div class="phc-stat-label">text_block Edit_dist (English)</div>
  <div class="phc-stat-foot">per-language slice</div>
</div>
</div>

<div class="grid cards" markdown>

-   __[Architecture](architecture/overview.md)__

    ---

    Five CUDA streams, six events, one CPU router. The 270 ms invariant and
    why the text-only GPU instruction stream is byte-identical with or
    without the new stages.

-   __[Models](models/detection.md)__

    ---

    Per-model cards: detection, classification, layout, recognition,
    table, and the in-process PP-FormulaNet-S formula stage. Input
    shapes, dynamic profiles, latency budgets.

-   __[Benchmarks](benchmarks/omnidocbench.md)__

    ---

    OmniDocBench v1.7 composite 49.0 on 1651 pages. Latency diary,
    sweep history, and the regression detector that pins the
    text-only invariant.

-   __[API](api/http.md)__

    ---

    HTTP endpoints (`/ocr/raw`, `/ocr/pixels`, `/ocr/batch`,
    `/ocr/pdf`, `/health/*`) plus the gRPC surface generated from
    `proto/turbo_ocr.proto`.

</div>

## Three-tier model fan-out

```mermaid
flowchart LR
  Upload[upload_image] --> Det[PaddleDet]
  Det --> Cls[PaddleCls]
  Det -. det_only_event_ .-> Layout[PaddleLayout]
  Cls --> Router{CuaRouter::classify}
  Layout --> Router
  Router -- text_indices --> Rec[PaddleRec<br/>rec_stream_]
  Router -- table_layout_ids --> Table[TableStage<br/>table_stream_]
  Router -- formula_layout_ids --> Formula[FormulaNet<br/>formula_stream_]
  Rec --> Assemble[result + reading_order]
  Table --> Assemble
  Formula --> Assemble
```

Detection + angle-cls feed the router; layout decides which downstream
pipeline owns each cell; rec, table, and formula run concurrently on
three named streams. See [Pipeline](architecture/pipeline.md) for the
node-by-node walk.

## Text-only short-circuit

```mermaid
flowchart LR
  Run["run_with_layout<br/>want_layout=false"] --> Upload[upload_image]
  Upload --> Det[PaddleDet]
  Det --> Cls[PaddleCls]
  Cls --> Rec[PaddleRec on rec_stream_]
  Rec --> Filter[drop_score filter]
  Filter --> Dispatch[dispatch_router_]
  Dispatch --> Guard{"router_ == nullptr<br/>OR layout.empty"}
  Guard -- yes, both hold --> Done[return — zero CUDA calls]
  Guard -. never reached on text-only .-> TableFormula[table_stream_ / formula_stream_]
```

Three guards make the short-circuit fire before any new CUDA API call —
`!router_` and `out.layout.empty()` at `src/pipeline/ocr_pipeline.cpp:436-437`,
and the post-classify `!has_table && !has_formula` at
`ocr_pipeline.cpp:453`. The GPU instruction stream on text-only pages is
bit-identical to the pre-router code. See
[CUDA Streams](architecture/cuda-streams.md) for the invariants.

## Table + formula opt-in

```mermaid
flowchart LR
  Init["OcrPipeline::init"] --> Loaded["text-only ready<br/>caller + rec_stream_"]
  Loaded -- "load_layout_model" --> WithLayout["+ layout_stream_<br/>+ det_only_event_"]
  WithLayout -- "load_router_models<br/>table paths set" --> WithTable["+ table_stream_<br/>+ table_done_event_"]
  WithLayout -- "load_router_models<br/>formula paths set" --> WithFormula["+ formula_stream_<br/>+ formula_done_event_"]
  WithTable --> Mixed[mixed-page pipeline]
  WithFormula --> Mixed
```

Streams and events for table / formula are **lazy-allocated inside
`load_router_models()`** — a pipeline that never calls it pays zero new
CUDA resources. Verifiable at
`src/pipeline/ocr_pipeline.cpp:167-184` (table) and
`ocr_pipeline.cpp:202-205` (formula).

## Headline numbers

| Metric | Test set | Value | Source |
|---|---|---:|---|
| Composite Overall (OmniDocBench v1.7) | full 1651 | **≈ 49.0** | `omnidocbench/result/md_quick_match_metric_result.json` (regen 2026-05-17, **pre formula-fix** — re-measure pending) |
| text_block Edit_dist — English | full 1651 | **0.079** | same file, per-language slice |
| table TEDS (.all) | full 1651 | **0.568** | same file |
| formula CDM | 125-doc table/formula subset | **0.805** | in-process PP-FormulaNet-S, FAST decoder — see [resources](resources_speed_accuracy.md) |
| table TEDS | 125-doc table/formula subset | **0.773** | same subset run |
| text-only p50 latency | — | **6.0 ms** | internal engineering notes (8 sweeps, RTX 5090, TRT 10.15.1.29) |

!!! note "Formula CDM is fixed; full-1651 re-measure pending"
    The earlier formula CDM 0.063 "regression" was an integration bug, **now resolved**: the in-process
    PP-FormulaNet-S stage (FAST decoder) scores **CDM 0.805** on the 125-doc table/formula subset. The
    ≈ 49.0 composite above is the **full-1651** run from 2026-05-17 that predates the fix and still embeds
    the broken-formula 0.063, so it is a floor pending a re-measure with the fixed stage. The two test
    sets are not directly comparable — the 125-doc cut is a deterministic, table/formula-heavy stratified
    subset; the full-1651 numbers are the whole OmniDocBench set.

Full per-language, per-layout, and per-attribute breakdowns live on the
[OmniDocBench results](benchmarks/omnidocbench.md) page; the per-fixture
latency sweep history is on the [Latency](benchmarks/latency.md) page.

The 6 ms p50 is the load-bearing constraint the entire architecture is
shaped around. The original 270 ms target turned out to be ~45×
conservative on Blackwell with TRT 10.15.

!!! info "See also"
    - [Architecture overview](architecture/overview.md) — design philosophy + the 270 ms invariant.
    - [Model interactions](models/interactions.md) — per-page life cycle sequence diagrams.
    - [OmniDocBench results](benchmarks/omnidocbench.md) — full per-language / per-layout / per-attribute tables.
    - [HTTP API](api/http.md) — request shapes, JSON response, curl examples.
