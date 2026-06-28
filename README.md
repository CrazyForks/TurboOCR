<p align="center">
  <img src="tests/benchmark/comparison/images/banner.png" alt="TurboOCR — the fastest GPU document parser." width="100%">
</p>

<p align="center">
  <strong>The fastest GPU document parser — OCR · layout · tables · formulas → Markdown, at 200–556 images/s on one GPU.</strong><br>
  C++ / CUDA / TensorRT / PP-OCRv6 &mdash; Linux + NVIDIA GPU
</p>

<h3 align="center">🎉 v3.0 — now powered by PP-OCRv6</h3>
<p align="center">
  <sub>New <code>medium</code> / <code>small</code> / <code>tiny</code> tiers · higher accuracy · faster defaults · <a href="#upgrading-to-v3-breaking-changes">breaking changes ↓</a></sub>
</p>

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ Star TurboOCR on GitHub</strong></a> — it helps others (and agents) find it.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/throughput-up_to_556_img%2Fs-blue?style=flat-square&logo=speedtest&logoColor=white" alt="up to 556 img/s">
  <a href="https://turboocr.com"><img src="https://img.shields.io/badge/website-turboocr.com-3B82F6?style=flat-square&logo=googlechrome&logoColor=white" alt="turboocr.com"></a>
  <a href="https://github.com/aiptimizer/TurboOCR/releases/latest"><img src="https://img.shields.io/github/v/release/aiptimizer/TurboOCR?style=flat-square&logo=github&logoColor=white" alt="Release"></a>
  <a href="https://ghcr.io/aiptimizer/turboocr"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"></a>
  <img src="https://img.shields.io/badge/C%2B%2B20-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-10.16-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10.16">
  <img src="https://img.shields.io/badge/gRPC-4285F4?style=flat-square&logo=google&logoColor=white" alt="gRPC">
  <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/badge/PP--OCRv6-PaddleOCR-0053D6?style=flat-square&logo=paddlepaddle&logoColor=white" alt="PaddleOCR"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#models">Models</a> &middot;
  <a href="#upgrading-to-v3-breaking-changes">v3 changes</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="https://aiptimizer.github.io/TurboOCR/">Docs</a>
</p>

---

An extremely fast GPU **document parser** — not just OCR. PP-OCRv6 detection +
recognition, plus layout, tables (→ HTML), formulas (→ LaTeX) and reading-order
**Markdown**, the whole pipeline on a single multi-stream CUDA/TensorRT engine,
locally (no VLM), behind HTTP and gRPC. It turns documents into structured data
at OCR speed — **200–556 images/s on one GPU**, where VLM document parsers like
PaddleOCR-VL run ~1–2 pages/s. On forms and receipts it is both the most accurate
open engine and 15–90× faster than classic OCR engines.

- 🚀 **Up to 556 img/s** (receipts) / **481 img/s** (forms) on one RTX 5090, fastest by default
- 🎯 **Most accurate on forms & receipts** &mdash; beats PaddleOCR-VL, PaddleOCR-Python, RapidOCR, EasyOCR and Tesseract ([benchmarks](#benchmarks))
- 🧠 **PP-OCRv6** &mdash; one model covers Latin + Chinese + Japanese; pick `tiny` (default) / `small` / `medium`
- 🌐 **More scripts** &mdash; Arabic, Cyrillic, Korean, Thai, Greek via retained PP-OCRv5 recognizers
- 📄 **PDF native** &mdash; pages rendered and OCR'd in parallel, optional page-image export & auto-rotation
- 🧩 **Layout + reading order** &mdash; PP-DocLayoutV3 (25 classes) and class-aware XY-cut, opt-in per request
- 🔢 **Tables & formulas** &mdash; opt-in SLANet+ table → HTML and PP-FormulaNet-S formula → LaTeX, emitted alongside the text ([how to enable](#tables--formulas))
- 🐳 **One-line Docker deploy** with TensorRT engines auto-built on first start, **Prometheus** metrics on `/metrics`

Full documentation: **<https://aiptimizer.github.io/TurboOCR/>**

---

## Quick Start

**Requirements:** Linux, NVIDIA driver 595+, Turing or newer GPU (RTX 20-series / GTX 16-series+).

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  ghcr.io/aiptimizer/turboocr:latest
```

First startup builds TensorRT engines from ONNX (~90s); the named volume caches
them for instant restarts. nginx (port 8000) reverse-proxies to the server.

```bash
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

```json
{"results": [{"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}]}
```

→ [Docker & deployment](https://aiptimizer.github.io/TurboOCR/build/docker/) · [Build from source](https://aiptimizer.github.io/TurboOCR/build/native/)

---

## Benchmarks

Like-for-like against the common OCR engines on a single RTX 5090. Two comparisons:
**whole-page OCR on English forms & receipts** (every engine), and **full-pipeline parsing
of complex English documents** — text, formulas and tables scored together on OmniDocBench.

### Forms & receipts — English (whole-page OCR)

Same images, same word-F1 metric for every engine; FUNSD (English forms) and CORD (English receipts), 50 pages each. Word-F1 = lowercased ≥2-char token overlap.

![OCR accuracy on forms and receipts](tests/benchmark/comparison/images/compare_accuracy.png)
![Throughput on forms and receipts](tests/benchmark/comparison/images/compare_throughput.png)

| Engine | FUNSD F1 | CORD F1 | Speed (FUNSD) |
|---|---:|---:|---:|
| **TurboOCR-medium** | **92.3%** | **93.4%** | 89 img/s |
| TurboOCR-small | 90.8% | 92.8% | 234 img/s |
| TurboOCR-tiny *(default)* | 85.4% | 88.9% | **481 img/s** |
| TurboOCR-v5 *(legacy)* | 90.2% | 91.8% | 249 img/s |
| PaddleOCR-VL-1.6 | 91.6% | 89.4% | 5 img/s |
| PaddleOCR PP-OCRv5 (Python) | 86.6% | 86.4% | 6 img/s |
| RapidOCR (GPU) | 69.1% | 82.6% | 2 img/s |
| EasyOCR | 59.8% | 67.3% | 3 img/s |
| Tesseract | 62.3% | 38.2% | 2 img/s |

TurboOCR has the best accuracy **and** is 15–90× faster than every other engine on forms and receipts.

### Complex documents — English (full pipeline, all metrics)

Papers and books need layout-aware parsing, so each pipeline is run end-to-end
(layout → region recognition → reading order) and scored by the **official OmniDocBench
scorer** on **English** documents (125 docs · 67 tables · 66 formulas). Every metric —
text, formula and table — is measured, so the Latin-only v5 path is a fair comparison and
is included.

| Pipeline | Text | Formula CDM | Table TEDS | Overall | Speed |
|---|---:|---:|---:|---:|---:|
| PaddleOCR-VL-1.6 | 97.1% | 0.973 | 0.915 | 0.953 | 0.94 pg/s |
| TurboOCR-medium | 95.6% | 0.917 | 0.819 | 0.897 | 35 pg/s |
| TurboOCR-small | 95.6% | 0.917 | 0.819 | 0.897 | 70 pg/s |
| TurboOCR-tiny *(default)* | 95.4% | 0.917 | 0.819 | 0.897 | **108 pg/s** |
| TurboOCR-v5 *(legacy)* | 95.8% | 0.917 | 0.819 | 0.898 | 108 pg/s |

Text = 1 − text-block edit distance; Table TEDS = structure-only; Overall = mean of the three.
Formula CDM and Table TEDS are **identical across the TurboOCR rows** — every tier (and v5)
shares the same PP-FormulaNet-S and SLANet-Plus stages; only the text recognizer differs. On
English text TurboOCR lands within **~1.5 points** of PaddleOCR-VL while running **35–115×
faster**; the rest of the Overall gap is table/formula recognition, where the VLM leads.

→ Full tables, metric definitions, languages, and how each pipeline is run: [Engine comparison](https://aiptimizer.github.io/TurboOCR/benchmarks/comparison/)

---

## Models

The pipeline is a small stack of specialised models, not one monolith. Text
detection + recognition + orientation always run; layout, table, and formula are
opt-in and only load when configured. Each stage links to its own model page.

| Stage | Model / arch | Size | Selected by | Docs |
|---|---|---:|---|---|
| **Text detection** | PP-OCRv6 det (DB, three tiers) | 1.7 / 9.4 / 59 MB | `OCR_MODEL` tier (`tiny`/`small`/`medium`) | [detection](docs/models/detection.md) |
| **Text recognition** | PP-OCRv6 rec (CRNN + CTC, Latin + Chinese + Japanese) | 4.3 / 20 / 73 MB | `OCR_MODEL` tier — default `tiny` | [recognition](docs/models/recognition.md) · [selection](docs/models/selection.md) |
| **Orientation** | PP-LCNet textline angle classifier | ~1 MB | always on (runs only on vertical lines) | [classification](docs/models/classification.md) |
| **Layout** | PP-DocLayoutV3 (RT-DETR-L, 25 classes) | ~124 MB | per request via `?layout=1`; disable with `DISABLE_LAYOUT=1` | [layout](docs/models/layout.md) |
| **Table → HTML** | SLANet-Plus (TRT FP16 CNN encoder + hand-written C++ GRU decoder) | ~5 MB | `TABLE_BACKEND=slanext` *(default backend)* + encoder paths | [table](docs/models/table.md) |
| **Formula → LaTeX** | PP-FormulaNet-S, in-process pure-C++ (ORT-CUDA-13, no Python) | ~294 MB | `FORMULA_BACKEND=ppformulanet_s` | [formula](docs/models/formula.md) |

The three OCR tiers (`tiny`/`small`/`medium`) all cover the same Latin + Chinese +
Japanese scripts via `OCR_MODEL` — they trade accuracy for speed, not language coverage:
`tiny` (default) for max throughput, `small` for a balance, `medium` for best accuracy
(per-tier FUNSD F1 + throughput are in the forms benchmark above). Other scripts use
retained PP-OCRv5 recognizers, also via `OCR_MODEL`: `arabic`, `eslav` (Cyrillic),
`korean`, `thai`, `greek`.

The table/formula stages always run **locally** in C++ by default (SLANet-Plus, and
PP-FormulaNet-S on ORT-CUDA-13).

→ [Model selection guide](https://aiptimizer.github.io/TurboOCR/models/selection/)

---

## Tables & formulas

Table and formula recognition are **opt-in**: the router only loads them when a
backend is configured at startup, so the default text path is untouched. Once a
backend is set, run any request with `layout` enabled and the response gains
`tables` (HTML + cell quads) and/or `formulas` (LaTeX) arrays.

| Capability | Enable at startup | Recognizer |
|---|---|---|
| Formula → LaTeX | `FORMULA_BACKEND=ppformulanet_s` | PP-FormulaNet-S |
| Table → HTML | `TABLE_BACKEND=slanext` (+ SLANet-Plus model paths) | SLANet-Plus |

```bash
docker run --gpus all -p 8000:8000 \
  -e FORMULA_BACKEND=ppformulanet_s -e TABLE_BACKEND=slanext \
  -v trt-cache:/home/ocr/.cache/turbo-ocr ghcr.io/aiptimizer/turboocr:latest

curl -X POST "http://localhost:8000/ocr/raw?layout=1" \
  --data-binary @paper.png -H "Content-Type: image/png"
```

→ [Tables](https://aiptimizer.github.io/TurboOCR/models/table/) · [Formulas](https://aiptimizer.github.io/TurboOCR/models/formula/)

---

## Running the legacy PP-OCRv5 models

The previous-generation **PP-OCRv5** detection + recognition models are retained
alongside the v6 default (`models/det_v5.onnx`, `models/rec_v5.onnx`,
`models/keys_v5.txt`). Select them by pointing the three per-stage path overrides
at the v5 files — everything else (layout, table, formula, orientation) is
unchanged, so it is a drop-in swap of just the text detector + recognizer:

```bash
DET_ONNX=models/det_v5.onnx \
REC_ONNX=models/rec_v5.onnx \
REC_DICT=models/keys_v5.txt \
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/paddle_highspeed_cpp
```

`DET_ONNX` / `REC_ONNX` / `REC_DICT` override `OCR_MODEL` per stage; the TensorRT
engine cache rebuilds for the v5 ONNX on first start (clear `~/.cache/turbo-ocr`
if v6 engines were cached under the default model names).

**Coverage caveat — Latin only.** The retained PP-OCRv5 recognizer dictionary
(`keys_v5.txt`) is 836 characters with **no CJK**. On English forms/receipts v5 is
within ~1.5–2 points of PP-OCRv6-medium (measured this release: FUNSD 90.3% vs
91.9%, CORD 91.7% vs 93.4% word-F1), but on the EN+ZH **OmniDocBench-125** set it
scores far lower (**52.7% vs 91.0%** text accuracy) because it cannot read the
Chinese pages. Use the v6 default for mixed-script documents; the v5 path is for
Latin-only workloads or direct A/B comparison.

---

## Upgrading to v3 (breaking changes)

v3 moves the default engine from PP-OCRv5 to **PP-OCRv6**. The changes since v2.3
sort into three buckets — only the first needs a config change.

**Breaking / config-incompatible** (action required):

- **`OCR_SERVER` removed.** PP-OCRv6 covers Latin + Chinese + Japanese in one model, so the separate Chinese-server recognizer toggle is gone. Non-Latin scripts (Arabic, Cyrillic, Korean, Thai, Greek) are served by retained PP-OCRv5 recognizers, selected via `OCR_MODEL`.
- **Clear the TensorRT engine cache on upgrade.** PP-OCRv6 ships new det/rec ONNX, so cached v5 engines must rebuild — wipe `~/.cache/turbo-ocr` (or the mounted `trt-cache` volume) once.
- **Sidecar-specific formula env is gone.** Only relevant if you set `PPFNS_SIDECAR_SCRIPT` / `PPFNS_SOCK` or launched `scripts/ppformulanet_s_sidecar.py` directly — all no longer exist (the experimental `TURBO_OCR_TRTLLM_DEBUG` is gone too). Plain `FORMULA_BACKEND=ppformulanet_s` users are unaffected (see *Additive / transparent* below).
- **Default `FORMULA_BACKEND` is now `ppformulanet_s`** (was `formulanet`), and the old `formulanet` backend was removed — `FORMULA_BACKEND=formulanet` now fails loudly at boot instead of running. Use `ppformulanet_s` (default, English/Latin), `ppformulanet_plus_m` (Chinese-capable), or `vlm`.
- **Table backend overhauled — old per-table-model env removed.** The wired/wireless table router and the `table_cls` classifier are gone; tables now run through a single SLANet-Plus structure model. `TABLE_CLS_TRT`, `TABLE_SLANEXT_WIRED_TRT`, `TABLE_SLANEXT_WIRELESS_TRT` and `TABLE_CELL_*` no longer have any effect, and the model release dropped `SLANeXt_wireless_*` + `table_cls.onnx` (~24 MB → ~8 MB). Enable tables with `TABLE_BACKEND=slanext` (+ `TABLE_SLANEXT_ENCODER_ONNX`); output is essentially unchanged (the wireless model was a byte-identical duplicate). Only affects deployments that set the old table env.

**Default-behaviour changes** (no config change, but output or runtime differ):

- **PP-OCRv6 is the default engine** (was PP-OCRv5), shipped as three tiers (`tiny`/`small`/`medium`) from the new `models-v3.0.0-ppocrv6` release. Recognition output changes vs v5.
- **Default tier is `tiny`** (max throughput). Set `OCR_MODEL=small` or `medium` for higher accuracy.
- **`LAYOUT_MERGE_MODE` default changed to `all`** (was effectively `large` / keep-outer). `all` keeps every detected box and drops nothing, so formulas/tables/titles nested inside a larger region survive (≈ +0.008 table TEDS, ≈ −0.006 formula CDM on OmniDocBench). Set `LAYOUT_MERGE_MODE=outer` to restore the previous behaviour. The mode *names* also changed (`outer`/`inner`/`all`, formerly `large`/`small`/`union`), but the old names still work as **deprecated aliases** — so the rename itself is not breaking. Modes: `outer` keeps the outer/container box and drops boxes nested inside it; `inner` keeps the innermost boxes and drops the pure containers; `all` keeps both.
- **Requests now time out at 60 s** instead of hanging unbounded. `REQUEST_TIMEOUT_MS` default changed `0` → `60000`: a wedged GPU slot returns `504 INFERENCE_TIMEOUT` and frees itself. Set `REQUEST_TIMEOUT_MS=0` to opt back into the old unbounded wait. A companion watchdog (`PIPELINE_HARD_KILL_MS`, default `600000` = 10 min) `_Exit`s the process for the orchestrator to restart **only** if a worker stays wedged mid-CUDA long after a recycle was already requested — so a genuine hang can now terminate the process instead of leaking a slot forever (this watchdog is inert when `REQUEST_TIMEOUT_MS=0`).
- **Detection resize defaults changed** (max-side `960` → `limit_type=min`, `limit_side_len=64`, `max_side_limit=1280`), so detection boxes — and therefore OCR output — differ slightly. Tune via `DET_MAX_SIDE_LIMIT` / `DET_LIMIT_TYPE` / `DET_LIMIT_SIDE_LEN` (`DET_MAX_SIDE` still honored).
- **GPU out-of-memory now returns `500 INFERENCE_ERROR`** instead of a blank `200`, and a sticky CUDA fault `_Exit`s the process for a clean restart. Under sustained overload, queued work whose client deadline already elapsed is dropped (the caller gets its `504`) rather than processed late. Clients should handle 5xx/504 and retry.
- **A bare launch announces text-only mode.** With neither `FORMULA_BACKEND` nor `TABLE_BACKEND` set, the server runs text-only (tables/formulas empty) and now logs a one-time `[Pipeline] NOTE: table + formula stages are DISABLED — running TEXT-ONLY …`, so a text-only run can't be silently mistaken for a full-document one.
- **New input-size caps** (previously-accepted requests are now rejected): `/ocr/batch` and gRPC `RecognizeBatch` cap at **1024 images** → `400 BATCH_TOO_LARGE` (split the batch or raise `MAX_BATCH_IMAGES`); `/ocr/pdf` rendered pages cap at **~40 MP/page** → very large pages at high DPI fail to render (lower `?dpi=` or raise `MAX_PDF_PAGE_PIXELS_MP`); and `/ocr`, `/ocr/raw`, `/ocr/batch`, `/infer` now also reject images over **128 MP total area** → `400 PIXELS_TOO_LARGE`, in addition to the existing per-side `MAX_IMAGE_DIM` guard (downscale, or raise `MAX_IMAGE_PIXELS_MP`).

**Additive / transparent** (nothing to do):

- **`OCR_MODEL` is the new selector name; `OCR_LANG` still works** as a deprecated alias (warns on use), so this is backward-compatible. Select by tier/model name (`tiny`/`small`/`medium`, or `arabic`/`eslav`/`korean`/`thai`/`greek`).
- **The Python formula sidecar is gone — transparently.** `FORMULA_BACKEND=ppformulanet_s` is now a fully **in-process pure-C++ PP-FormulaNet-S recognizer** on ORT-CUDA-13: same backend name, same output, no separate Python process to launch or manage. (The GPU path is FAST split-graph only; the CPU-only build decodes formulas on ORT CPU automatically.) Only callers of the deleted sidecar script / `PPFNS_SIDECAR_SCRIPT` env are affected (see *Breaking* above).
- **New `POST /ocr/markdown` route** (GPU build) exports a parsed page as faithful Markdown. Purely additive — existing routes are unchanged.
- **Oversized-image guard on `/infer`.** Like the other image routes, `/infer` now rejects inputs whose dimensions exceed `MAX_IMAGE_DIM` (default `16384`) with `400 DIMENSIONS_TOO_LARGE` (a decompression-bomb guard). Only affects callers that were sending images larger than 16384 px on a side.
- **New `*_degraded` response signals.** When a configured stage produces nothing, the JSON now carries `text_degraded` / `table_degraded` / `formula_degraded` (+ a `*_warning` string) on `/ocr`, `/ocr/raw`, `/ocr/batch` and `/ocr/pdf`, and `/ocr/markdown` sets an `X-OCR-Degraded` header — so a partial result is never a silent clean `200` (a configured-but-failed stage also now fails at boot rather than serving empties). New fields only; ignore them and nothing changes.
- **New `ppformulanet_plus_m` formula backend** (in-process, Chinese-capable) and **`GET /capabilities`** (runtime feature/route discovery) — both opt-in and additive.

---

## API

One binary serves HTTP and gRPC from a shared GPU pipeline pool.

| Endpoint | Purpose |
|---|---|
| `POST /ocr/raw` | OCR raw image bytes (fastest) |
| `POST /ocr` | OCR base64 image in JSON |
| `POST /ocr/pixels` | Zero-decode raw pixel buffer |
| `POST /ocr/batch` | Batch of images |
| `POST /ocr/pdf` | PDF → text (optional page images & auto-rotate) |
| `POST /ocr/markdown` | Page → faithful Markdown (GPU build; requires layout) |
| `POST /infer` | OCR + layout / reading-order / blocks in one structured response |
| `GET /capabilities` | Runtime feature & route discovery |
| `GET /metrics` | Prometheus metrics |
| `GET /health` · `/health/live` · `/health/ready` | Liveness / readiness probes |

All endpoints accept `?layout=1` (region detection + reading order). Example:

```bash
curl -X POST "http://localhost:8000/ocr/raw?layout=1" \
  --data-binary @document.png -H "Content-Type: image/png"
```

→ [HTTP API](https://aiptimizer.github.io/TurboOCR/api/http/) · [gRPC API](https://aiptimizer.github.io/TurboOCR/api/grpc/) · [Monitoring](https://aiptimizer.github.io/TurboOCR/api/monitoring/)

---

## Configuration

Everything is configured by environment variable (and an equivalent CLI flag).
Common ones:

| Variable | Default | Description |
|---|---|---|
| `OCR_MODEL` | `tiny` | `tiny` / `small` / `medium`, or a PP-OCRv5 script model |
| `DISABLE_LAYOUT` | `0` | `1` skips the layout model (~300–500 MB VRAM) |
| `LAYOUT_MERGE_MODE` | `all` | Nested-box policy: `all` (keep every box) / `outer` (outer regions only) / `inner` (innermost only). Old `large`/`small`/`union` accepted as aliases. |
| `LAYOUT_KEEP_NESTED_CHILDREN` | `0` | Only affects `outer`/`inner` modes: `1` still keeps the model's nested child regions (`figure_title`, `footnote`, `formula_number`, `paragraph_title`) instead of dropping them inside a parent. Formulas are always kept; no effect under default `all`. |
| `REQUEST_TIMEOUT_MS` | `60000` | Per-request inference deadline; on overrun returns `504` and frees the slot. `0` = unbounded (pre-v3 behaviour). |
| `PIPELINE_POOL_SIZE` | auto | Concurrent GPU pipelines |

→ [Full configuration reference (35+ variables)](https://aiptimizer.github.io/TurboOCR/build/config/)

---

## Building from Source

```bash
# Docker (recommended)
docker build -f docker/Dockerfile.gpu -t turboocr .
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr turboocr

# Native (PP-OCRv6 models auto-fetched into ./models/ on first build)
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/paddle_highspeed_cpp
```

Needs GCC 13.3+/C++20, CUDA + TensorRT 10.2+, OpenCV 4.x, Drogon 1.9+, gRPC.
Wuffs, Clipper, and PDFium are vendored in `third_party/`.

→ [Build guide & GPU-architecture notes](https://aiptimizer.github.io/TurboOCR/build/native/)

---

## Acknowledgements

Built on open-source work:

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Baidu) — PP-OCRv6 / PP-OCRv5 detection, recognition, and classification models, plus PP-DocLayoutV3 layout detection. This project would not exist without their research and pre-trained weights.
- **[Drogon](https://drogon.org)** — high-performance async C++ HTTP framework.
- **[Wuffs](https://github.com/google/wuffs)** — fast PNG decoder by Google (vendored).
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** — PDF rendering and text extraction (vendored).
- **[Clipper](http://www.angusj.com/delphi/clipper.php)** — polygon clipping for text-detection post-processing (vendored).

## License

MIT. See [LICENSE](LICENSE).

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ Star TurboOCR on GitHub</strong></a><br>
  <sub>Sponsored by <a href="https://miruiq.com"><strong>Miruiq</strong></a> — AI-powered data extraction from PDFs and documents — and <a href="https://diaiq.com"><strong>DiaIQ</strong></a>.</sub>
</p>
