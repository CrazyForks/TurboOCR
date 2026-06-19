<p align="center">
  <img src="tests/benchmark/comparison/images/banner.png" alt="TurboOCR — Fast GPU OCR server." width="100%">
</p>

<p align="center">
  <strong>GPU-accelerated OCR server. 15–90× faster than other OCR engines.</strong><br>
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

A production OCR server that runs PP-OCRv6 detection + recognition (+ optional
layout and PDF parsing) on a single multi-stream CUDA/TensorRT pipeline, behind
HTTP and gRPC. On forms and receipts it is both the most accurate open engine and
15–90× faster than the alternatives.

- 🚀 **Up to 556 img/s** (receipts) / **447 img/s** (forms) on one RTX 5090, fastest by default
- 🎯 **Most accurate on forms & receipts** &mdash; beats PaddleOCR-VL, PaddleOCR-Python, RapidOCR, EasyOCR and Tesseract ([benchmarks](#benchmarks))
- 🧠 **PP-OCRv6** &mdash; one model covers Latin + Chinese + Japanese; pick `tiny` (default) / `small` / `medium`
- 🌐 **More scripts** &mdash; Arabic, Cyrillic, Korean, Thai, Greek via retained PP-OCRv5 recognizers
- 📄 **PDF native** &mdash; pages rendered and OCR'd in parallel, four extraction modes, optional page-image export & auto-rotation
- 🧩 **Layout + reading order** &mdash; PP-DocLayoutV3 (25 classes) and class-aware XY-cut, opt-in per request
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

Like-for-like against the common OCR engines on a single RTX 5090. Two comparisons,
because the two classes of engine are built for different jobs.

### Forms & receipts — whole-page OCR

Same images, same word-F1 metric for every engine; FUNSD (English forms) and CORD (English receipts), 50 pages each. Word-F1 = lowercased ≥2-char token overlap.

![OCR accuracy on forms and receipts](tests/benchmark/comparison/images/compare_accuracy.png)
![Throughput on forms and receipts](tests/benchmark/comparison/images/compare_throughput.png)

| Engine | FUNSD F1 | CORD F1 | Speed (FUNSD) |
|---|---:|---:|---:|
| **TurboOCR-medium** | **91.9%** | **93.4%** | 83 img/s |
| TurboOCR-small | 90.3% | 92.8% | 225 img/s |
| TurboOCR-tiny *(default)* | 84.5% | 88.9% | **447 img/s** |
| TurboOCR-v5 *(legacy)* | 90.2% | 91.8% | 249 img/s |
| PaddleOCR-VL-1.6 | 91.6% | 89.4% | 5 img/s |
| PaddleOCR PP-OCRv5 (Python) | 86.6% | 86.4% | 6 img/s |
| RapidOCR (GPU) | 69.1% | 82.6% | 2 img/s |
| EasyOCR | 59.8% | 67.3% | 3 img/s |
| Tesseract | 62.3% | 38.2% | 2 img/s |

TurboOCR has the best accuracy **and** is 15–90× faster than every other engine on forms and receipts.

### Complex documents — full pipeline

Papers and books need layout-aware parsing, so each TurboOCR tier and PaddleOCR-VL are run
through their **full pipeline** (layout → region recognition → reading order) and scored by
the **official OmniDocBench scorer** (text-block edit distance, OmniDocBench-125, EN+ZH):

![Complex-document text accuracy and speed per model](tests/benchmark/comparison/images/compare_complex_pipeline.png)

| Pipeline | OmniDocBench-125 text accuracy | Throughput |
|---|---:|---:|
| **PaddleOCR-VL-1.5** (PP-DocLayoutV3 + 0.9B VLM) | **95.5%** | 0.94 pages/s |
| TurboOCR-small (PP-OCRv6 + PP-DocLayoutV3) | 91.3% | 70 pages/s |
| TurboOCR-medium | 91.0% | 35 pages/s |
| TurboOCR-tiny | 89.9% | 108 pages/s |

On complex full-page documents the PaddleOCR-VL vision-language model is the most accurate,
but TurboOCR's CNN pipeline is within ~4–6 points while running **35–115× faster**.
Use the VLM when every point of accuracy matters; use TurboOCR when throughput matters.

→ Full tables, metric definitions, languages, and how each pipeline is run: [Engine comparison](https://aiptimizer.github.io/TurboOCR/benchmarks/comparison/)

---

## Models

PP-OCRv6 (Latin + Chinese + Japanese) in three tiers via `OCR_MODEL`:

| `OCR_MODEL` | FUNSD F1 | Throughput | Use it for |
|---|---:|---:|---|
| `tiny` *(default)* | 84.5% | ~447 img/s | Max throughput — edge / high-volume |
| `small` | 90.3% | ~225 img/s | Balanced accuracy/speed |
| `medium` | **91.9%** | ~83 img/s | Best accuracy |

Other scripts use retained PP-OCRv5 recognizers, also via `OCR_MODEL`: `arabic`,
`eslav` (Cyrillic), `korean`, `thai`, `greek`.

→ [Model selection guide](https://aiptimizer.github.io/TurboOCR/models/selection/)

---

## Upgrading to v3 (breaking changes)

v3 moves the default engine from PP-OCRv5 to **PP-OCRv6**. Changes since v2.3:

- **PP-OCRv6 is now the default** (was PP-OCRv5), shipped as three tiers (`tiny`/`small`/`medium`) from the new `models-v3.0.0-ppocrv6` release. Recognition output and the model files change — **clear the TensorRT engine cache on upgrade** so engines rebuild from the new ONNX.
- **`OCR_MODEL` replaces `OCR_LANG`.** Select by tier/model name (`tiny`/`small`/`medium`, or `arabic`/`eslav`/`korean`/`thai`/`greek`). `OCR_LANG` still works as a **deprecated** alias (warns on use).
- **`OCR_SERVER` removed.** PP-OCRv6 covers Latin + Chinese + Japanese in one model, so the separate Chinese-server recognizer toggle is gone. Non-Latin scripts (Arabic, Cyrillic, Korean, Thai, Greek) are served by retained PP-OCRv5 recognizers.
- **Default tier is `tiny`** (max throughput). Set `OCR_MODEL=small` or `medium` for higher accuracy.
- **New: `LAYOUT_MERGE_MODE`** (default `large`; `small`/`union` keep nested boxes — for forms).

---

## API

One binary serves HTTP and gRPC from a shared GPU pipeline pool.

| Endpoint | Purpose |
|---|---|
| `POST /ocr/raw` | OCR raw image bytes (fastest) |
| `POST /ocr` | OCR base64 image in JSON |
| `POST /ocr/pixels` | Zero-decode raw pixel buffer |
| `POST /ocr/batch` | Batch of images |
| `POST /ocr/pdf` | PDF → text (4 modes; optional page images & auto-rotate) |
| `GET /metrics` | Prometheus metrics |

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
| `LAYOUT_MERGE_MODE` | `large` | Nested-box policy: `large` (regions) / `small` / `union` (keep all — for forms) |
| `PIPELINE_POOL_SIZE` | auto | Concurrent GPU pipelines |
| `ENABLE_PDF_MODE` | `ocr` | `ocr` / `geometric` / `auto` / `auto_verified` |

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
  <sub>Main Sponsor: <a href="https://miruiq.com"><strong>Miruiq</strong></a> — AI-powered data extraction from PDFs and documents.</sub>
</p>
