# Model Selection — `OCR_MODEL`

The recognizer is **PP-OCRv6** — a single script-agnostic model covering
**Latin + Chinese + Japanese**, shipped in three tiers. Scripts outside that
coverage are served by retained PP-OCRv5 recognizers behind the same shared v6
detector. One environment variable, `OCR_MODEL`, picks which recognizer the
server loads at startup.

## PP-OCRv6 tiers

The default is the throughput-first **`tiny`** tier. Step up to **`small`** for a
balanced accuracy/speed point, or **`medium`** for best accuracy.

| `OCR_MODEL` | FUNSD F1 | Throughput | Use it for |
| --- | ---: | ---: | --- |
| `tiny` *(default)* | ~84.5% | ~447 img/s | Maximum throughput — edge / high-volume |
| `small` | ~90.3% | ~225 img/s | Balanced accuracy/speed |
| `medium` | **~91.9%** | ~83 img/s | Best accuracy |

All three tiers cover the same Latin + Chinese + Japanese scripts; they trade
accuracy for speed, not language coverage.

!!! note "`tiny` is freely selectable"
    The `tiny` tier is now a first-class `OCR_MODEL` value — no opt-in
    environment flag is required to enable it.

## Other scripts (PP-OCRv5)

Scripts outside PP-OCRv6's coverage are served by retained PP-OCRv5
recognizers, selected with the same `OCR_MODEL` variable:

| `OCR_MODEL` | Script |
| --- | --- |
| `arabic` | Arabic |
| `eslav` | East-Slavic Cyrillic (Russian, Ukrainian, …) |
| `korean` | Hangul + basic Latin |
| `thai` | Thai |
| `greek` | Greek (dedicated recognizer) |

## Selecting a model

`OCR_MODEL` is the selector. Set it at startup; all models are baked into the
image, so there is no runtime download. An unknown value fails startup with the
list of valid models.

```bash
# Best accuracy
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e OCR_MODEL=medium ghcr.io/aiptimizer/turboocr:latest

# Cyrillic
docker run --gpus all -p 8000:8000 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e OCR_MODEL=eslav ghcr.io/aiptimizer/turboocr:latest
```

!!! warning "`OCR_LANG` is deprecated"
    `OCR_LANG` is a deprecated alias of `OCR_MODEL`. It still works but warns on
    use, and `OCR_MODEL` wins when both are set. Prefer `OCR_MODEL`.

!!! info "See also"
    - [Engine comparison](../benchmarks/comparison.md) — full accuracy and throughput numbers across engines and datasets.
    - [Build · Configuration](../build/config.md) — `OCR_MODEL` and the other environment-variable knobs.
    - [Recognition](recognition.md) — how the selected recognizer runs inside the pipeline.
    - [Detection](detection.md) — the shared v6 detector that feeds every recognizer.
