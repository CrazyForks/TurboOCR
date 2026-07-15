# Running the legacy PP-OCRv5 models

The previous-generation **PP-OCRv5** detection + recognition models
(`models/det_v5.onnx`, `models/rec_v5.onnx`, `models/keys_v5.txt`) are **not**
part of the default v3 release bundle — `scripts/fetch_release_models.sh` ships
only the v6 tiers and the retained per-script v5 recognizers. To use the v5
Latin detector/recognizer, supply those three files yourself, then point the
per-stage path overrides at them — everything else (layout, table, formula,
orientation) is unchanged, so it is a drop-in swap of just the text detector +
recognizer:

```bash
DET_ONNX=models/det_v5.onnx \
REC_ONNX=models/rec_v5.onnx \
REC_DICT=models/keys_v5.txt \
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server
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
