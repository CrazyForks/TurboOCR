# Models

!!! abstract "TL;DR"
    `scripts/fetch_release_models.sh` pulls every supported PP-OCRv5
    bundle plus the language-agnostic detection / classification /
    layout heads from a pinned GitHub Release, verifies SHA-256 against
    `SHA256SUMS.txt`, and lays them out on disk in the layout the
    server expects at startup.

## Release URL

```bash
MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v2.1.0}"
```

Override `MODELS_RELEASE_URL` to pin a different release tag (e.g.
inside an air-gapped mirror). The fetcher first downloads
`SHA256SUMS.txt` and then refuses any asset whose hash doesn't match
(`fetch_release_models.sh:34-46`).

## Per-language download flow

For every language in `LANGS=(chinese greek eslav arabic korean thai)`
(plus `chinese-server` when `OCR_INCLUDE_SERVER=1`):

```text
${MODELS_RELEASE_URL}/rec-${lang}.onnx   →  models/rec/${lang}/rec.onnx
${MODELS_RELEASE_URL}/dict-${lang}.txt   →  models/rec/${lang}/dict.txt
```

!!! note "chinese-server reuses chinese's dict"
    `chinese-server` is a bigger PP-OCRv5 backbone trained on the same
    label set, so the fetcher copies `chinese/dict.txt` rather than
    pulling a duplicate (`fetch_release_models.sh:67-70`).

The shared / Latin-default heads land at:

```text
${MODELS_RELEASE_URL}/det.onnx       →  models/det.onnx
${MODELS_RELEASE_URL}/cls.onnx       →  models/cls.onnx
${MODELS_RELEASE_URL}/rec.onnx       →  models/rec.onnx       (Latin)
${MODELS_RELEASE_URL}/keys.txt       →  models/keys.txt       (Latin)
${MODELS_RELEASE_URL}/layout.onnx    →  models/layout/layout.onnx
```

## On-disk inventory

Verified tree from this checkout (`find models -maxdepth 3 -type f`):

```text
cls.onnx
det.onnx
keys.txt
MANIFEST.txt
rec.onnx
layout/layout.onnx
rec/arabic/{rec.onnx,dict.txt}
rec/chinese/{rec.onnx,dict.txt}
rec/eslav/{rec.onnx,dict.txt}
rec/greek/{rec.onnx,dict.txt}
rec/korean/{rec.onnx,dict.txt}
rec/thai/{rec.onnx,dict.txt}
script_id/{script_id.onnx, script_id_v2.pt, meta.json, training_log.json}
table/slanext_encoder/{SLANeXt_wired_encoder.onnx, SLANeXt_wired_decoder.bin, SLANeXt_dict_infer.txt}   # SLANet-Plus — the only local table backend
formula/ppformulanet_s/{fast/{encoder,prep,step_batched}.onnx, inference_trt.onnx, tokenizer.json}        # PP-FormulaNet-S (GPU fast graphs + CPU fused)
formula/ppformulanet_plus_m/{encoder,prep,decoder_step,decoder_step_384}.onnx, tokenizer.json             # plus-M (Chinese, opt-in)
```

!!! note "Superseded / archival files on disk"
    Older table and formula artifacts (`table/{table_cls,slanet_plus,table_struct_tatr,table_struct_nemotron}.onnx`,
    the 3-engine `formula/{encoder,decoder,image_resizer}.onnx` set, and `*.orig.onnx`/`*.bak`/intermediate
    `ppformulanet_s/inference*.onnx` exports) remain on disk for archival but are **not loaded and not shipped** —
    the live backends are SLANet-Plus (table) and PP-FormulaNet-S / plus-M (formula) above.

!!! tip "MANIFEST.txt is the canonical ledger"
    `models/MANIFEST.txt` records the exact SHA-256 the server expects
    for each asset, plus inline surgery notes explaining why
    pre-surgery copies are retained.

## Persisted cache in Docker

Both images mount `/home/ocr/.cache/turbo-ocr` as the persistent cache
target. The Dockerfile symlinks `/app/models/rec → /home/ocr/.cache/
turbo-ocr/models/rec` **before** running `fetch_release_models.sh`, so
every per-language bundle lands directly in the cache volume
(`docker/Dockerfile.gpu:104-115`). A single
`-v trt-cache:/home/ocr/.cache/turbo-ocr` thus persists:

- TensorRT engine plans (built from ONNX on first start, ~90 s).
- All per-language recognition bundles.

The `det`, `cls`, `rec.onnx`, `keys.txt`, and `layout/` heads stay
inside the image because they're language-agnostic.

## Formula & table backends (how they run)

### PP-FormulaNet-S — in-process ORT-CUDA FAST split graphs

The native graph's Paddle `WhileOp` (lowered by `paddle2onnx` to `Loop` +
`ScatterElements`) is hard for TensorRT's Loop optimizer, so formula does **not**
run on TRT. Instead it runs **in-process on ONNX Runtime CUDA-13** as a host-driven
split-graph loop — `fast/{encoder,prep,step_batched}.onnx` — matching the fused
reference (CDM ≈ 0.811) at ~8× speed. The fused `inference_trt.onnx` is used only by
the CPU-only build (`CpuFormulaRecognizer`). See [formula](../models/formula.md).
The opt-in Chinese model `ppformulanet_plus_m` ships its own split graphs in the
model dir. (Older `*.orig.onnx`/`*.bak`/3-engine exports are archival only.)

### Table — SLANet-Plus (TRT encoder + host GRU decoder)

The live and only local table backend is **SLANet-Plus**: a TRT-FP16 CNN encoder
plus a hand-written C++ GRU decoder, loaded from `table/slanext_encoder/` via
`TABLE_BACKEND=slanext`. The earlier wired/wireless `table_cls` router, and the
`table_struct_tatr` / `table_struct_nemotron` detectors, were evaluated and
**removed** (measured worse on borderless tables); their files remain on disk as
archival only. See [table](../models/table.md).

## `FETCH_MODELS` CMake option

Both Dockerfiles pass `-DFETCH_MODELS=OFF` because the fetcher already
ran in an earlier image layer. For native builds, leave the default
behaviour and run `fetch_release_models.sh` once by hand:

```bash
bash scripts/fetch_release_models.sh
```

Re-running is idempotent — the SHA-256 verification will reject any
asset that drifts from the pinned hashes.

!!! info "See also"
    - [Build → Native](native.md) — when to invoke the fetcher by
      hand.
    - [Build → Docker](docker.md) — how the fetcher is staged into the
      image.
    - [Models → Formula](../models/formula.md) — the three-engine
      flow that replaces PP-FormulaNet-S.
