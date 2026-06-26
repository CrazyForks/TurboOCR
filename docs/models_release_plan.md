# Plan: republishing the model bundle to GitHub Releases

**Status: PLAN ONLY — nothing in here has been executed. No release was created,
no asset was uploaded, nothing was pushed.** This documents *how* the gitignored
`models/` bundle could be re-hosted so a fresh clone can self-heal, and prepares
the inputs (inventory + checksum command) without performing any network action.

## 1. Why this is needed

`models/` is gitignored (9.9 GB) and fetched at build time by
`scripts/fetch_release_models.sh` from a GitHub Release. That release base —
`https://github.com/aiptimizer/TurboOCR/releases/download/models-v3.0.0-ppocrv6`
— now returns **404** (e.g. `…/det.onnx` → 404), so a clean clone cannot build:
`-DFETCH_MODELS=ON` hard-fails (correctly, loudly) on the missing assets.

The fetch script already supports a **mirror fallback** and per-asset **SHA256
verification**, so re-hosting only requires (a) somewhere to put the bytes and
(b) a matching `SHA256SUMS.txt`. No code change is required to *consume* a new
release — just set `MODELS_RELEASE_URL` (or `…_FALLBACK`) to the new base.

## 2. What the running server actually loads (the essential set)

The default local pipeline reads a small subset of the 9.9 GB tree. Everything
else on disk is provenance/experimental and should be **excluded** from a release.

| Group | Files | Size |
|---|---|---|
| OCR tiny tier | `det_tiny.onnx`, `rec_tiny.onnx`, `keys_tiny.txt` | ~6 MB |
| OCR small tier | `det_small.onnx`, `rec_small.onnx` | ~30 MB |
| OCR medium tier | `det.onnx`, `rec.onnx`, `keys.txt` | ~134 MB |
| Angle classifier | `cls.onnx` | ~1 MB |
| Layout | `layout/layout.onnx` (PP-DocLayoutV3) | ~124 MB |
| Autorotate | `doc_ori.onnx` | ~6.5 MB |
| Table (SLANet-Plus) | `table/slanext_encoder/{SLANeXt_wired_encoder.onnx, SLANeXt_wired_decoder.bin, SLANeXt_dict_infer.txt}` | ~8 MB |
| Formula (PP-FormulaNet-S) | `formula/ppformulanet_s/{inference_trt.onnx, tokenizer.json}` | ~310 MB |
| **Subtotal (local, no VL)** | | **~640 MB** |
| External VL (hybrid only) | `vlm/paddleocr_vl_1_5/` (PaddleOCR-VL-1.5-0.9B) | ~1.8 GB |

**Deliberately excluded** (on disk but not loaded by the default pipeline; do not
upload):
- `formula/ppformulanet_s/*.original`, `*_patched*.onnx`, `inference.onnx`,
  `inference.pdiparams`, `inference.yml` — ~2.7 GB of pre-surgery / intermediate
  export artifacts. Only `inference_trt.onnx` + `tokenizer.json` are loaded.
- `vlm/glm_ocr/` (2.5 GB) — unused alternative VLM.
- `vlm/paddleocr_vl_1_5_nvfp4/` (790 MB) — W4A4 quant proven to destroy accuracy.
- `table/slanet_plus.onnx`, `table/table_struct_{tatr,nemotron}*`,
  `det*.int8*`, `rec*.int8*`, `rec_v5`, `det_v5`, `script_id/` — superseded /
  experimental (see `models/MANIFEST.txt` surgery notes).

> These exclusions are *plan-time* only — none are deleted. They stay on disk
> (gitignored, not re-fetchable) per the standing "do not delete models" rule.

## 3. GitHub Release constraints

- **Per-asset limit: 2 GiB.** The only essential file near this is the VL
  `model.safetensors` (1.92 GB) — under the limit, OK. (`vlm/glm_ocr/model.safetensors`
  is 2.65 GB and would need splitting, but it is excluded anyway.)
- **No release-size cap** for public repos beyond the per-asset limit; dozens of
  assets are fine.
- **Anonymous download rate limits** apply — hence the existing mirror fallback.
- Asset filenames are **flat** (no `/`). Nested paths (`layout/layout.onnx`,
  `rec/greek/rec.onnx`) are already flattened by the fetch script's naming
  (`layout.onnx`, `rec-greek.onnx`); the table/formula additions must follow the
  same flattening (e.g. `slanext_wired_encoder.onnx`, `ppformulanet_s_trt.onnx`).

## 4. Procedure (to run later, with explicit approval — NOT now)

1. **Stage the essential set** into a flat directory:
   ```bash
   # (illustrative — flatten nested paths to release-legal flat asset names)
   mkdir -p /tmp/models_release
   cp models/{det_tiny,rec_tiny,det_small,rec_small,det,rec,cls,doc_ori}.onnx /tmp/models_release/
   cp models/{keys_tiny,keys}.txt /tmp/models_release/
   cp models/layout/layout.onnx /tmp/models_release/
   cp models/table/slanext_encoder/SLANeXt_wired_encoder.onnx /tmp/models_release/slanext_wired_encoder.onnx
   cp models/table/slanext_encoder/SLANeXt_wired_decoder.bin /tmp/models_release/slanext_wired_decoder.bin
   cp models/table/slanext_encoder/SLANeXt_dict_infer.txt /tmp/models_release/slanext_dict_infer.txt
   cp models/formula/ppformulanet_s/inference_trt.onnx /tmp/models_release/ppformulanet_s_trt.onnx
   cp models/formula/ppformulanet_s/tokenizer.json /tmp/models_release/ppformulanet_s_tokenizer.json
   ```
2. **Generate the checksum manifest** the fetch script verifies against:
   ```bash
   ( cd /tmp/models_release && sha256sum * > SHA256SUMS.txt )
   ```
3. **Create the release + upload** (requires per-turn approval; never run unprompted):
   ```bash
   gh release create models-v3.1.0-local \
     --repo <owner>/<repo> --title "Model bundle v3.1.0 (local pipeline)" \
     --notes "OCR tiers + layout + SLANeXt table + PP-FormulaNet-S + PaddleOCR-VL-1.5" \
     /tmp/models_release/*
   ```
   (or `gh release upload models-v3.1.0-local /tmp/models_release/*` to add to an
   existing tag.)
4. **Point the fetch at the new base** (no code change needed for consumers):
   ```bash
   MODELS_RELEASE_URL=https://github.com/<owner>/<repo>/releases/download/models-v3.1.0-local \
     scripts/fetch_release_models.sh
   ```

## 5. Code change required to cover table + formula

`scripts/fetch_release_models.sh` currently fetches OCR/layout/lang/doc_ori only.
To make a clean clone build the **local accuracy** pipeline, add `fetch_verified`
lines for the table + formula assets (flattened names from step 1) and write them
into the nested layout the server expects:
```bash
mkdir -p "$OUT/table/slanext_encoder" "$OUT/formula/ppformulanet_s"
fetch_verified "slanext_wired_encoder.onnx"    "$OUT/table/slanext_encoder/SLANeXt_wired_encoder.onnx"
fetch_verified "slanext_wired_decoder.bin"     "$OUT/table/slanext_encoder/SLANeXt_wired_decoder.bin"
fetch_verified "slanext_dict_infer.txt"        "$OUT/table/slanext_encoder/SLANeXt_dict_infer.txt"
fetch_verified "ppformulanet_s_trt.onnx"       "$OUT/formula/ppformulanet_s/inference_trt.onnx"
fetch_verified "ppformulanet_s_tokenizer.json" "$OUT/formula/ppformulanet_s/tokenizer.json"
```

## 6. Alternative host (if GitHub is unsuitable)

A Hugging Face model repo handles multi-GB files natively (LFS), has no 2 GiB
per-file ceiling, and gives a stable CDN URL. The fetch script's
`MODELS_RELEASE_URL` already accepts "any HTTP base that hosts the same asset
names + SHA256SUMS.txt", so an HF `resolve/main` base drops in with zero code
change. Recommended for the VL weights specifically.

## 7. Verification (after a real upload)

On a clean checkout: `cmake -S . -B build -DFETCH_MODELS=ON …` should fetch +
SHA-verify every asset and build. Spot-check: server boots, `/capabilities`
lists table/formula backends, and a formula/table image returns non-empty output
(no `formula_degraded` / `table_degraded`).
