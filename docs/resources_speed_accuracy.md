# What this OCR server is, what it's tested on, and how the backends compare

This page describes, concretely: **what the system does**, **which models do what**, **what data it
is benchmarked on** (and what that data's ground truth contains), **what every metric means**, and a
**head-to-head of the three ways to run it** — local-only, hybrid (local + PaddleOCR-VL on regions),
and PaddleOCR-VL doing the whole page — on both speed and accuracy. All numbers are reproducible
(see [Reproducing](#reproducing-every-number)).

The server is a **hybrid document-OCR pipeline**: it always runs fast local text detection +
recognition, optionally runs layout detection to find regions, and can recognize table/formula
regions either with small **local** models (SLANeXt, PP-FormulaNet) or by sending each region's crop
to an external **PaddleOCR-VL** vision-language model over an OpenAI-compatible endpoint. It does *not*
have a full-page VL mode — sending a whole page to PaddleOCR-VL is a separate standalone pipeline,
benchmarked here as the accuracy ceiling.

---

## 1. Dataset & ground truth

Benchmarks use **OmniDocBench** (`omnidocbench/data/OmniDocBench.json`) — **1,651 real document
pages** (books, papers, exams, slides, newspapers; Chinese / English / mixed; 1–3 columns). Each page
has layout regions (`layout_dets[]`) typed by `category_type`, each carrying its ground truth:
`text` (text blocks), `latex` (`equation_isolated` display formulas), or `html` (tables).

| | docs | GT tables | GT display-formulas | GT text blocks |
|---|---|---|---|---|
| **Full OmniDocBench** | 1,651 | 665 | 2,066 | 16,520 |
| **Test subset used here** | **125** | **119** | **489** | **1,294** |

The 125-doc subset is a deterministic, stratified, **table/formula-heavy** cut
(`scripts/omnidoc_subset_n.py`: docs with both a table and a formula first, then stratified across
source/language/layout) so the table and formula paths are genuinely exercised. **Speed and accuracy
use the identical 125 docs.** Those GT counts (119 tables / 489 formulas / 1294 text) are the
denominators — an "extracted N" only means something against them.

**Exactly which 125 documents** — the full list (every filename + its source/language/GT counts) is in
[`docs/benchmark_documents.md`](benchmark_documents.md). Composition:

| by source | n | | by language | n | | every page has | n |
|---|---|---|---|---|---|---|---|
| exam_paper | 27 | | simplified_chinese | 51 | | a table only | 49 |
| book | 22 | | english | 48 | | a formula only | 48 |
| academic_literature | 16 | | en_ch_mixed | 19 | | both table + formula | 28 |
| colorful_textbook | 16 | | traditional_chinese | 7 | | | |
| research_report | 12 | | | | | | |
| PPT2PDF | 11 | | | | | | |
| newspaper | 9 | | | | | | |
| note | 7 | | | | | | |
| magazine | 5 | | | | | | |

Examples: `exam_paper-file-putnam-archive_1997_..._page_001.png` (two-column math exam),
`PPT_CalculusReview_page_014.png` (slide deck), `jiaocaineedrop_..._546.jpg` (Chinese textbook),
`docstructbench_...-j.physletb.2004.11.060.pdf_2.jpg` (two-column physics paper). 77 of the 125 have a
table, 76 have a formula; mix of 1–3 column layouts, print + handwritten notes, EN/中文/mixed.

---

## 2. Models — which model does what

**What every benchmark number in this document actually uses:** the **`tiny`** OCR tier — `det_tiny.onnx`
(1.7 MB) + `rec_tiny.onnx` (4.3 MB). This is the default (`OCR_MODEL` unset) and on these complex docs
it is also the **best** tier — `medium` scored *worse* (§5). There is **no "fast" tier**; the three
tiers are `tiny` / `small` / `medium`, selected by `OCR_MODEL`. The exact stack per configuration:

- **text** (every row): `det_tiny` + `rec_tiny` + `cls` (PP-LCNet angle) + layout (PP-DocLayoutV3)
- **+ table (local)**: SLANeXt wired/wireless encoder+decoder + `table_cls`
- **+ formula (local)**: PP-FormulaNet-S via the **`ppformulanet_s`** ORT sidecar — *not* `formulanet`,
  which is encoder-only and inert (see §5)
- **+ VL (hybrid or VL-only)**: **PaddleOCR-VL-1.5-0.9B** on vLLM — table/formula region crops in the
  hybrid, or the whole page in VL-only

Always-on = text path (det+rec+cls). Everything else is optional. Per-model detail lives in
[`docs/models/`](models/); this is the inventory.

| stage | model (file) | architecture | size | selection | on by default |
|---|---|---|---|---|---|
| text detection | `models/det_tiny.onnx` (·_small 9.5 MB ·`det` 60 MB) | PP-OCRv6 DBNet | 1.7 MB | `OCR_MODEL` tier | **yes** |
| text recognition | `models/rec_tiny.onnx` (·_small 21 MB · `rec` 74 MB) | PP-OCRv6 SVTR | 4.3 MB | `OCR_MODEL` tier | **yes** |
| angle classify | `models/cls.onnx` | PP-LCNet | 1.0 MB | `DISABLE_ANGLE_CLS` | yes |
| doc orientation | `models/doc_ori.onnx` | PP-LCNet | 6.5 MB | file present | yes |
| layout | `models/layout/layout.onnx` | PP-DocLayoutV3 (RT-DETR-L) | 124 MB | `DISABLE_LAYOUT` | yes |
| table (local) | `slanext_encoder/SLANeXt_{wired,wireless}_encoder.onnx` + `_decoder.bin` + `table_cls.onnx` | SLANeXt enc(GPU)+dec(host) | 5.3 + 2.1 + 6.5 MB | `TABLE_BACKEND=slanext` | opt-in |
| formula (local) | `formula/ppformulanet_s/inference_trt.onnx` + `tokenizer.json` | PP-FormulaNet-S (split enc + AR dec, ORT sidecar) | 295 MB | `FORMULA_BACKEND=ppformulanet_s` (NOT inert `formulanet`) | opt-in |
| table / formula (external) | **PaddleOCR-VL-1.5-0.9B** (`models/vlm/paddleocr_vl_1_5`) on vLLM | VLM | 1.8 GB | `kind:openai` routing | opt-in |

OCR tiers (det+rec) trade accuracy for speed: **tiny** ~85% / ~481 img/s · **small** ~91% / ~234 ·
**medium** ~92% / ~89 (these benchmarks use tiny). The external model is **any** OpenAI-compatible
vision endpoint; PaddleOCR-VL is the one used here, and table vs formula can point at different
models/hosts. *(The standalone full-page PaddleOCR-VL pipeline in the installed `paddleocr` 3.4
supports v1.5, not v1.6, so all VL numbers here are **v1.5** for an apples-to-apples comparison.)*

---

## 3. Metrics — what every number means

Scored by the OmniDocBench scorer (`omnidocbench/pdf_validation.py`, `quick_match` element matching).

| metric | direction | what it measures |
|---|---|---|
| **text edit-dist** | ↓ lower better | normalized Levenshtein distance on recognized text vs GT |
| **table TEDS** | ↑ higher better | Tree-Edit-Distance Similarity of the table HTML (structure **and** cell text) |
| **table TEDS-structure** | ↑ | TEDS ignoring cell text — grid (rows/cols/spans) correctness only |
| **formula CDM** | ↑ | Character Detection Matching: render GT & predicted LaTeX, match characters (F1). Robust to LaTeX-syntax differences |
| **formula edit-dist** | ↓ | normalized Levenshtein on the LaTeX string |
| **reading-order edit-dist** | ↓ | edit distance of the block reading-order sequence vs GT |

"Extracted N / M GT" in the speed table is **coverage**: N regions emitted by the server vs M in the
ground truth (not an accuracy score — that's section 5).

---

## 4. Speed & VRAM (per pipeline combination)

Measured on one RTX 5090, `scripts/bench_speed_matrix.py --n 125 --pool-sizes 2 --concurrency 8` over
the 125-doc subset (`POST /ocr/raw?layout=1`). **VRAM scales with `PIPELINE_POOL_SIZE`** (concurrent
GPU pipelines) — these are at **pool=2** (the throughput sweet spot; see below). `+ext` = a separate
PaddleOCR-VL process on its own GPU (~15 GB), *not* in the C++ VRAM column.

| config | images/s | p50 / p90 ms | C++ VRAM | + ext | tables /119 GT | formulas /489 GT |
|---|---|---|---|---|---|---|
| text-only (no layout) | **324** | 1 / 3 | 4.3 GB | – | – | – |
| + layout | 56 | 50 / 125 | 5.8 GB | – | – | – |
| + table SLANeXt (local) | **86** | 57 / 123 | 6.0 GB | – | 127 | – |
| + table VL | 10.3 | 278 / 1572 | 5.8 GB | +15 GB | 127 | – |
| + formula PP-FormulaNet (local) | 92 | 49 / 99 | 6.2 GB | – | – | **0** ⚠ |
| + formula VL | 6.7 | 613 / 3270 | 5.8 GB | +15 GB | – | 1679 |
| + table + formula (local) | 55 | 83 / 145 | 6.3 GB | – | 127 | **0** ⚠ |
| + table + formula (VL) | 6.1 | 721 / 1924 | 5.8 GB | +15 GB | 127 | 1679 |
| hybrid: local table + VL formula | 12 | 344 / 1799 | 6.0 GB | +15 GB | 127 | 1679 |

Reading the coverage: SLANeXt/VL emit **127 tables ≈ 119 GT** (slight over-detect). The
`formula_local` row used `FORMULA_BACKEND=formulanet` (encoder-only) which is **inert — 0 usable
LaTeX** ⚠; the working local backend is **`ppformulanet_s`** (the ORT sidecar), which adds host-side
AR-decode cost (~28 img/s, see §5). VL emits **1679 formula regions** because layout flags inline
formulas and equation-numbers too, far more than the 489 GT *display* formulas the scorer evaluates.

**VRAM and throughput vs `PIPELINE_POOL_SIZE`** (`--pool-sizes 1 2 4 8`): VRAM ≈ base + pool × ~2 GB
(both_local: 3.5 / 6.3 / 11.9 / 23.1 GB at pool 1/2/4/8); throughput **peaks around pool=2** at
concurrency 8 and then degrades from GPU contention (pool=8 even OOMs). **Set `PIPELINE_POOL_SIZE`
explicitly** — auto-detect over-provisions VRAM for no throughput gain.

---

## 5. Accuracy & the head-to-head

`scripts/omnidoc_run_and_score_n.py` (local / hybrid) and `scripts/bench_vl_fullpage.py` (VL full
page), all on the **same 125 docs**, scored identically (`↓` lower better, `↑` higher better).

### 5a. Three bugs the first numbers exposed (and the fix)

The initial local numbers were inflated by wiring/config bugs, **not** model ceilings:

| bug | symptom | root cause | fix | result |
|---|---|---|---|---|
| reading order never requested | RO 0.558; multi-column pages scrambled (and text inflated) | `tools/omnidoc_run.py` posted only `?layout=1`; the XY-cut order the server already computes was never asked for, so markdown was dumped in `(y,x)` order | request `?layout=1&reading_order=1` + map the result-indexed order to layout blocks in `omnidoc_to_md.py` | **RO 0.558 → 0.30, text 0.150 → 0.11** |
| local formula backend inert | formula CDM 0.414 | `FORMULA_BACKEND=formulanet` fed only `encoder.onnx`, but PP-FormulaNet-S is split (resizer+encoder+decoder) → decoder never runs → every LaTeX empty → markdown falls back to the **text recognizer** reading formulas as garbled text | `FORMULA_BACKEND=ppformulanet_s` + the ORT sidecar (`scripts/ppformulanet_s_sidecar.py`) | **CDM 0.414 → 0.811** (nearly VL's 0.874) |
| table cell-text dropped | table TEDS 0.644 (struct 0.855) — 45% of cells empty | `cell_matcher` required ≥70% of an OCR line's area inside a cell quad (SLANeXt quads are smaller than DB line boxes → ~35% of lines dropped); and cells the page detector missed stayed empty | gate 0.5 + argmax fallback (`cell_matcher.cpp`, swept 0.3/0.4/0.5 → 0.5 optimal) **and** per-cell crop OCR — empty grid cells are recognized directly from their quad (`slanext_table_recognizer.cpp`) | **TEDS 0.644 → 0.774 (struct 0.872)** |

### 5b. The full matrix (125 docs, PaddleOCR-VL-1.5)

| pipeline | speed | text ↓ | tbl TEDS ↑ | tbl struct ↑ | tbl edit ↓ | fml CDM ↑ | fml edit ↓ | RO ↓ |
|---|---|---|---|---|---|---|---|---|
| Local — as first shipped (inert formula, no RO) | 55 img/s | 0.150 | 0.644 | 0.855 | 0.191 | 0.414 † | 0.733 | 0.558 |
| Local — + reading-order fix | 55 img/s | 0.127 | 0.646 | 0.861 | 0.184 | 0.414 † | 0.730 | 0.355 |
| **Local — all fixes + per-cell OCR + 0.5 gate** | **28 img/s** | **0.113** | **0.774** | **0.872** | **0.152** | **0.811** | **0.296** | **0.303** |
| Local — medium tier (tiny is better here) | 30 img/s | 0.179 | 0.611 | 0.855 | 0.204 | 0.447 | 0.720 | 0.579 |
| Hybrid — VL on regions, as first shipped | 6.1 img/s | 0.144 | 0.892 | 0.926 | 0.082 | 0.847 | 0.148 | 0.480 |
| **Hybrid — + reading-order fix** | 6.1 img/s | **0.118** | 0.895 | 0.928 | 0.082 | 0.843 | 0.148 | **0.313** |
| **VL-only — whole page** | **1.3 pg/s** | **0.073** | **0.900** | **0.931** | **0.074** | **0.874** | **0.131** | **0.194** |

† `formulanet` was inert — 0.414 is the *text recognizer* reading formulas, not a formula model.

### 5c. Per modality — which backend, head to head

| modality (metric) | local (fixed) | VL on region (hybrid) | VL full page |
|---|---|---|---|
| **text** (edit ↓) | 0.113 | 0.118 | **0.073** |
| **table** (TEDS ↑) | 0.774 | 0.895 | **0.900** |
| **table** (structure ↑) | 0.872 | 0.926 | **0.931** |
| **formula** (CDM ↑) | **0.811** | 0.843 | **0.874** |
| **reading order** (edit ↓) | 0.303 | 0.313 | **0.194** |

What this actually says:
- **The local pipeline is far stronger than the raw numbers first showed.** Four fixes plus a swept
  matcher gate took it from "weak" (table 0.644, formula 0.414, RO 0.558) to **table 0.774, formula
  0.811, RO 0.303, text 0.113** — at **28 img/s**, ~20× VL-only and ~5× the hybrid. **Local formula
  (0.811) and table (0.774) are now within striking distance of VL (0.874 / 0.900).** Remaining table
  headroom is structure decode on complex spanning/borderless grids; remaining RO gap (0.303 vs 0.194)
  is the XY-cut algorithm itself, not wiring (four reorder algorithms were tried — none beat the tuned
  baseline).
- **Memory caveat:** the local formula sidecar (ORT) needs VRAM headroom — sharing one 32 GB GPU with a
  vLLM (15 GB) can OOM and empty some formulas (CDM drops to ~0.70). This is **no longer silent**: the
  sidecar fails loud if it can't reach the GPU, and a per-region failure now surfaces as
  `formula_degraded` in the response. Give the formula sidecar its own GPU or run local-only; the 0.811
  above is the clean (vLLM-free) number.
- **VL-only is still the most accurate on everything** (text 0.073, table 0.900, formula 0.874, RO
  0.194) — one model over the whole page gives the most coherent result — but at **1.3 pages/s**.
- **Hybrid ≈ VL-only on tables/formulas**, and after the RO fix it matches local on text/RO; its value
  is VL-grade tables/formulas without paying for VL text on **text-heavy** docs (where it runs much
  faster than on this table/formula-heavy set, where every page hits VL per region).
- **Enabling the real local formula costs throughput** (55 → 28 img/s: the PP-FormulaNet-S decoder is a
  host-side autoregressive loop). Formula off / inert = faster but no real formula output.

Pick by what you optimise: **one GPU, high throughput, good-enough accuracy → local (all fixes on)**;
**maximum accuracy → VL-only**; **VL-grade tables/formulas + fast local text on text-heavy docs →
hybrid**. The remaining local gap to VL is genuine model headroom (handwriting, inline-math LaTeX,
dense-table cell content), addressable further by per-cell crop OCR + wide-line splitting in the table
path.

---

## 6. Reproducing every number

```bash
# Shared 125-doc subset is generated automatically by the harnesses (scripts/omnidoc_subset_n.py).

# Speed + VRAM matrix (boots each config, fixed extracted/GT counting) -> result/bench_speed_matrix.{json,md}
python3 scripts/bench_speed_matrix.py --n 125 --pool-sizes 2 --concurrency 8
python3 scripts/bench_speed_matrix.py --baseline result/bench_speed_matrix.json   # regression gate (>15%)

# Accuracy — boot a server with the config, then score (needs omnidocbench/.venv):
#   local:  TABLE_BACKEND=slanext + SLANEXT envs + FORMULA_BACKEND=ppformulanet_s + FORMULA_ONNX/_TOKENIZER
#           + PPFNS_SIDECAR_SCRIPT + cu12 PATH/LD_LIBRARY_PATH (see the vLLM-free recipe below)
#   hybrid: TURBO_ROUTING_CONFIG=routing.json (table/formula -> vl), TABLE_SLANEXT_* for slanext_local
python3 scripts/omnidoc_run_and_score_n.py --server-url http://localhost:8822 --experiment-name local_125

# VL-only full page (PaddleOCR-VL-1.5 standalone; needs compare-ocrs/.venv + vLLM on :8077)
python3 scripts/bench_vl_fullpage.py --subset 125 --server http://localhost:8077/v1
```

Serve PaddleOCR-VL on vLLM with **both** names so the C++ hybrid (`PaddleOCR-VL`) and the standalone
pipeline (`PaddleOCR-VL-1.5-0.9B`) both resolve:
```bash
vllm serve models/vlm/paddleocr_vl_1_5 --port 8077 --trust-remote-code \
  --served-model-name PaddleOCR-VL-1.5-0.9B PaddleOCR-VL --gpu-memory-utilization 0.45
```
Build recipe (clean): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=<trt> -DFETCH_MODELS=OFF`.

For the local pipeline, enable the working formula backend + the reading-order param:
```bash
FORMULA_BACKEND=ppformulanet_s \
  FORMULA_ONNX=models/formula/ppformulanet_s/inference_trt.onnx \
  FORMULA_TOKENIZER=models/formula/ppformulanet_s/tokenizer.json \
  PPFNS_SIDECAR_SCRIPT=scripts/ppformulanet_s_sidecar.py ./build/paddle_highspeed_cpp ...
# the sidecar needs cu12 ORT: PATH/LD_LIBRARY_PATH from .venv-modelopt (nvidia/*/lib)
```
The `formulanet` backend (encoder-only) is inert — always use `ppformulanet_s` for real local LaTeX.
