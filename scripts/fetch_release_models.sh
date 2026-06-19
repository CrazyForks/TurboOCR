#!/bin/bash
# Pre-bake every supported model bundle into /app/models/ at Docker build
# time, pulling from the TurboOCR GitHub Release and verifying SHA256.
#
# Output layout (matches what the server reads at startup):
#   /app/models/
#     det.onnx + rec.onnx + keys.txt            # PP-OCRv6 medium (flat default)
#     det_small.onnx + rec_small.onnx           # PP-OCRv6 small tier
#     det_tiny.onnx + rec_tiny.onnx + keys_tiny.txt   # PP-OCRv6 tiny tier
#     cls.onnx                                  # angle classifier (script-agnostic)
#     rec/greek/{rec.onnx, dict.txt}            # retained PP-OCRv5 scripts
#     rec/eslav/{rec.onnx, dict.txt}            # (v6 covers Latin/CJK only)
#     rec/arabic/{rec.onnx, dict.txt}
#     rec/korean/{rec.onnx, dict.txt}
#     rec/thai/{rec.onnx, dict.txt}
#     layout/layout.onnx                        # PP-DocLayoutV3 (?layout=1)
#     doc_ori.onnx                              # autorotate (variant release)

set -euo pipefail

MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v3.0.0-ppocrv6}"
OUT="${OUT:-models}"

mkdir -p "$OUT"
echo "[fetch_release_models] base=$MODELS_RELEASE_URL  out=$OUT"

SUMS_FILE="$OUT/SHA256SUMS.release.txt"
wget --tries=3 --timeout=30 --retry-connrefused -nv \
  "${MODELS_RELEASE_URL}/SHA256SUMS.txt" -O "$SUMS_FILE"

fetch_verified() {
  local asset=$1 target=$2
  echo "  $asset -> $target"
  wget --tries=3 --timeout=60 --retry-connrefused -nv \
    "${MODELS_RELEASE_URL}/${asset}" -O "${target}.part"
  local expected
  expected=$(awk -v a="$asset" '$2 == a {print $1}' "$SUMS_FILE")
  [[ -z "$expected" ]] && { echo "    ERROR: no SHA entry for $asset" >&2; exit 1; }
  local actual
  actual=$(sha256sum "${target}.part" | awk '{print $1}')
  [[ "$actual" != "$expected" ]] && {
    echo "    ERROR: sha256 mismatch for $asset" >&2
    echo "      expected: $expected" >&2
    echo "      actual:   $actual" >&2
    rm -f "${target}.part"
    exit 1
  }
  mv "${target}.part" "$target"
}

# PP-OCRv6 medium (flat default) + small/tiny tiers as flat siblings.
# medium/small share keys.txt; tiny ships its own 6,904-char keys_tiny.txt.
fetch_verified "det.onnx"        "$OUT/det.onnx"
fetch_verified "rec.onnx"        "$OUT/rec.onnx"
fetch_verified "keys.txt"        "$OUT/keys.txt"
fetch_verified "det_small.onnx"  "$OUT/det_small.onnx"
fetch_verified "rec_small.onnx"  "$OUT/rec_small.onnx"
fetch_verified "det_tiny.onnx"   "$OUT/det_tiny.onnx"
fetch_verified "rec_tiny.onnx"   "$OUT/rec_tiny.onnx"
fetch_verified "keys_tiny.txt"   "$OUT/keys_tiny.txt"

# Angle classifier (script-agnostic, serves all paths; PP-OCRv6 ships none).
fetch_verified "cls.onnx"        "$OUT/cls.onnx"

# PP-DocLayoutV3 (~124 MB) — required for ?layout=1 endpoints.
mkdir -p "$OUT/layout"
fetch_verified "layout.onnx" "$OUT/layout/layout.onnx"

# Retained PP-OCRv5 scripts (nested layout) with no PP-OCRv6 equivalent.
LANGS=(greek eslav arabic korean thai)

for lang in "${LANGS[@]}"; do
  mkdir -p "$OUT/rec/$lang"
  fetch_verified "rec-${lang}.onnx"  "$OUT/rec/${lang}/rec.onnx"
  fetch_verified "dict-${lang}.txt"  "$OUT/rec/${lang}/dict.txt"
done

# PP-LCNet_x1_0_doc_ori (~6.5 MB) — autorotate (?autorotate=1). Lives on the
# pdf-page-images variant models release, not the base bundle; sha pinned here
# so the asset can't change under us. Recipe: scripts/export_doc_ori.py.
DOC_ORI_RELEASE_URL="${DOC_ORI_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v2.4.0-pdf-page-images}"
DOC_ORI_SHA256="96e898f047a0e460ba0652e9afb8c874e53872821cfd7a3fec53a5ab62df92f0"
echo "  doc_ori.onnx -> $OUT/doc_ori.onnx"
wget --tries=3 --timeout=60 --retry-connrefused -nv \
  "${DOC_ORI_RELEASE_URL}/doc_ori.onnx" -O "$OUT/doc_ori.onnx.part"
if [[ "$(sha256sum "$OUT/doc_ori.onnx.part" | awk '{print $1}')" != "$DOC_ORI_SHA256" ]]; then
  echo "    ERROR: sha256 mismatch for doc_ori.onnx" >&2
  rm -f "$OUT/doc_ori.onnx.part"
  exit 1
fi
mv "$OUT/doc_ori.onnx.part" "$OUT/doc_ori.onnx"

rm -f "$SUMS_FILE"  # not shipped in image
echo ""
echo "[fetch_release_models] baked:"
find "$OUT" -type f -printf "  %-40p  %s bytes\n" | sort
