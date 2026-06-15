#!/bin/bash
# Pre-bake every supported language bundle into /app/models/ at Docker build
# time, pulling from the TurboOCR GitHub Release and verifying SHA256.
#
# Output layout (matches what the server reads at startup):
#   /app/models/
#     det.onnx                                  # language-agnostic
#     cls.onnx
#     rec.onnx + keys.txt                       # Latin default (flat layout)
#     rec/chinese/{rec.onnx, dict.txt}
#     rec/chinese-server/{rec.onnx, dict.txt}
#     rec/greek/{rec.onnx, dict.txt}
#     rec/eslav/{rec.onnx, dict.txt}
#     rec/arabic/{rec.onnx, dict.txt}
#     rec/korean/{rec.onnx, dict.txt}
#     rec/thai/{rec.onnx, dict.txt}
#     doc_ori.onnx                              # autorotate (variant release)

set -euo pipefail

MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v2.1.0}"
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

# Shared + Latin default (flat layout)
fetch_verified "det.onnx"  "$OUT/det.onnx"
fetch_verified "cls.onnx"  "$OUT/cls.onnx"
fetch_verified "rec.onnx"  "$OUT/rec.onnx"
fetch_verified "keys.txt"  "$OUT/keys.txt"

# PP-DocLayoutV3 (~124 MB) — required for ?layout=1 endpoints.
mkdir -p "$OUT/layout"
fetch_verified "layout.onnx" "$OUT/layout/layout.onnx"

# Per-language (nested layout). Chinese server is opt-in via OCR_INCLUDE_SERVER=1.
LANGS=(chinese greek eslav arabic korean thai)
if [[ "${OCR_INCLUDE_SERVER:-0}" == "1" ]]; then
  LANGS+=(chinese-server)
fi

for lang in "${LANGS[@]}"; do
  mkdir -p "$OUT/rec/$lang"
  fetch_verified "rec-${lang}.onnx"  "$OUT/rec/${lang}/rec.onnx"
  if [[ "$lang" == "chinese-server" ]]; then
    cp "$OUT/rec/chinese/dict.txt" "$OUT/rec/chinese-server/dict.txt"
  else
    fetch_verified "dict-${lang}.txt" "$OUT/rec/${lang}/dict.txt"
  fi
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
