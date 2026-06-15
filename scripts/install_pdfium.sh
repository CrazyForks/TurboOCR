#!/bin/bash
# Ensure third_party/pdfium holds the PDFium SDK matching the build architecture.
#
# The repo vendors an x86_64 PDFium in third_party/pdfium. On aarch64 that copy
# is the wrong arch, so this script fetches pdfium-linux-arm64 from
# bblanchon/pdfium-binaries and unpacks it over third_party/pdfium. On x86_64
# the vendored copy is used as-is (no network needed).
#
# Caveat: bblanchon's aarch64 build aborts at startup on kernels whose page size
# is neither 4 KiB nor 16 KiB (e.g. some RHEL/CentOS aarch64 configured at
# 64 KiB). Standard Ubuntu/Debian aarch64 and NVIDIA Jetson/L4T use 4 KiB pages
# and work. See https://github.com/bblanchon/pdfium-binaries/issues/148
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PDFIUM_DIR="$ROOT/third_party/pdfium"

# Prefer Docker's TARGETARCH (amd64/arm64) when set; else the host machine.
ARCH="${TARGETARCH:-$(uname -m)}"
case "$ARCH" in
  x86_64|amd64|x64) PDFIUM_ARCH="x64" ;;
  aarch64|arm64)    PDFIUM_ARCH="arm64" ;;
  *) echo "install_pdfium: unsupported arch '$ARCH'" >&2; exit 1 ;;
esac

# Idempotent: if the installed PDFium already matches the target arch, do
# nothing. Falls back to "x64 vendored copy is fine" when `file` is unavailable.
if [ -f "$PDFIUM_DIR/lib/libpdfium.so" ]; then
  if command -v file >/dev/null 2>&1; then
    cur="$(file -b "$PDFIUM_DIR/lib/libpdfium.so" 2>/dev/null || echo '')"
    case "$PDFIUM_ARCH" in
      x64)   echo "$cur" | grep -q "x86-64"  && { echo "install_pdfium: third_party/pdfium already x86_64"; exit 0; } ;;
      arm64) echo "$cur" | grep -q "aarch64" && { echo "install_pdfium: third_party/pdfium already aarch64"; exit 0; } ;;
    esac
  elif [ "$PDFIUM_ARCH" = "x64" ]; then
    echo "install_pdfium: x86_64 — using vendored third_party/pdfium"
    exit 0
  fi
fi

# Pin via PDFIUM_RELEASE (a bblanchon tag like chromium/6996); default latest.
PDFIUM_RELEASE="${PDFIUM_RELEASE:-latest}"
if [ "$PDFIUM_RELEASE" = "latest" ]; then
  URL="https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-${PDFIUM_ARCH}.tgz"
else
  URL="https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_RELEASE}/pdfium-linux-${PDFIUM_ARCH}.tgz"
fi

echo "install_pdfium: fetching ${PDFIUM_ARCH} PDFium from $URL"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TMP/pdfium.tgz"
else
  wget -q "$URL" -O "$TMP/pdfium.tgz"
fi

# Optional integrity check (set PDFIUM_SHA256 in CI for a pinned release).
if [ -n "${PDFIUM_SHA256:-}" ]; then
  echo "${PDFIUM_SHA256}  $TMP/pdfium.tgz" | sha256sum -c -
fi

rm -rf "$PDFIUM_DIR"
mkdir -p "$PDFIUM_DIR"
# bblanchon tarballs unpack to {include/, lib/libpdfium.so, ...} with no top dir.
tar -xzf "$TMP/pdfium.tgz" -C "$PDFIUM_DIR"

if [ ! -f "$PDFIUM_DIR/lib/libpdfium.so" ]; then
  echo "install_pdfium: libpdfium.so missing after unpack" >&2
  exit 1
fi
echo "install_pdfium: installed ${PDFIUM_ARCH} PDFium into $PDFIUM_DIR"
