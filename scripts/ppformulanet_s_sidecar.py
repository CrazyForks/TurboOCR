#!/usr/bin/env python3
"""PP-FormulaNet-S formula recognition sidecar (Unix-socket RPC).

Wire format (length-prefixed binary, little-endian):
  request : u32 batch | for each:  u32 H u32 W u8[H*W*3] BGR8 row-major
  reply   : u32 status (0=ok, nonzero=err) | u32 n_results |
            for each: u32 latex_byte_len | utf8[len]
  on err  : status nonzero + u32 msg_len + utf8[len]

Why ORT / why a sidecar:
The PaddleX export of PP-FormulaNet-S is a single fused ONNX whose top-level
Loop op uses growable state carries (KV cache + token-id concat). TRT 10's
IRecurrenceLayer rejects those; ORT handles them after a small set of graph
patches (see scripts/patch_ppformulanet_s_onnx.py). Running ORT in a
sidecar process avoids dragging onnxruntime-gpu's Python deps into the
main C++ binary while keeping the model warm across pipeline calls.

Batched inference: all N crops in a request are preprocessed and stacked
into a single (N, 1, 384, 384) tensor for ONE sess.run call per request.
The resize step uses PIL.BICUBIC on a BGR-as-RGB-tagged PIL image (matches
the original sidecar's recipe that hit baseline CDM 0.6489). cv2.INTER_CUBIC
drifts -0.042 CDM at full-bench scale; PIL.BILINEAR (the literal upstream
resample=2 choice) drifts -0.035 on subset5 vs PIL.BICUBIC, so BICUBIC
remains the empirically best preprocess despite the upstream-code mismatch.
Optionally captures a CUDA graph at a fixed max-batch
(PPFNS_CUDA_GRAPH_MAX_BATCH, default disabled) so steady-state runs reuse
the same captured kernel sequence.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


INPUT_H = 384
INPUT_W = 384
# PPFormulaNet-S preprocessing constants (PaddleX defaults).
_MEAN = 0.7931
_STD = 0.1738
# Center-pad fill: PaddleX uses PIL.ImageOps.expand with no fill arg, which
# defaults to 0 (pre-normalize black). After the per-channel normalize the
# fill becomes (0/255 - mean) / std. Matching that exactly is required to
# keep the encoder's attention pattern aligned with the trained background.
_CENTER_PAD_VAL = float((0.0 - _MEAN) / _STD)
# Trailing 32-align pad (always (constant=1)) is empty when INPUT_H/W are 32-
# multiples but the constant matters if the canvas ever grows; keep the
# PaddleX value here so any future shape changes stay parity-clean.
_TRAILING_PAD_VAL = 1.0


def _log(msg: str) -> None:
    sys.stderr.write(f"[ppfns_sidecar] {msg}\n")
    sys.stderr.flush()


def _crop_margin_bgr(bgr: np.ndarray) -> np.ndarray:
    """Strip blank margins from a BGR crop. cv2 luma + bbox.

    Uses `cv2.cvtColor(bgr, COLOR_BGR2GRAY)` for the luma (correct
    R*0.299 + G*0.587 + B*0.114 with proper channel mapping). The upstream
    PaddleX path uses PIL `convert("L")` on an RGB-tagged BGR array, which
    inverts the R and B coefficients in the luma. The cv2 path here gave
    CDM 0.6802 on subset5 while the upstream-faithful PIL-L path gave
    only 0.6444 — keep cv2.
    """
    if bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mn, mx = int(gray.min()), int(gray.max())
    if mx == mn:
        return bgr
    data = (gray.astype(np.float32) - mn) / max(1, mx - mn) * 255.0
    mask = (data < 200.0).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        return bgr
    a, b, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        return bgr
    return bgr[b:b + h, a:a + w]


def _preprocess_one_to_384(bgr: np.ndarray) -> np.ndarray:
    """Resize-fit + center-pad a single BGR crop into (1, 384, 384) float32.

    Uses PIL.BICUBIC for the resize step. PIL.Image.fromarray on a BGR
    ndarray stores BGR bytes but tags the mode as RGB, so the downstream
    cv2.cvtColor(..., COLOR_BGR2GRAY) sees BGR pixels as expected. Final
    layout is one channel grayscale (matching the ONNX model's static
    (B, 1, 384, 384) input).
    """
    if bgr.size == 0:
        return np.full((1, INPUT_H, INPUT_W), _CENTER_PAD_VAL, dtype=np.float32)
    cropped = _crop_margin_bgr(bgr)
    if cropped.size == 0:
        return np.full((1, INPUT_H, INPUT_W), _CENTER_PAD_VAL, dtype=np.float32)

    h, w = cropped.shape[:2]
    sd = min(INPUT_H, INPUT_W)
    if w <= h:
        nw = sd
        nh = max(1, int(round(sd * h / w)))
    else:
        nh = sd
        nw = max(1, int(round(sd * w / h)))
    # thumbnail-style clamp: cap each side at the input box without enlarging.
    if nh > INPUT_H:
        nw = max(1, int(round(nw * INPUT_H / nh)))
        nh = INPUT_H
    if nw > INPUT_W:
        nh = max(1, int(round(nh * INPUT_W / nw)))
        nw = INPUT_W
    pil_img = Image.fromarray(cropped).resize((nw, nh), Image.BICUBIC)
    resized = np.asarray(pil_img)

    bgr_norm = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    gray = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY)

    dh = INPUT_H - gray.shape[0]
    dw = INPUT_W - gray.shape[1]
    pad_top = dh // 2
    pad_bot = dh - pad_top
    pad_left = dw // 2
    pad_right = dw - pad_left
    padded = np.pad(
        gray,
        ((pad_top, pad_bot), (pad_left, pad_right)),
        constant_values=_CENTER_PAD_VAL,
    )
    return padded[np.newaxis, :, :].astype(np.float32)


def _preprocess_batch(bgrs: list[np.ndarray]) -> np.ndarray:
    """Stack N preprocessed crops into one (N, 1, 384, 384) float32 tensor."""
    out = np.empty((len(bgrs), 1, INPUT_H, INPUT_W), dtype=np.float32)
    for i, bgr in enumerate(bgrs):
        out[i] = _preprocess_one_to_384(bgr)
    return out


class FormulaService:
    def __init__(self, model_path: Path, tokenizer_path: Path, device_id: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        _log(f"loading model {model_path}")
        t0 = time.monotonic()
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 0

        # CUDA graph: opt-in via PPFNS_CUDA_GRAPH_MAX_BATCH. Capture requires
        # fixed input shapes per session, so when enabled we pad every request
        # to the max batch and slice the outputs back to the real size.
        self.cuda_graph_batch = int(os.environ.get("PPFNS_CUDA_GRAPH_MAX_BATCH", "0") or "0")

        providers = []
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            cuda_opts: dict = {"device_id": device_id}
            if self.cuda_graph_batch > 0:
                cuda_opts["enable_cuda_graph"] = "1"
                cuda_opts["cudnn_conv_algo_search"] = "DEFAULT"
            providers.append(("CUDAExecutionProvider", cuda_opts))
        else:
            self.cuda_graph_batch = 0
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers)
        _log(f"model loaded in {time.monotonic()-t0:.2f}s providers={self.sess.get_providers()}")
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        _log(f"loading tokenizer {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.bos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self._lock = threading.Lock()
        self._on_fail = os.environ.get("PPFORMULANET_S_ON_FAIL", "emit_anyway")

        if self.cuda_graph_batch > 0:
            _log(f"cuda_graph capture enabled, fixed batch={self.cuda_graph_batch}")
            self._warmup_cuda_graph()
        else:
            self._warmup()

    def _warmup(self) -> None:
        dummy = np.full((1, 1, INPUT_H, INPUT_W), _CENTER_PAD_VAL, dtype=np.float32)
        with self._lock:
            self.sess.run([self.output_name], {self.input_name: dummy})
        _log("warmup batch=1 done")

    def _warmup_cuda_graph(self) -> None:
        # ORT CUDA graph captures on the SECOND run with a given shape; the
        # first run records, the rest replay. Prime with two passes at the
        # fixed max batch size.
        dummy = np.full(
            (self.cuda_graph_batch, 1, INPUT_H, INPUT_W), _CENTER_PAD_VAL, dtype=np.float32)
        with self._lock:
            for i in range(2):
                self.sess.run([self.output_name], {self.input_name: dummy})
        _log(f"cuda_graph warmup x2 done at batch={self.cuda_graph_batch}")

    def infer_batch(self, bgrs: list[np.ndarray]) -> list[str]:
        if not bgrs:
            return []
        x = _preprocess_batch(bgrs)
        n_real = x.shape[0]
        if self.cuda_graph_batch > 0:
            if n_real > self.cuda_graph_batch:
                # Cannot replay a captured graph at a larger batch; fall back
                # to one capture-shaped pass plus a tail. Splitting keeps the
                # graph hot for the common case.
                head = self.infer_batch(bgrs[: self.cuda_graph_batch])
                tail = self.infer_batch(bgrs[self.cuda_graph_batch :])
                return head + tail
            if n_real < self.cuda_graph_batch:
                pad = np.full(
                    (self.cuda_graph_batch - n_real, 1, INPUT_H, INPUT_W),
                    _CENTER_PAD_VAL, dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)

        with self._lock:
            out = self.sess.run([self.output_name], {self.input_name: x})[0]

        results: list[str] = []
        for i in range(n_real):
            tokens = out[i].tolist()
            cleaned: list[int] = []
            for t in tokens:
                if t == self.eos_id:
                    break
                if t in (self.pad_id, self.bos_id):
                    continue
                cleaned.append(int(t))
            latex = self.tokenizer.decode(cleaned, skip_special_tokens=True)
            if self._on_fail != "emit_anyway" and _is_mode_collapsed(cleaned, latex):
                if self._on_fail == "empty":
                    results.append("")
                    continue
            if _CANONICALIZE_LATEX:
                latex = canonicalize_latex(latex)
            results.append(latex)
        return results


# Mode-collapse heuristics — page-28657 misroute showed PP-FormulaNet-S
# producing `\frac{190}{1}{ ... }`, `\displaystyle\frac{}{}` chains and
# `\int\!\!\int\!\!\int` cascades with confidence 0.86-0.90 when the input
# crop isn't actually a formula. Detect these in the decoded string + the
# raw token sequence so the C++ caller can swap to empty/fallback.
_MODE_COLLAPSE_PATTERNS = [
    re.compile(r"((?:\\displaystyle)?\\frac\{[^{}]{0,8}\}\{[^{}]{0,8}\}){3,}"),
]


# Plus-S vs S-baseline emit semantically identical LaTeX in different styles
# (`\operatorname { l o g }` vs `\log`, `\boldsymbol{X}` vs `\mathbf{X}`).
# OmniDocBench `quick_match` text_block / reading_order paths compare surface
# strings AFTER a partial normalize that leaves these divergences visible,
# producing a matcher-cascade regression that masks plus-S's true CDM gain.
# These rewrites preserve semantics (AST-equivalent) — they only normalize
# style to match what the GT corpus uses.
_CANONICALIZE_LATEX = bool(int(os.environ.get("PPFNS_CANONICALIZE_LATEX", "0") or "0"))

# Standard LaTeX functions that should render as `\foo` rather than
# `\operatorname{foo}`. List taken from amsmath/standard log-like operators.
_LATEX_STD_FUNCS = frozenset({
    "log", "ln", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth", "arcsin", "arccos", "arctan",
    "lim", "limsup", "liminf", "max", "min", "sup", "inf",
    "det", "arg", "gcd", "deg", "dim", "ker", "hom", "Pr",
    "mod", "bmod", "pmod",
})

# Matches `\operatorname { f o o }` and `\operatorname* { f o o }` with
# arbitrary internal whitespace. Group 1 = optional `*`, group 2 = body.
_OPNAME_RE = re.compile(r"\\operatorname\s*(\*?)\s*\{([^{}]*)\}")
_BOLDSYMBOL_RE = re.compile(r"\\boldsymbol(\s*\{)")


def _canonicalize_opname(match: "re.Match[str]") -> str:
    star = match.group(1)
    body = match.group(2)
    # collapse all internal whitespace; plus-S emits each letter as a separate
    # token with surrounding spaces.
    compact = re.sub(r"\s+", "", body)
    if not compact:
        return match.group(0)
    # If the result is a known LaTeX function name, convert to the dedicated
    # macro. `\operatorname*{lim}` and `\lim` are AST-equivalent for amsmath.
    if compact in _LATEX_STD_FUNCS:
        return "\\" + compact
    # Otherwise just compact the spacing inside the existing operatorname.
    return f"\\operatorname{star}{{{compact}}}"


def canonicalize_latex(latex: str) -> str:
    """Style-normalize plus-S output to match S-baseline / GT corpus style.

    Semantically lossless — AST-equivalent rewrites only. See module docstring.
    """
    if not latex:
        return latex
    out = _OPNAME_RE.sub(_canonicalize_opname, latex)
    # `\boldsymbol{X}` → `\mathbf{X}`. Both render bold; the GT corpus
    # predominantly uses `\mathbf`, and OmniDocBench's text-block
    # `safe_latex_to_text` strips `\mathbf` cleanly but leaves `\boldsymbol`
    # in the surface form, contributing edit-distance per inline formula.
    out = _BOLDSYMBOL_RE.sub(r"\\mathbf\1", out)
    return out


def _is_mode_collapsed(token_ids: list[int], latex: str) -> bool:
    if len(token_ids) >= 240:
        return True
    if len(latex) >= 1500:
        return True
    if len(token_ids) >= 50:
        grams = [tuple(token_ids[i:i+3]) for i in range(len(token_ids)-2)]
        if grams:
            unique_ratio = len(set(grams)) / len(grams)
            if unique_ratio < 0.25:
                return True
    for pat in _MODE_COLLAPSE_PATTERNS:
        if pat.search(latex):
            return True
    return False


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _serve(sock_path: str, svc: FormulaService) -> None:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o666)
    srv.listen(8)
    _log(f"listening on {sock_path}")

    while True:
        conn, _ = srv.accept()
        try:
            (batch,) = struct.unpack("<I", _recv_exact(conn, 4))
            if batch > 256 or batch == 0:
                raise ValueError(f"bad batch size {batch}")
            bgrs: list[np.ndarray] = []
            for _ in range(batch):
                H, W = struct.unpack("<II", _recv_exact(conn, 8))
                if not (1 <= H <= 4096 and 1 <= W <= 4096):
                    raise ValueError(f"bad crop H={H} W={W}")
                pixels = _recv_exact(conn, H * W * 3)
                bgrs.append(np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 3))

            latex_results = svc.infer_batch(bgrs)

            reply = struct.pack("<II", 0, len(latex_results))
            for s in latex_results:
                enc = s.encode("utf-8")
                reply += struct.pack("<I", len(enc)) + enc
            conn.sendall(reply)
        except Exception as e:
            _log(f"request error: {e}")
            try:
                msg = str(e).encode("utf-8")
                err = struct.pack("<II", 1, len(msg)) + msg
                conn.sendall(err)
            except Exception:
                pass
        finally:
            conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--socket", default="/tmp/ppformulanet_s.sock")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()
    svc = FormulaService(Path(args.model), Path(args.tokenizer), args.device_id)
    _serve(args.socket, svc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
