# HTTP API

!!! abstract "TL;DR"
    Six endpoints. `POST /ocr/raw` is the fast path; `POST /ocr/pixels`
    skips decoding entirely; `POST /ocr/batch` and `POST /ocr/pdf` cover
    multi-image and PDF inputs. `GET /health/live` and `GET /health/ready`
    are Kubernetes-friendly probes. Default bind: `http://localhost:8000`.

Routes are registered in three files:

- `src/routes/common_routes.cpp` — `/health/live`, `/health/ready`,
  `/ocr/raw` (CPU path), `/ocr` (base64 JSON).
- `src/routes/image_routes.cpp` — GPU `/ocr/raw`, `/ocr/pixels`,
  `/ocr/batch`.
- `src/routes/pdf_routes.cpp` — `/ocr/pdf` (CPU + GPU overloads).

## Shared query parameters

Parsed by `server::parse_query_options()` in
`include/turbo_ocr/server/server_types.h:307`. Every parameter accepts
`1` / `true` / `on` / `yes` (or the negated equivalents).

| Param | Default | Effect |
|---|---|---|
| `layout` | `0` | Run PP-DocLayoutV3 and emit a `layout` array. |
| `reading_order` | `0` | XY-cut over the layout boxes; emits `reading_order`. Auto-enables `layout`. |
| `as_blocks` | `0` | Emit paragraph-level `blocks`. Auto-enables `layout` + `reading_order`. |
| `tables` | startup default | Run the CUA router's table branch (SLANet+/Nemotron) and emit `tables`. Auto-enables `layout`. |
| `formulas` | startup default | Run the 3-engine LaTeX-OCR formula branch and emit `formulas`. Auto-enables `layout`. |
| `rec_mode` | env `OCR_REC_MODE` | Per-request recognizer override (`mobile` / `server`). |

!!! note "Optional fields stay byte-identical when empty"
    `layout`, `reading_order`, `blocks`, `tables`, `formulas` are
    emitted only when non-empty — text-only pages produce a response
    indistinguishable from the pre-feature shape. See
    `emit_pipeline_result_json` in
    `include/turbo_ocr/common/serialization.h:567`.

!!! warning "LAYOUT_DISABLED"
    Requesting `layout=1`, `reading_order=1`, `as_blocks=1`, `tables=1`,
    or `formulas=1` against a server started with `DISABLE_LAYOUT=1`
    returns `400 LAYOUT_DISABLED`. The error message tells the caller
    exactly how to unblock it.

---

## `POST /ocr/raw`

Raw image bytes in the request body. JPEG is GPU-decoded with nvJPEG
(falling back to OpenCV); PNG goes through the Wuffs fast path;
everything else uses `cv::imdecode`.

- **Body**: raw image bytes (`image/jpeg`, `image/png`, `image/webp`,
  `image/bmp`, `image/tiff`).
- **Dim guard**: `MAX_IMAGE_DIM` (default `16384`, clamped to
  `[64, 65535]`) — pre-decode for PNG/JPEG (header sniff) and
  post-decode for the rest.

### Request

=== "bash"

    ```bash
    curl -X POST http://localhost:8000/ocr/raw \
         --data-binary @page.jpg \
         -H 'Content-Type: image/jpeg'
    ```

=== "python"

    ```python
    import requests
    with open("page.jpg", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/raw",
            data=f.read(),
            headers={"Content-Type": "image/jpeg"},
        )
    print(r.json()["results"][0]["text"])
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("page.jpg")).arrayBuffer();
    const r = await fetch("http://localhost:8000/ocr/raw", {
      method: "POST",
      headers: {"Content-Type": "image/jpeg"},
      body: bytes,
    });
    console.log((await r.json()).results[0].text);
    ```

### With layout + reading order + tables

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/raw?layout=1&reading_order=1&tables=1' \
         --data-binary @page.png \
         -H 'Content-Type: image/png'
    ```

=== "python"

    ```python
    import requests
    with open("page.png", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/raw",
            params={"layout": 1, "reading_order": 1, "tables": 1},
            data=f.read(),
            headers={"Content-Type": "image/png"},
        )
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("page.png")).arrayBuffer();
    const r = await fetch(
      "http://localhost:8000/ocr/raw?layout=1&reading_order=1&tables=1",
      {method: "POST", headers: {"Content-Type": "image/png"}, body: bytes});
    ```

### Response shape

```json
{
  "results": [
    {"id": 0, "text": "Hello world", "confidence": 0.987,
     "bounding_box": [[12,8],[180,8],[180,32],[12,32]], "layout_id": 0}
  ],
  "layout": [
    {"id": 0, "class": "text", "class_id": 1, "confidence": 0.94,
     "bounding_box": [[10,4],[800,4],[800,40],[10,40]]}
  ],
  "reading_order": [0],
  "tables": [],
  "formulas": []
}
```

Error codes: `EMPTY_BODY`, `IMAGE_DECODE_FAILED`, `DIMENSIONS_TOO_LARGE`,
`INVALID_PARAMETER`, `LAYOUT_DISABLED`, `INFERENCE_ERROR`, `SERVER_BUSY`.

---

## `POST /ocr/pixels`

Skip image decoding entirely — caller hands the server an already-decoded
BGR or grayscale buffer. Zero decode overhead, the lowest-latency entry
point.

- **Body**: raw pixel data, exactly `width × height × channels` bytes.
- **Required headers**: `X-Width`, `X-Height`. Optional `X-Channels`
  (defaults to `3`; only `1` or `3` accepted).
- **Dim guard**: same `MAX_IMAGE_DIM` as `/ocr/raw`.

### Request

=== "bash"

    ```bash
    curl -X POST http://localhost:8000/ocr/pixels \
         -H 'X-Width: 1280' -H 'X-Height: 720' -H 'X-Channels: 3' \
         -H 'Content-Type: application/octet-stream' \
         --data-binary @frame.bgr
    ```

=== "python"

    ```python
    import cv2, requests
    img = cv2.imread("frame.png")  # BGR uint8
    h, w, c = img.shape
    r = requests.post(
        "http://localhost:8000/ocr/pixels",
        data=img.tobytes(),
        headers={"X-Width": str(w), "X-Height": str(h), "X-Channels": str(c),
                 "Content-Type": "application/octet-stream"},
    )
    ```

=== "javascript"

    ```javascript
    // bgr = Uint8Array length = w*h*3
    await fetch("http://localhost:8000/ocr/pixels", {
      method: "POST",
      headers: {
        "X-Width": String(w), "X-Height": String(h), "X-Channels": "3",
        "Content-Type": "application/octet-stream",
      },
      body: bgr,
    });
    ```

Response shape is identical to `/ocr/raw`.

Error codes: `MISSING_HEADER`, `INVALID_HEADER`, `INVALID_DIMENSIONS`,
`DIMENSIONS_TOO_LARGE`, `BODY_SIZE_MISMATCH`, plus the shared set.

---

## `POST /ocr/batch`

JSON array of base64-encoded images. Decoded with nvJPEG batch decode
when ≥2 inputs are JPEG; mixed batches fall back to per-slot decode.

!!! tip "Per-slot error alignment"
    Per-slot errors keep the response array aligned with the input order
    so a single bad image never silently drops the rest of the batch.
    Successful slots get `null`; failed slots get a string tag like
    `"base64_decode_failed"` or `"dimensions_too_large (32000x32000 > 16384x16384)"`.

### Request

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/batch?layout=1' \
         -H 'Content-Type: application/json' \
         -d '{"images": ["'$(base64 -w0 page1.jpg)'", "'$(base64 -w0 page2.jpg)'"]}'
    ```

=== "python"

    ```python
    import base64, json, requests
    images = [base64.b64encode(open(p, "rb").read()).decode()
              for p in ("page1.jpg", "page2.jpg")]
    r = requests.post(
        "http://localhost:8000/ocr/batch",
        params={"layout": 1},
        json={"images": images},
    )
    ```

=== "javascript"

    ```javascript
    const toB64 = async (path) => {
      const buf = new Uint8Array(await (await fetch(path)).arrayBuffer());
      return btoa(String.fromCharCode(...buf));
    };
    const body = {images: [await toB64("page1.jpg"), await toB64("page2.jpg")]};
    await fetch("http://localhost:8000/ocr/batch?layout=1", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    ```

### Response shape

```json
{
  "batch_results": [
    {"results": [{"text": "page1 line", "confidence": 0.96,
                  "bounding_box": [[0,0],[200,0],[200,30],[0,30]]}],
     "layout": []},
    {"results": [], "layout": []}
  ],
  "errors": [null, "decode_failed"]
}
```

Whole-request 400s: `INVALID_JSON`, `EMPTY_BATCH`, `INVALID_PARAMETER`.

---

## `POST /ocr/pdf`

Accepts a PDF as raw body, base64 JSON, or multipart. Renders pages with
`fastpdf2png` (PDFium-backed) and runs the pipeline per page, with an
optional PDFium text-layer fast path that avoids OCR entirely when the
embedded text is trustworthy.

- **Body**: one of
    - raw bytes (`application/pdf`),
    - JSON `{"pdf": "<base64>"}`,
    - multipart with field name `file` or `pdf`.
- **`mode` query** (default `ocr`):
    - `ocr` — always render + OCR every page.
    - `geometric` — use the PDF text layer for content; render only when
      `layout=1` or as a safety net.
    - `auto` — text layer when trusted (`text_layer_quality == "trusted"`),
      OCR otherwise.
    - `auto_verified` — GPU only. Runs OCR, then cross-checks every
      detection against the PDF text layer; replaces matches with the
      native string (`source: "pdf"`). On CPU this aliases to `auto`.
- **`dpi` query**: 50–600 (default `100`).

!!! warning "Page cap"
    `MAX_PDF_PAGES` defaults to `2000`. Exceeding returns
    `400 PDF_TOO_LARGE` with the limit echoed back in the message.

### Request

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/pdf?mode=auto&dpi=150&layout=1' \
         --data-binary @doc.pdf -H 'Content-Type: application/pdf'
    ```

=== "python"

    ```python
    import requests
    with open("doc.pdf", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/pdf",
            params={"mode": "auto", "dpi": 150, "layout": 1},
            data=f.read(),
            headers={"Content-Type": "application/pdf"},
        )
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("doc.pdf")).arrayBuffer();
    await fetch("http://localhost:8000/ocr/pdf?mode=auto&dpi=150&layout=1", {
      method: "POST",
      headers: {"Content-Type": "application/pdf"},
      body: bytes,
    });
    ```

### Response shape

```json
{
  "pages": [
    {
      "page": 1, "page_index": 0,
      "dpi": 150, "width": 1240, "height": 1754,
      "results": [
        {"text": "Title", "confidence": 1.0,
         "bounding_box": [[50,80],[420,80],[420,120],[50,120]],
         "source": "pdf"}
      ],
      "layout": [],
      "mode": "geometric",
      "text_layer_quality": "trusted"
    }
  ]
}
```

`source` is `"ocr"` (omitted) for pixel-derived text or `"pdf"` for
text-layer (or `auto_verified`-promoted) entries. `text_layer_quality` is
one of `"absent"`, `"rejected"`, `"trusted"` (see
`text_layer_quality_for()` in `src/routes/pdf_routes.cpp:86`).

Error codes: `MISSING_PDF`, `MISSING_FILE`, `INVALID_MULTIPART`,
`BASE64_DECODE_FAILED`, `EMPTY_BODY`, `EMPTY_PDF`, `INVALID_DPI`,
`PDF_TOO_LARGE`, `PDF_RENDER_FAILED`.

---

## Health probes

### `GET /health/live`

Liveness probe. Always returns `200 ok` once the process is up.

```bash
curl http://localhost:8000/health/live
```

### `GET /health/ready`

Readiness probe. Invokes the `readiness_check` closure passed in by
`main.cpp` — verifies the pipeline can submit work without blocking, so
it correctly stays `503 NOT_READY` while TensorRT engines are being
built on first start.

```bash
curl http://localhost:8000/health/ready
```

!!! info "nginx retry-after"
    The bundled `docker/nginx.conf.template` remaps upstream **502 → 503**
    with `Retry-After: 15` so clients back off rather than treating
    engine-build time as a hard failure
    (`docker/nginx.conf.template:36-47`).

## Error envelope

Every error response carries the shared envelope:

```json
{"error": {"code": "DIMENSIONS_TOO_LARGE",
           "message": "Image dimensions 32000x32000 exceed maximum of 16384x16384"}}
```

Codes are documented inline in `proto/ocr.proto:12-17` and are the same
strings the gRPC server surfaces in the `x-error-code` trailing metadata
field.

!!! info "See also"
    - [gRPC API](grpc.md) — same surface, protobuf-shaped.
    - [CUA router](../architecture/router.md) — what `tables=1` /
      `formulas=1` actually trigger.
    - [Build → Docker](../build/docker.md) — env-var matrix and nginx
      template behaviour.
