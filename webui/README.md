# TurboOCR Studio (web GUI)

A small Vite + React single-page app for driving a TurboOCR server: upload a
document, run OCR with layout / tables / formulas, and see the recognised boxes
drawn as an overlay on the image. Fully separated from the C++ server — it talks
to it only over the public HTTP API.

## What it shows

- **Overlay viewer** — the image with zoom/pan and color-coded overlay polygons
  for text lines, layout regions, tables, and formulas. Each layer toggles
  independently; region colors follow the 25 layout classes.
- **Inspector** — tabs for recognised text, tables (rendered HTML in a
  sandboxed iframe), formulas (LaTeX via KaTeX), Markdown (`/ocr/markdown`), and
  the raw JSON.
- **Capability-aware controls** — on load it reads `GET /capabilities` and
  disables the Tables / Formulas toggles when the server wasn't started with
  those backends.

## How it talks to the server (no CORS)

The browser only ever calls this app's own `/api/*`. That prefix is proxied to
the TurboOCR server — by Vite in `npm run dev`, by nginx in the Docker image —
so the server is reached same-origin and needs no CORS headers or changes.

## Run it (Docker, both services)

From the repo root:

```bash
# GPU server + GUI
docker compose -f docker-compose.demo.yml up --build
# or CPU-only
docker compose -f docker-compose.demo.cpu.yml up --build
```

Then open http://localhost:3000.

## Run it (local dev)

Point a TurboOCR server at port 8000 (e.g. `docker run --gpus all -p 8000:8000
-e TABLE_BACKEND=slanext -e FORMULA_BACKEND=ppformulanet_s
ghcr.io/aiptimizer/turboocr:latest`), then:

```bash
cd webui
npm install
npm run dev          # http://localhost:3000, proxies /api -> :8000
# different backend:
OCR_BACKEND_URL=http://my-host:8000 npm run dev
```

## Scope

v1 handles single images via `POST /ocr/raw`. PDFs (`/ocr/pdf`), batch, and the
single-crop `/infer` playground are natural follow-ups.
