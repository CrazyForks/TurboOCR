#pragma once

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::pipeline { class PipelineDispatcher; }

namespace turbo_ocr::routes {

/// Register /ocr/pdf — GPU path (parallel page OCR via dispatcher).
/// `pdf_only_batch > 0` (the validated ServerConfig pdf_batch) groups OCR
/// pages into chunks of that size and runs them through the pipeline's
/// batched path — required in pdf_only mode where det is a static-shape
/// engine batched at exactly that dim, and the throughput point of the mode.
/// `default_dpi` is the render DPI when the request doesn't pass ?dpi=
/// (cfg.pdf_dpi in pdf_only mode so renders match the static det profile
/// sizing; 100 otherwise).
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int pdf_only_batch = 0,
                        int default_dpi = 100,
                        int max_pdf_pages = 2000);

/// Register /ocr/pdf — CPU path (sequential page OCR via InferFunc).
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int max_pdf_pages = 2000);

} // namespace turbo_ocr::routes
