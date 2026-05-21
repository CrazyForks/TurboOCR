#pragma once

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::pipeline { class PipelineDispatcher; }

namespace turbo_ocr::routes {

/// Register /ocr/pdf — GPU path (parallel page OCR via dispatcher).
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available);

/// Register /ocr/pdf — CPU path (sequential page OCR via InferFunc).
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available);

/// Register GET /pdf/images/{sha}/{page_idx}.{format} — cached page image retrieval.
/// Must be called after any register_pdf_route() call so the cache is warm.
void register_pdf_image_route(server::WorkPool &pool,
                               render::PdfRenderer &pdf_renderer);

} // namespace turbo_ocr::routes
