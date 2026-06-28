#pragma once

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::pipeline { class PipelineDispatcher; }

namespace turbo_ocr::routes {

/// Register /ocr/pdf — GPU path (parallel page OCR via dispatcher).
/// `default_dpi` is the render DPI when the request doesn't pass ?dpi=.
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int default_dpi = 100,
                        int max_pdf_pages = 2000,
                        bool doc_ori_available = false);

/// Register /ocr/pdf — CPU path (sequential page OCR via InferFunc).
/// `orient_fn` (optional) detects a page's clockwise rotation for autorotate;
/// pass an empty function to disable (autorotate requests then 400).
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int max_pdf_pages = 2000,
                        server::OrientFunc orient_fn = {});

} // namespace turbo_ocr::routes
