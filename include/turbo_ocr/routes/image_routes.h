#pragma once

#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::routes {

/// Register /ocr/raw, /ocr/batch, /ocr/pixels routes (GPU paths).
/// `max_batch_images` caps the /ocr/batch images[] count (over it -> 400);
/// the handler allocates O(n) before touching any image, so this is the
/// admission gate against an unbounded-array OOM.
void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available,
                           int max_batch_images = 1024);

} // namespace turbo_ocr::routes
