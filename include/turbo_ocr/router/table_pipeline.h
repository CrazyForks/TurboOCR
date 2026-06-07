#pragma once

// Forward declaration shim so CuaRouter can hold a TablePipeline* without
// pulling in the table-pipeline implementation header (which transitively
// drags TensorRT/CUDA). Real definition lives in
// include/turbo_ocr/table/table_stage.h, owned by the table-impl agent.

namespace turbo_ocr::router {

class TablePipeline;

} // namespace turbo_ocr::router
