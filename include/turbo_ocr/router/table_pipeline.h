#pragma once

// Forward declaration shim so CuaRouter can hold a TablePipeline* without
// pulling in a table-pipeline implementation header (which would transitively
// drag in TensorRT/CUDA). The concrete TablePipeline dispatch target is not
// yet implemented; the router's table_ pointer stays null until it lands.

namespace turbo_ocr::router {

class TablePipeline;

} // namespace turbo_ocr::router
