#include "turbo_ocr/table/table_recognizer.h"

#include <iostream>
#include <memory>

#include "turbo_ocr/backends/openai_endpoint.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/table/slanext_table_recognizer.h"
#include "turbo_ocr/table/vlm_table_recognizer.h"

namespace turbo_ocr::table {

std::unique_ptr<ITableRecognizer> make_table_recognizer(std::string_view backend) {
  if (backend == "slanext")  return std::make_unique<SlanextTableRecognizer>();
  if (backend == "vlm")      return std::make_unique<VLMTableRecognizer>();
  std::cerr << "[TableRecognizer] unknown TABLE_BACKEND='" << backend
            << "' (expected 'slanext' or 'vlm')\n";
  return nullptr;
}

std::unique_ptr<ITableRecognizer>
make_table_recognizer(const routing::BackendSpec &spec) {
  if (spec.kind == routing::Kind::Openai)
    return std::make_unique<backends::OpenAIEndpoint>(spec);
  return make_table_recognizer(std::string_view{spec.engine});
}

} // namespace turbo_ocr::table
