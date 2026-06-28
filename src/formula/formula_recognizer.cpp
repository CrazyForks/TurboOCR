#include "turbo_ocr/formula/formula_recognizer.h"

#include <cstring>
#include <iostream>
#include <memory>

#include "turbo_ocr/backends/openai_endpoint.h"
#include "turbo_ocr/formula/ppformulanet_ort.h"
#include "turbo_ocr/formula/vlm_formula.h"
#include "turbo_ocr/routing/routing_config.h"

namespace turbo_ocr::formula {

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend) {
  if (backend == "ppformulanet_s") {
    // Pure in-process ORT decoder (no Python sidecar). GPU FAST host-loop only.
    return std::make_unique<PPFormulaNetOrt>();
  }
  if (backend == "ppformulanet_plus_m") {
    // PP-FormulaNet_plus-M (B6 encoder + 512-d x 6-layer MBart decoder): the
    // Chinese-formula model. In-process ORT FAST host-loop (encoder.onnx + prep.onnx +
    // static-KV decoder_step.onnx, single greedy token/step). The FAST split graphs are
    // required (no fused fallback).
    return std::make_unique<PPFormulaNetOrt>("ppformulanet_plus_m");
  }
  if (backend == "vlm") {
    return std::make_unique<VLMFormula>();
  }
  std::cerr << "[FormulaRecognizer] unknown FORMULA_BACKEND='" << backend
            << "' (expected 'ppformulanet_s', 'ppformulanet_plus_m', or 'vlm')\n";
  return nullptr;
}

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const routing::BackendSpec &spec) {
  if (spec.kind == routing::Kind::Openai)
    return std::make_unique<backends::OpenAIEndpoint>(spec);
  return make_formula_recognizer(std::string_view{spec.engine});
}

} // namespace turbo_ocr::formula
