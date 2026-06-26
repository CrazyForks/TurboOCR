#include "turbo_ocr/formula/formula_recognizer.h"

#include <cstring>
#include <iostream>
#include <memory>

#include "turbo_ocr/backends/openai_endpoint.h"
#include "turbo_ocr/formula/formulanet.h"
#include "turbo_ocr/formula/ppformulanet_ort.h"
#include "turbo_ocr/formula/vlm_formula.h"
#include "turbo_ocr/routing/routing_config.h"

namespace turbo_ocr::formula {

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend) {
  if (backend == "formulanet") {
    return std::make_unique<FormulaNet>();
  }
  if (backend == "ppformulanet_s") {
    // Pure in-process ORT decoder (no Python sidecar). GPU FAST host-loop by default,
    // PPFNS_EXACT=1 for the fused graph, FORMULA_DEVICE=cpu for ORT-CPU.
    return std::make_unique<PPFormulaNetOrt>();
  }
  if (backend == "vlm") {
    return std::make_unique<VLMFormula>();
  }
  std::cerr << "[FormulaRecognizer] unknown FORMULA_BACKEND='" << backend
            << "' (expected 'formulanet', 'ppformulanet_s', or 'vlm')\n";
  return nullptr;
}

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const routing::BackendSpec &spec) {
  if (spec.kind == routing::Kind::Openai)
    return std::make_unique<backends::OpenAIEndpoint>(spec);
  return make_formula_recognizer(std::string_view{spec.engine});
}

} // namespace turbo_ocr::formula
