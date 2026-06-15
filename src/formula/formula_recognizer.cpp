#include "turbo_ocr/formula/formula_recognizer.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "turbo_ocr/formula/formulanet.h"
#include "turbo_ocr/formula/ppformulanet_s.h"
#include "turbo_ocr/formula/vlm_formula.h"

namespace turbo_ocr::formula {

std::string resolve_formula_backend_env() {
  const char *env = std::getenv("FORMULA_BACKEND");
  if (env == nullptr || env[0] == '\0') return "formulanet";
  return std::string(env);
}

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend) {
  if (backend == "formulanet") {
    return std::make_unique<FormulaNet>();
  }
  if (backend == "ppformulanet_s") {
    return std::make_unique<PPFormulaNetS>();
  }
  if (backend == "vlm") {
    return std::make_unique<VLMFormula>();
  }
  std::cerr << "[FormulaRecognizer] unknown FORMULA_BACKEND='" << backend
            << "' (expected 'formulanet', 'ppformulanet_s', or 'vlm')\n";
  return nullptr;
}

} // namespace turbo_ocr::formula
