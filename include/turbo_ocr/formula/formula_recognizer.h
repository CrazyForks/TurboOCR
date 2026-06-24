#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::routing { struct BackendSpec; }

namespace turbo_ocr::formula {

struct FormulaEngineResult {
  std::string latex;
  std::size_t token_count = 0;
  bool        hit_eos = false;
};

// Backend-agnostic formula recognition interface. Implementations may use
// any execution shape internally — split TRT engines with a host AR loop
// (FormulaNet), a fused image-to-tokens engine that owns its own loop
// (PP-FormulaNet-S), an ORT sidecar, or a remote VLM.
//
// Contract: load the on-disk artefact bundle from a directory, optionally
// load a tokenizer, then answer run() with a FormulaEngineResult per input
// box in input order.
class IFormulaRecognizer {
public:
  virtual ~IFormulaRecognizer() noexcept = default;

  [[nodiscard]] virtual bool load_model_dir(const std::string &model_dir) = 0;

  // Tokenizer is backend-defined. Some backends embed their tokenizer in
  // the model bundle; pass an empty string in that case and the call is a
  // no-op that returns true.
  [[nodiscard]] virtual bool load_tokenizer(const std::string &path) = 0;

  [[nodiscard]] virtual std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) = 0;

  // --- Async decouple (opt-in) -------------------------------------------
  // Remote (HTTP/VLM) backends can split run() into a fast non-blocking
  // submit (D2H + PNG-encode + global-pool submit, done on the GPU pipeline
  // worker) and a later await+parse (done OFF the GPU worker, on the work
  // pool / HTTP thread). Local TRT/ORT backends leave these defaulted and
  // keep the synchronous run() — supports_async() stays false for them.
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  // Submit each box's crop and return one raw-response future per box, in
  // input order, WITHOUT blocking on the network. Default: not supported.
  [[nodiscard]] virtual std::vector<std::future<std::string>>
  submit_async(const GpuImage & /*page*/, const std::vector<Box> & /*boxes*/,
               cudaStream_t /*stream*/) {
    return {};
  }

  // Turn one raw endpoint response (resolved from a submit_async future) into
  // the final LaTeX. Default identity; remote backends apply their parser.
  [[nodiscard]] virtual std::string
  parse_async_result(const std::string &raw) const { return raw; }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;

  [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;
};

// Factory. `backend` is the key used to select the implementation
// (typically read from FORMULA_BACKEND). Unknown keys return nullptr.
//
// Supported keys:
//   "formulanet"      — split encoder + decoder + host AR loop (default)
//   "ppformulanet_s"  — PP-FormulaNet-S fused image-to-tokens TRT engine
//   "vlm"             — OpenAI-compatible vLLM endpoint (env-configured)
std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend);

// Overload from a resolved routing::BackendSpec: kind==Openai -> the generic
// OpenAIEndpoint; kind==Local -> the engine-keyed factory above.
std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const routing::BackendSpec &spec);

} // namespace turbo_ocr::formula
