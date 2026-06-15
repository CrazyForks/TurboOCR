#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <sys/types.h>

#include "turbo_ocr/formula/formula_recognizer.h"

namespace turbo_ocr::formula {

// PP-FormulaNet-S backend.
//
// Implemented as a Python ORT sidecar process talking over a Unix
// domain socket. TRT 10 cannot build the fused PaddleX export because
// the top-level Loop op uses growable state carries (KV cache + token
// concat); ORT handles those after a small graph patch
// (scripts/patch_ppformulanet_s_onnx.py). Running ORT in a sidecar
// avoids dragging the onnxruntime-gpu Python deps into the main C++
// binary while keeping the model warm across pipeline calls.
//
// The sidecar binary is `scripts/ppformulanet_s_sidecar.py`; it is
// spawned during load_model_dir() and torn down when this object
// destructs (server lifetime).
class PPFormulaNetS final : public IFormulaRecognizer {
public:
  PPFormulaNetS();
  ~PPFormulaNetS() noexcept override;

  PPFormulaNetS(const PPFormulaNetS &) = delete;
  PPFormulaNetS &operator=(const PPFormulaNetS &) = delete;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  [[nodiscard]] bool is_ready() const noexcept override {
    return sidecar_pid_ > 0 && sock_path_.size() > 0;
  }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "ppformulanet_s";
  }

private:
  bool spawn_sidecar(const std::string &model_path,
                     const std::string &tokenizer_path);
  bool wait_socket_ready();
  bool rpc_call(const std::vector<std::vector<uint8_t>> &crops_bgr,
                const std::vector<int> &widths,
                const std::vector<int> &heights,
                std::vector<std::string> &out_latex);

  std::string sock_path_;
  pid_t       sidecar_pid_ = -1;
  std::mutex  rpc_mu_;
  // Host page buffer (grow-only) for the per-call D2H copy of `page`.
  std::vector<uint8_t> host_page_;
};

} // namespace turbo_ocr::formula
