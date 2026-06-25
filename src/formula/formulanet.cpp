#include "turbo_ocr/formula/formulanet.h"

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/engine/onnx_to_trt.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <span>

namespace turbo_ocr::formula {

FormulaNet::FormulaNet() = default;

bool FormulaNet::load_model(const std::string &onnx_path) {
  // INERT AT RUNTIME on TRT 10.15: the fused MTP-K=3 `Loop` op in
  // `formulanet_s.onnx` fails engine build with an `IRecurrenceLayer` shape
  // mismatch ([3] vs [1]); see `.claude/plans/10_trt_gate_results.md` (T9
  // FAIL). The pivot per plan 03 §4.2 / risk #1 is an encoder-once + host-
  // driven decoder_step loop until EOS over two re-exported ONNXes
  // (encoder + decoder_step). The `FormulaNet::run` dispatch surface stays
  // stable across the pivot; only this `infer()`-side path changes. ONNX
  // contracts (KV-cache names/shapes, decoder_step signature) are pending
  // upstream re-export — do not speculate.
  std::string trt_path = engine::ensure_trt_engine(onnx_path, "formula");
  if (trt_path.empty()) {
    std::cerr << std::format("[FormulaNet] ensure_trt_engine failed: {}",
                              onnx_path) << '\n';
    return false;
  }
  engine_ = std::make_unique<engine::TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << std::format("[FormulaNet] failed to load TRT engine: {}",
                              trt_path) << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!allocate_buffers())      return false;

  // FAIL LOUD: the single fused-Loop engine is INERT on TRT 10.15 (see the
  // load_model header comment) — the MTP decode never runs, so run() would
  // return empty LaTeX for every region while is_ready()==true. Serving that
  // silently disables the formula stage, violating the no-silent-failure
  // contract. Refuse to load so the fail-loud recognizer_registry boot aborts
  // with a clear cause instead of starting a server that produces no formulas.
  // Set TURBO_FORMULANET_ALLOW_INERT=1 only for the host-loop unit tests that
  // exercise the decode trace without expecting real engine output.
  if (const char *allow = std::getenv("TURBO_FORMULANET_ALLOW_INERT");
      !allow || !allow[0]) {
    std::cerr << "[FormulaNet] REFUSING to load: the fused-Loop formulanet "
                 "engine is inert on TRT 10.15 (decode never runs -> silent "
                 "empty LaTeX). Use FORMULA_BACKEND=ppformulanet_s (working "
                 "ORT sidecar) or vlm. Set TURBO_FORMULANET_ALLOW_INERT=1 to "
                 "override (testing only).\n";
    engine_.reset();
    buffers_allocated_ = false;
    return false;
  }
  return true;
}

bool FormulaNet::load_tokenizer(const std::string &tokenizer_json_path) {
  auto tok = FormulaTokenizer::load(tokenizer_json_path);
  if (!tok.has_value()) {
    std::cerr << std::format("[FormulaNet] tokenizer load failed: {}",
                              tokenizer_json_path) << '\n';
    return false;
  }
  tokenizer_ = std::make_unique<FormulaTokenizer>(std::move(*tok));
  return true;
}

bool FormulaNet::discover_tensor_names() {
  // Plan 03 §3: input `x`, output `fetch_name_0`. paddle2onnx may rename; fall
  // back to "first input" / "first output" if the canonical names aren't there.
  const auto &ins = engine_->input_names();
  const auto &outs = engine_->output_names();
  if (ins.empty() || outs.empty()) {
    std::cerr << "[FormulaNet] engine reports no inputs/outputs\n";
    return false;
  }
  input_name_  = ins[0];
  output_name_ = outs[0];
  for (const auto &n : ins) {
    if (n == "x") input_name_ = n;
  }
  for (const auto &n : outs) {
    if (n == "fetch_name_0") output_name_ = n;
  }
  return true;
}

bool FormulaNet::allocate_buffers() {
  if (buffers_allocated_) return true;

  const std::size_t B    = static_cast<std::size_t>(kFormulaBatchMax);
  const std::size_t HW   = static_cast<std::size_t>(kernels::kFormulaH) *
                            static_cast<std::size_t>(kernels::kFormulaW);
  const std::size_t Tmax = static_cast<std::size_t>(kFormulaMaxOutputTokens);

  d_subrects_ = CudaPtr<kernels::FormulaSubRect>(B);
  d_bbox_     = CudaPtr<int32_t>(B * 4);
  d_input_    = CudaPtr<float>(B * HW);
  d_output_   = CudaPtr<int64_t>(B * Tmax);

  h_subrects_ = CudaHostPtr<kernels::FormulaSubRect>(B);
  h_output_   = CudaHostPtr<int64_t>(B * Tmax);

  // Bind tensor addresses once; pointers stay stable for the engine's lifetime.
  engine_->set_tensor_address(input_name_,  d_input_.get());
  engine_->set_tensor_address(output_name_, d_output_.get());

  buffers_allocated_ = true;
  return true;
}

namespace {

// Convert a det-style Box (4 corners, possibly rotated) into an axis-aligned
// sub-rect clamped to the page. Width / height >= 1.
kernels::FormulaSubRect aabb_subrect(const Box &b, int page_w, int page_h) {
  auto rect = turbo_ocr::aabb(b); // [x0, y0, x1, y1]
  int x0 = std::clamp(rect[0], 0, std::max(0, page_w - 1));
  int y0 = std::clamp(rect[1], 0, std::max(0, page_h - 1));
  int x1 = std::clamp(rect[2], x0, page_w);
  int y1 = std::clamp(rect[3], y0, page_h);
  int w  = std::max(1, x1 - x0);
  int h  = std::max(1, y1 - y0);
  return kernels::FormulaSubRect{x0, y0, w, h};
}

} // namespace

bool FormulaNet::run_sub_batch(const GpuImage &page,
                               const std::vector<Box> &boxes,
                               int begin, int end,
                               cudaStream_t stream,
                               std::vector<FormulaEngineResult> &out) {
  int B = end - begin;
  if (B <= 0) return true;
  if (B > kFormulaBatchMax) {
    std::cerr << std::format("[FormulaNet] sub-batch {} exceeds max {}",
                              B, kFormulaBatchMax) << '\n';
    return false;
  }

  // Build sub-rects on host, upload to device.
  for (int i = 0; i < B; ++i) {
    h_subrects_.get()[i] = aabb_subrect(boxes[begin + i], page.cols, page.rows);
  }
  CUDA_CHECK(cudaMemcpyAsync(d_subrects_.get(), h_subrects_.get(),
                              B * sizeof(kernels::FormulaSubRect),
                              cudaMemcpyHostToDevice, stream));

  // Preprocess: bbox + fused resize/normalize/grayscale -> (B, 1, 384, 384).
  kernels::cuda_fused_formula_pre(page, d_subrects_.get(), B,
                                   d_bbox_.get(), d_input_.get(), stream);

  // TRT execute. The Loop op runs internally; we only get a real output shape
  // after stream sync (plan 03 §4.2).
  nvinfer1::Dims4 in_dims{B, 1, kernels::kFormulaH, kernels::kFormulaW};
  if (!engine_->set_input_shape(input_name_, in_dims)) {
    std::cerr << "[FormulaNet] set_input_shape failed\n";
    return false;
  }
  if (!engine_->execute(stream)) {
    std::cerr << "[FormulaNet] TRT execute failed\n";
    return false;
  }

  // ── MANDATORY: sync before reading the data-dependent output shape ──────
  // No CUDA Graph capture wraps this enqueue (plan 03 §4.2). ~5 µs overhead.
  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvinfer1::Dims out_dims = engine_->tensor_shape(output_name_);
  if (out_dims.nbDims != 2 || out_dims.d[0] != B) {
    std::cerr << std::format("[FormulaNet] unexpected output rank/batch "
                              "(nbDims={}, d0={})",
                              out_dims.nbDims,
                              out_dims.nbDims > 0 ? out_dims.d[0] : -1) << '\n';
    return false;
  }
  int T = static_cast<int>(out_dims.d[1]);
  if (T <= 0) {
    // Empty decode for every row in the sub-batch.
    for (int i = 0; i < B; ++i) {
      out[begin + i] = FormulaEngineResult{};
    }
    return true;
  }
  int T_clamped = std::min(T, kFormulaMaxOutputTokens);

  // Truncate D2H to actual (B, T_clamped). Use the same stream then sync.
  const std::size_t bytes =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T_clamped) *
      sizeof(int64_t);
  // Note: d_output_ is allocated contiguously as [B_max, T_max]. The engine
  // wrote into the [0..B*T] prefix with row stride T. When T < T_max, rows
  // are NOT contiguous w.r.t. the host buffer -- copy per row.
  for (int i = 0; i < B; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        h_output_.get() + i * T_clamped,
        d_output_.get() + i * T_clamped,
        T_clamped * sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
  }
  (void)bytes;
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Decode rows via the tokenizer. The tokenizer trims at the first EOS and
  // applies the LaTeX postprocess internally; we expose token_count / hit_eos
  // for downstream metrics (plan 03 §9 FormulaResult contract).
  const int64_t eos_id = tokenizer_->eos_id();  // learned, not a literal
  for (int i = 0; i < B; ++i) {
    const int64_t *row = h_output_.get() + i * T_clamped;
    std::size_t tok_count = static_cast<std::size_t>(T_clamped);
    bool hit_eos = false;
    for (int t = 0; t < T_clamped; ++t) {
      if (row[t] == eos_id) {
        tok_count = static_cast<std::size_t>(t);
        hit_eos = true;
        break;
      }
    }
    FormulaEngineResult r;
    r.latex = tokenizer_->decode(
        std::span<const int64_t>(row, static_cast<std::size_t>(T_clamped)));
    r.token_count = tok_count;
    r.hit_eos = hit_eos;
    out[begin + i] = std::move(r);
  }
  return true;
}

std::vector<FormulaEngineResult>
FormulaNet::run(const GpuImage &page, const std::vector<Box> &boxes,
                cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) [[unlikely]] return out;
  if (!is_ready()) {
    std::cerr << "[FormulaNet] not ready (load_model/load_tokenizer first)\n";
    return out;
  }
  if (page.empty()) [[unlikely]] {
    std::cerr << "[FormulaNet] empty page GpuImage\n";
    return out;
  }
  const int N = static_cast<int>(boxes.size());
  out.resize(static_cast<std::size_t>(N));
  for (int beg = 0; beg < N; beg += kFormulaBatchMax) {
    int end = std::min(N, beg + kFormulaBatchMax);
    if (!run_sub_batch(page, boxes, beg, end, stream, out)) {
      // On failure, leave the remaining entries default-constructed and stop.
      return out;
    }
  }
  return out;
}

} // namespace turbo_ocr::formula
