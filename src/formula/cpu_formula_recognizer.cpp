#include "turbo_ocr/formula/cpu_formula_recognizer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "turbo_ocr/formula/ppformulanet_preprocess.h"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

namespace {
constexpr int S = kFormulaInputSize;  // 384
}  // namespace

bool CpuFormulaRecognizer::load(const std::string &model_path,
                                const std::string &tokenizer_json) {
  fs::path mp(model_path);
  fs::path onnx = fs::is_directory(mp) ? (mp / "inference_trt.onnx") : mp;
  if (!fs::exists(onnx)) {
    std::cerr << "[CpuFormula] model not found: " << onnx << '\n';
    return false;
  }
  if (!fused_.load_cpu(onnx.string())) {
    std::cerr << "[CpuFormula] fused graph load_cpu failed: " << onnx << '\n';
    return false;
  }
  tok_ = FormulaTokenizer::load(tokenizer_json);
  if (!tok_) {
    std::cerr << "[CpuFormula] tokenizer load failed: " << tokenizer_json << '\n';
    return false;
  }
  ready_ = true;
  std::cerr << "[CpuFormula] CPU decode path ready (fused graph, ORT CPUExecutionProvider)\n";
  return true;
}

std::string CpuFormulaRecognizer::recognize(const cv::Mat &bgr_crop) {
  if (!ready_ || bgr_crop.empty()) return {};
  std::vector<float> in((size_t)S * S);
  cv::Mat cont = bgr_crop.isContinuous() ? bgr_crop : bgr_crop.clone();
  formula_preprocess_one(cont.ptr<uint8_t>(), cont.cols, cont.rows, in.data());
  std::vector<int64_t> flat; int64_t L = 0;
  if (!fused_.run_tokens("x", "fetch_name_0", in.data(), 1, flat, L)) return {};
  const int64_t EOS = tok_->eos_id();
  std::vector<int64_t> seq;
  for (int64_t j = 0; j < L; ++j) {
    int64_t t = flat[j];
    if (t == EOS) break;
    if (t != 0 && t != 1) seq.push_back(t);
  }
  std::string latex = tok_->decode(seq, /*post_process=*/false);
  static const bool drop_collapse = std::getenv("PPFNS_DROP_COLLAPSE") != nullptr;
  if (drop_collapse && formula_is_mode_collapsed(seq, latex)) return {};
  return latex;
}

std::vector<std::string>
CpuFormulaRecognizer::recognize_regions(const cv::Mat &page,
                                        const std::vector<Box> &boxes) {
  std::vector<std::string> out;
  if (!ready_ || page.empty() || boxes.empty()) return out;
  out.resize(boxes.size());

  // Chunked batched decode of the fused graph (matches the GPU cpu_ branch /
  // PPFNS_CHUNK semantics): smaller batches keep the encoder's batched conv
  // bit-matched to the per-crop reference while amortizing the AR Loop.
  const int N = (int)boxes.size();
  static const int chunk = []{
    const char *e = std::getenv("PPFNS_CHUNK");
    int c = e ? std::atoi(e) : 8;
    return c < 1 ? 1 : (c > 32 ? 32 : c);
  }();
  static const bool drop_collapse = std::getenv("PPFNS_DROP_COLLAPSE") != nullptr;
  const int64_t EOS = tok_->eos_id();

  std::vector<float> host_in;  // [B,1,S,S]
  for (int s0 = 0; s0 < N; s0 += chunk) {
    const int B = std::min(chunk, N - s0);
    host_in.assign((size_t)B * S * S, 0.0f);
    for (int i = 0; i < B; ++i) {
      auto cr = clamped_crop_rect(boxes[s0 + i], page.cols, page.rows);
      cv::Rect roi(cr[0], cr[1], cr[2], cr[3]);
      cv::Mat crop = page(roi).clone();  // contiguous BGR8 sub-image
      formula_preprocess_one(crop.ptr<uint8_t>(), crop.cols, crop.rows,
                             host_in.data() + (size_t)i * S * S);
    }
    std::vector<int64_t> flat; int64_t L = 0;
    if (!fused_.run_tokens("x", "fetch_name_0", host_in.data(), B, flat, L)) {
      std::cerr << "[CpuFormula] fused decode failed (chunk s0=" << s0 << ")\n";
      continue;  // leave these slots empty
    }
    for (int i = 0; i < B; ++i) {
      std::vector<int64_t> seq;
      for (int64_t j = 0; j < L; ++j) {
        int64_t t = flat[(size_t)i * L + j];
        if (t == EOS) break;
        if (t != 0 && t != 1) seq.push_back(t);
      }
      std::string latex = tok_->decode(seq, /*post_process=*/false);
      out[s0 + i] = (drop_collapse && formula_is_mode_collapsed(seq, latex))
                        ? std::string()
                        : latex;
    }
  }
  return out;
}

}  // namespace turbo_ocr::formula
