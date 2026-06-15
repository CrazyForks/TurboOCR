#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/engine/rec_graph_profiles.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/server/env_utils.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace turbo_ocr::engine {

using turbo_ocr::detection::read_det_max_side;
using turbo_ocr::detection::kDetMaxSideMin;

// TensorRT builder optimization level: 0..5 (TRT 10 range). Higher = better
// kernel selection at the cost of build time. Default 5 — same as before
// this knob existed. Operators on small instances or with strict cold-start
// budgets can drop to 3 (build ~3-5× faster, runtime regression typically
// <5%). Read once on first call; subsequent calls hit the function-static.
[[nodiscard]] static int read_trt_opt_level() {
  static const int v = server::env_int("TRT_OPT_LEVEL", 5, 0, 5);
  return v;
}

// ---------- PDF-only fixed-resolution mode ----------------------------------
// When the operator boots with TURBO_OCR_PDF_ONLY=1, the DET TRT engine is
// built with a STATIC (min==opt==max) input profile at {pdf_batch, 3,
// pdf_page_h, pdf_page_w} — TRT picks tactics optimal for that exact size
// and elides every dynamic-shape branch at execute time. Pages render at
// the pdf_dpi default and det resize-fits them into the static shape. Rec
// stays dynamic 5-bucket and cls is skipped (hybrid mode — see ServerConfig).
//
// Same env vars + bounds as ServerConfig (which validates them and aborts
// boot on errors before any engine builds, so these reads see sane values).
[[nodiscard]] static int read_pdf_page_h() { return server::env_int("TURBO_OCR_PDF_PAGE_H", 1280, 32, 4096); }
[[nodiscard]] static int read_pdf_page_w() { return server::env_int("TURBO_OCR_PDF_PAGE_W", 960,  32, 4096); }
[[nodiscard]] static int read_pdf_batch()  { return server::env_int("TURBO_OCR_PDF_BATCH",  8,    1,  8); }

static class BuildLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // Concurrent TRT builds (e.g. det/rec/cls/layout in parallel during
      // warmup) can interleave on std::cerr. Build phase only — no impact on
      // request hot path.
      static std::mutex log_mu;
      std::lock_guard<std::mutex> lock(log_mu);
      std::cerr << "[TRT Build] " << msg << '\n';
    }
  }
} s_logger;

std::string get_engine_cache_dir() {
  // User override
  if (auto *env = std::getenv("TRT_ENGINE_CACHE"))
    return env;

  // Try ~/.cache/turbo-ocr/
  if (auto *home = std::getenv("HOME")) {
    auto dir = std::string(home) + "/.cache/turbo-ocr";
    fs::create_directories(dir);
    return dir;
  }

  // Fallback to /tmp
  auto dir = std::string("/tmp/turbo-ocr-engines");
  fs::create_directories(dir);
  return dir;
}

std::string get_cached_engine_path(const std::string &onnx_path,
                                   const std::string &type) {
  auto cache_dir = get_engine_cache_dir();

  // Build a cache key from: onnx file size + mtime + TRT version
  auto onnx_size = fs::file_size(onnx_path);
  auto onnx_mtime = fs::last_write_time(onnx_path).time_since_epoch().count();

  int trt_major = 0, trt_minor = 0, trt_patch = 0;
#ifdef NV_TENSORRT_MAJOR
  trt_major = NV_TENSORRT_MAJOR;
  trt_minor = NV_TENSORRT_MINOR;
  trt_patch = NV_TENSORRT_PATCH;
#endif

  // GPU compute capability (engines are GPU-architecture specific)
  int gpu_major = 0, gpu_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_minor, cudaDevAttrComputeCapabilityMinor, 0));

  // CUDA driver + runtime versions: a host driver upgrade can invalidate
  // engines cached under the previous driver (cryptic CUDA errors at
  // deserialize time). TRT version covers majors but not driver minor
  // patches, so include both integers (e.g. 13020 for CUDA 13.2) directly.
  int cuda_driver = 0, cuda_runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

  // Cache key includes: onnx identity, TRT version, GPU arch, and profile version.
  // Bump kProfileVersion when optimization profiles change for det/rec/cls.
  // Adding a NEW model type (e.g. "layout") does NOT require a bump because
  // the cache key includes `type` — new types live in their own hash space.
  // 2026-04-26: bumped because the det profile MAX now tracks DET_MAX_SIDE
  // instead of being hardcoded at 960.
  static constexpr int kProfileVersion = 20260426;

  auto key = "v" + std::to_string(kProfileVersion) + ":" + type + ":" +
      onnx_path + ":" + std::to_string(onnx_size) + ":" +
      std::to_string(onnx_mtime) + ":" + std::to_string(trt_major) + "." +
      std::to_string(trt_minor) + "." + std::to_string(trt_patch) + ":sm" +
      std::to_string(gpu_major) + "." + std::to_string(gpu_minor) +
      ":drv" + std::to_string(cuda_driver) +
      ":rt" + std::to_string(cuda_runtime);
  // For det, the MAX dim of the optimization profile depends on DET_MAX_SIDE,
  // so each operator config gets its own engine (Triton/vLLM pattern).
  if (type == "det")
    key += ":dms" + std::to_string(read_det_max_side());
  // rec gets extra static (batch,width) profiles + maxAuxStreams=0 ONLY when
  // CUDA graphs are opted into. Default (graphs off) keeps the original
  // single-profile cache key, so existing cached engines are reused as-is and
  // no rebuild is triggered on upgrade.
  if (type == "rec" && TrtEngine::graphs_enabled())
    key += ":gp12aux0";
  // PDF-only static det engine: the cache must distinguish between
  // different fixed shapes / batches, so bake the page dims + batch into
  // the cache key — a DPI change rebuilds rather than reusing stale.
  if (type == "det_pdf_static") {
    key += ":pdfh" + std::to_string(read_pdf_page_h()) +
           ":pdfw" + std::to_string(read_pdf_page_w()) +
           ":pdfb" + std::to_string(read_pdf_batch());
  }
  // TRT_OPT_LEVEL changes which kernels TensorRT picks, so the produced
  // engine differs. Operators that toggle the level get separate cached
  // engines instead of silently reusing a stale one.
  key += ":opt" + std::to_string(read_trt_opt_level());
  auto hash = std::hash<std::string>{}(key);

  return cache_dir + "/" + type + "_" + std::to_string(hash) + ".trt";
}

// A QDQ-bearing ONNX (modelopt INT8/FP8 output) carries explicit Quantize/
// Dequantize nodes whose zero-point dtypes pin the compute precision. TRT 10
// honors light QDQ under a weakly-typed FP16 network, but heavier modelopt
// exports (INT32 zero-points on weight reshapes, mixed FP8/INT8) hit
// "DequantizeLayer can only run in INT8/FP8/FP4/INT4" under weak typing — the
// modelopt-recommended fix is a STRONGLY-TYPED network, where every tensor's
// type comes from the graph and the autotuner only picks tactics. Detected by
// the quantized-variant filename convention used across the repo's scripts.
[[nodiscard]] static bool is_quantized_onnx(const std::string &p) {
  return p.find(".int8.") != std::string::npos ||
         p.find("_int8") != std::string::npos ||
         p.find("_fp8") != std::string::npos ||
         p.find(".fp8.") != std::string::npos;
}

static bool build_engine(const std::string &onnx_path,
                          const std::string &trt_path,
                          const std::string &type) {
  const bool strongly_typed = is_quantized_onnx(onnx_path);
  std::cout << "Building TRT engine: " << onnx_path << " -> " << trt_path;
  if (type == "det") std::cout << " (DET_MAX_SIDE=" << read_det_max_side() << ")";
  if (strongly_typed) std::cout << " [strongly-typed QDQ]";
  std::cout << '\n';

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(s_logger));
  if (!builder) return false;

  // Weakly-typed FP16 by default (bit 0 clear). Quantized QDQ ONNX needs a
  // strongly-typed network instead (bit 0 = kSTRONGLY_TYPED): TRT then reads
  // precision from the graph and kFP16 must NOT be set (it's rejected).
  const uint32_t net_flags =
      strongly_typed
          ? (1U << static_cast<uint32_t>(
                 nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
          : 0U;
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(net_flags));
  if (!network) return false;

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, s_logger));
  if (!parser || !parser->parseFromFile(onnx_path.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    return false;

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  // Layout needs more workspace than det/rec/cls because PP-DocLayoutV3 is a
  // DETR-family model with large attention matmuls; 1 GB is fine for the
  // others. Det workspace scales with DET_MAX_SIDE: at 4096 the activation
  // tensors are ~17× larger than at 960, so 1 GB can be tight. Capped at
  // 4 GB (same ceiling as layout) so even 4096-side builds fit on a 16 GB
  // card alongside rec/cls/layout.
  size_t workspace_bytes;
  if (type == "layout") {
    workspace_bytes = 4ULL << 30;          // 4 GiB
  } else if (type == "det") {
    // (det_max/960)² × 1 GiB, capped at 4 GiB. At default 960 → 1 GiB
    // (unchanged). At 2048 → ~4 GiB. At 4096 → 4 GiB (cap).
    int det_max = read_det_max_side();
    double scale = (static_cast<double>(det_max) / 960.0) *
                   (static_cast<double>(det_max) / 960.0);
    size_t scaled = static_cast<size_t>((1ULL << 30) * scale);
    workspace_bytes = std::min(scaled, size_t(4ULL << 30));
    if (workspace_bytes < (1ULL << 30)) workspace_bytes = 1ULL << 30;
  } else if (type == "slanext_wired" || type == "slanext_wireless" ||
             type == "formula") {
    workspace_bytes = 2ULL << 30;          // 2 GiB (plan 07 §5)
  } else if (type == "det_pdf_static") {
    // Static-shape PDF det engine: same workspace policy as dynamic det,
    // scaled by the larger of the two PDF page dims (since both are fixed).
    double side = std::max(read_pdf_page_h(), read_pdf_page_w());
    double scale = (side / 960.0) * (side / 960.0);
    size_t scaled = static_cast<size_t>((1ULL << 30) * scale);
    workspace_bytes = std::min(scaled, size_t(4ULL << 30));
    if (workspace_bytes < (1ULL << 30)) workspace_bytes = 1ULL << 30;
  } else {
    workspace_bytes = 1ULL << 30;          // rec, cls, table_cls, table_cell_*
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
  // Strongly-typed networks reject kFP16 — precision is fixed by the graph.
  if (!strongly_typed)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // TRT_OPT_LEVEL: 0..5, default 5. Lower trades runtime perf for build time.
  const int opt_level = read_trt_opt_level();
  config->setBuilderOptimizationLevel(opt_level);
  std::cout << "[TRT] builder optimization level: " << opt_level << '\n';

  auto profile = builder->createOptimizationProfile();
  auto input = network->getInput(0);
  bool profile_added_in_branch = false;

  if (type == "det") {
    // MAX tracks DET_MAX_SIDE (default 960). MIN is the floor (32, after
    // round-down-to-32). OPT is the sweet-spot for typical inputs, capped
    // by MAX so a small DET_MAX_SIDE doesn't violate MIN<=OPT<=MAX.
    int det_max = read_det_max_side();
    int det_opt = std::min(640, det_max);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, kDetMaxSideMin, kDetMaxSideMin});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, det_opt, det_opt});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{1, 3, det_max, det_max});
  } else if (type == "det_pdf_static") {
    // PDF-only mode: every page is rendered to the SAME (H, W) at a fixed
    // DPI, so we pin the profile to a single shape (min==opt==max) at the
    // configured batch. TRT can fully specialize: no dynamic-shape branches
    // at execute time, tactic search runs once for this exact shape.
    const int B = read_pdf_batch();
    const int H = read_pdf_page_h();
    const int W = read_pdf_page_w();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{B, 3, H, W});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{B, 3, H, W});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{B, 3, H, W});
  } else if (type == "rec") {
    // Profile 0: dynamic profile, covers all shapes (the default serving path).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 48, 48});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 48, 320});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{32, 3, 48, 4000});
    // Static (batch,width) profiles + maxAuxStreams=0 only when CUDA graphs
    // are opted into — they exist solely to give the baked-graph path frozen,
    // shape-specialized contexts. Keeping them out of the default build means
    // the default rec engine (and its cache key) is unchanged from baseline.
    if (TrtEngine::graphs_enabled()) {
      config->addOptimizationProfile(profile);
      for (const auto &gp : kRecGraphProfiles) {
        auto *p = builder->createOptimizationProfile();
        const nvinfer1::Dims4 d{gp.batch, 3, 48, gp.width};
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, d);
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, d);
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, d);
        config->addOptimizationProfile(p);
      }
      config->setMaxAuxStreams(0);
      profile_added_in_branch = true;
    }
  } else if (type == "layout") {
    // PP-DocLayoutV3 has 3 inputs: image [B,3,800,800], im_shape [B,2],
    // scale_factor [B,2]. paddle2onnx does not guarantee input ordering, so
    // dispatch by name. Batch profile: min=1, opt=4, max=8 (measured sweet
    // spot on RTX 5090: 1.18 ms / image at B=4).
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *in = network->getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{4, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{8, 3, 800, 800});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{4, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{8, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for layout model: " << name << '\n';
        return false;
      }
    }
  } else if (type == "table_cls") {
    // PP-LCNet_x1_0_table_cls (224×224, 2 classes wired/wireless). Shape
    // profile from turbostruct-rs/crates/turbostruct-engine/src/builder/
    // profiles.rs::populate_table_cls (lines 166-183).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 3, 224, 224});
  } else if (type == "table_cell_wired" || type == "table_cell_wireless") {
    // RT-DETR-L_{wired,wireless}_table_cell_det — same triple-input contract
    // as PP-DocLayoutV3 but at 640×640. Profile from profiles.rs::
    // populate_table_cell (lines 185-222).
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *in = network->getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{4, 3, 640, 640});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{4, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for " << type << ": " << name
                  << '\n';
        return false;
      }
    }
  } else if (type == "slanext_wired" || type == "slanext_wireless") {
    // SLANeXt {wired,wireless} encoder, 488×488. Decoder Loop body is part
    // of the same engine — internal dynamic dim handled by TRT. Profile from
    // profiles.rs::populate_slanext (lines 224-243).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{4, 3, 488, 488});
  } else if (type == "formula") {
    // PP-FormulaNet-S — single grayscale 384×384 input, MTP-K=3 decoder
    // Loop op. Profile from profiles.rs::populate_formula (lines 245-276).
    // HGNetV2-B4 encoder carries Q/DQ ops baked into the ONNX; TRT honors
    // those under FP16 weakly-typed networks. Do NOT setFlag(kINT8) with a
    // fresh calibrator here — the decoder Loop body must stay FP16 (plan
    // 07 §5).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 1, 384, 384});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 1, 384, 384});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 1, 384, 384});
  } else if (type == "doc_ori") {
    // PP-LCNet_x1_0_doc_ori document-orientation classifier, 224x224. Opt at
    // batch 1 (one page per autorotate detect); max 8 to match the model's
    // own dynamic profile. Must match kDocOriSize in
    // classification/doc_orientation_common.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 3, 224, 224});
  } else {
    // cls: PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25), input
    // 80x160. Must match kClsImageH/kClsImageW in classification/paddle_cls.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{128, 3, 80, 160});
  }
  if (!profile_added_in_branch)
    config->addOptimizationProfile(profile);

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) return false;

  std::error_code ec;
  fs::create_directories(fs::path(trt_path).parent_path(), ec);
  if (ec) {
    std::cerr << "[TRT] Failed to create cache dir for " << trt_path << ": "
              << ec.message() << '\n';
    return false;
  }

  // Write to a temp file first, then atomic rename (prevents corruption from
  // concurrent builds in multi-replica Docker deployments).
  //
  // Both the open and the close-after-write check log on failure with the
  // strerror text. Earlier the open path was a silent `return false`, and
  // a uid-mismatch on a bind-mounted engine cache (host uid 1000, container
  // ocr uid 1001) hit it as EACCES — which surfaced only as "Failed to
  // build engine from <onnx>" from the caller, despite TRT having built
  // the engine successfully. Hours of bisection later, ergo this errno log.
  const auto tmp_path = trt_path + ".tmp." + std::to_string(getpid());
  std::ofstream file(tmp_path, std::ios::binary);
  if (!file) {
    std::cerr << "[TRT] Failed to open temp file " << tmp_path << ": "
              << std::strerror(errno) << '\n';
    return false;
  }
  file.write(static_cast<const char *>(plan->data()), plan->size());
  file.close();
  if (!file) {
    std::cerr << "[TRT] Failed to write temp file " << tmp_path
              << " (disk full?): " << std::strerror(errno) << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }
  fs::rename(tmp_path, trt_path, ec);
  if (ec) {
    std::cerr << "[TRT] Failed to rename " << tmp_path << " -> " << trt_path
              << ": " << ec.message() << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }

  std::cout << "Built: " << trt_path << " ("
            << static_cast<double>(plan->size()) / (1024 * 1024) << " MB)\n";
  return true;
}

void sweep_orphan_engine_temps(int min_age_seconds) {
  std::error_code ec;
  const auto cache_dir = get_engine_cache_dir();
  if (!fs::exists(cache_dir, ec) || ec) return;

  const auto now = fs::file_time_type::clock::now();
  int removed = 0;
  for (const auto &entry : fs::directory_iterator(cache_dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const auto name = entry.path().filename().string();
    if (name.find(".tmp.") == std::string::npos) continue;

    const auto mtime = fs::last_write_time(entry.path(), ec);
    if (ec) { ec.clear(); continue; }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - mtime).count();
    if (age < min_age_seconds) continue;

    fs::remove(entry.path(), ec);
    if (!ec) ++removed;
    ec.clear();
  }
  if (removed > 0)
    std::cout << "[TRT] Removed " << removed
              << " orphan engine temp file(s) from " << cache_dir << '\n';
}

std::string ensure_trt_engine(const std::string &onnx_path,
                               const std::string &type) {
  if (!fs::exists(onnx_path)) {
    std::cerr << "[TRT] ONNX not found: " << onnx_path << '\n';
    return {};
  }

  auto trt_path = get_cached_engine_path(onnx_path, type);

  if (fs::exists(trt_path)) {
    std::cout << "Using cached engine: " << trt_path;
    if (type == "det") std::cout << " (DET_MAX_SIDE=" << read_det_max_side() << ")";
    std::cout << '\n';
    return trt_path;
  }

  if (!build_engine(onnx_path, trt_path, type)) {
    std::cerr << "[TRT] Failed to build engine from: " << onnx_path << '\n';
    return {};
  }

  return trt_path;
}

} // namespace turbo_ocr::engine
