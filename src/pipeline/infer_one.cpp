#include "infer_one.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "turbo_ocr/common/backend_error.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"

namespace turbo_ocr::pipeline {

std::string infer_one_region(
    const GpuImage &gpu_img, int img_w, int img_h, cudaStream_t stream,
    const std::string &modality, const std::string &backend_name,
    const routing::BackendSpec *inline_spec,
    const std::function<table::ITableRecognizer *(const std::string &)>
        &pick_table,
    const std::function<formula::IFormulaRecognizer *(const std::string &)>
        &pick_formula) {
  using turbo_ocr::Box;

  Box full{};  // tl, tr, br, bl over the whole crop
  full[0] = {0, 0};        full[1] = {img_w, 0};
  full[2] = {img_w, img_h}; full[3] = {0, img_h};
  std::vector<Box> regions{ full };

  if (modality == "table") {
    // Transient inline backend (built + loaded here, on the worker thread/
    // context) or the named registry backend.
    std::unique_ptr<table::ITableRecognizer> transient;
    table::ITableRecognizer *rec = nullptr;
    if (inline_spec) {
      transient = table::make_table_recognizer(*inline_spec);  // nullptr on unknown engine -> handled as unavailable below
      if (transient && transient->load()) rec = transient.get();
    } else {
      rec = pick_table(backend_name);
    }
    if (!rec || !rec->is_ready())
      throw turbo_ocr::BackendUnavailableError(
          modality + " backend '" + (inline_spec ? "<inline>" : backend_name) +
          "' is unavailable (not loaded/ready)");
    cudaStreamSynchronize(stream);
    auto r = rec->run(gpu_img, regions, /*page_ocr=*/{}, stream);
    return r.empty() ? std::string() : std::move(r[0].html);
  }
  if (modality == "formula") {
    std::unique_ptr<formula::IFormulaRecognizer> transient;
    formula::IFormulaRecognizer *rec = nullptr;
    if (inline_spec) {
      transient = formula::make_formula_recognizer(*inline_spec);  // nullptr on unknown engine -> handled as unavailable below
      if (transient && transient->load_model_dir(inline_spec->model_path) &&
          transient->load_tokenizer("")) rec = transient.get();
    } else {
      rec = pick_formula(backend_name);
    }
    if (!rec || !rec->is_ready())
      throw turbo_ocr::BackendUnavailableError(
          modality + " backend '" + (inline_spec ? "<inline>" : backend_name) +
          "' is unavailable (not loaded/ready)");
    cudaStreamSynchronize(stream);
    auto r = rec->run(gpu_img, regions, stream);
    return r.empty() ? std::string() : std::move(r[0].latex);
  }
  return "";
}

} // namespace turbo_ocr::pipeline
