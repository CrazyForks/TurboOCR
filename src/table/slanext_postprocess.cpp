#include "turbo_ocr/table/slanext_postprocess.h"

#include <algorithm>
#include <limits>

namespace turbo_ocr::table {

namespace {

// Compute (h_scale, w_scale) used to project loc_preds back to the original
// image. SLANeXt uses the non-SLANet branch.
std::pair<float, float> bbox_scales(int padded_w, int padded_h,
                                    int ori_w, int ori_h) {
    const float pw = static_cast<float>(padded_w);
    const float ph = static_cast<float>(padded_h);
    const float rw = pw / static_cast<float>(ori_w);
    const float rh = ph / static_cast<float>(ori_h);
    const float r = std::min(rw, rh);
    return {ph / r, pw / r};
}

} // namespace

StructureResult decode_structure(
    const float* structure_probs,
    const float* loc_preds,
    std::size_t t,
    std::size_t v,
    const CharDict& dict,
    int padded_w,
    int padded_h,
    int ori_w,
    int ori_h) {
    const std::size_t sos = dict.sos_idx();
    const std::size_t eos = dict.eos_idx();
    const auto [h_scale, w_scale] = bbox_scales(padded_w, padded_h, ori_w, ori_h);
    // PaddleX: `scales[0::2] = [h_scale]*4; scales[1::2] = [w_scale]*4`.
    // h_scale lands on x-coordinate indices — preserved literally for
    // byte-equal output, even though the naming looks swapped.
    const float scales[8] = {
        h_scale, w_scale, h_scale, w_scale, h_scale, w_scale, h_scale, w_scale,
    };

    std::vector<std::string> structure;
    structure.reserve(t);
    std::vector<StructureCell> cells;
    float score_sum = 0.0f;
    std::size_t score_n = 0;

    for (std::size_t step = 0; step < t; ++step) {
        const float* row = structure_probs + step * v;
        std::size_t best_i = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < v; ++i) {
            if (row[i] > best_v) {
                best_v = row[i];
                best_i = i;
            }
        }
        if (step > 0 && best_i == eos) break;
        if (best_i == sos || best_i == eos) continue;

        std::string_view token = dict.token(best_i);
        if (dict.is_td_token(best_i)) {
            const float* lrow = loc_preds + step * 8;
            std::array<int, 8> bbox{};
            for (std::size_t k = 0; k < 8; ++k) {
                bbox[k] = static_cast<int>(lrow[k] * scales[k]);
            }
            cells.push_back(StructureCell{best_i, bbox});
        }
        structure.emplace_back(token);
        score_sum += best_v;
        ++score_n;
    }

    std::vector<std::string> wrapped;
    wrapped.reserve(structure.size() + 6);
    wrapped.emplace_back("<html>");
    wrapped.emplace_back("<body>");
    wrapped.emplace_back("<table>");
    for (auto& s : structure) wrapped.emplace_back(std::move(s));
    wrapped.emplace_back("</table>");
    wrapped.emplace_back("</body>");
    wrapped.emplace_back("</html>");

    const float structure_score =
        score_n > 0 ? score_sum / static_cast<float>(score_n) : 0.0f;
    return StructureResult{std::move(wrapped), std::move(cells), structure_score};
}

} // namespace turbo_ocr::table
