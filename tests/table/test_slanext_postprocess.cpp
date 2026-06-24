#include <catch_amalgamated.hpp>

#include <cmath>

#include "turbo_ocr/table/slanext_dict.h"
#include "turbo_ocr/table/slanext_postprocess.h"
#include "turbo_ocr/table/table_types.h"

using turbo_ocr::table::CharDict;
using turbo_ocr::table::decode_structure;
using turbo_ocr::table::default_dict;
using turbo_ocr::table::SLANEXT_VOCAB;

namespace {

std::vector<float> one_hot(std::size_t t, std::size_t v,
                           const std::vector<std::size_t>& ids) {
    REQUIRE(ids.size() == t);
    std::vector<float> out(t * v, 0.0f);
    for (std::size_t step = 0; step < t; ++step) {
        out[step * v + ids[step]] = 1.0f;
    }
    return out;
}

std::size_t find_token(const CharDict& d, std::string_view t) {
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.token(i) == t) return i;
    }
    return d.len();
}

std::size_t any_td(const CharDict& d) {
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.is_td_token(i)) return i;
    }
    return d.len();
}

} // namespace

TEST_CASE("decode walks until eos", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    REQUIRE(v == SLANEXT_VOCAB);
    const std::size_t header_idx = find_token(dict, "<thead>");
    REQUIRE(header_idx < v);
    std::vector<std::size_t> ids = {dict.sos_idx(), header_idx, dict.eos_idx(),
                                    header_idx};
    auto probs = one_hot(4, v, ids);
    std::vector<float> loc(4 * 8, 0.0f);
    auto r = decode_structure(probs.data(), loc.data(), 4, v, dict, 488, 488,
                              100, 100);
    REQUIRE(r.structure.size() == 7); // 6 wrapper + 1 thead
    REQUIRE(r.structure[0] == "<html>");
    REQUIRE(r.structure[3] == "<thead>");
    REQUIRE(r.cells.empty());
}

TEST_CASE("td token emits scaled bbox", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td_idx = any_td(dict);
    REQUIRE(td_idx < v);
    std::vector<std::size_t> ids = {dict.sos_idx(), td_idx, dict.eos_idx()};
    auto probs = one_hot(3, v, ids);
    std::vector<float> loc(3 * 8, 0.0f);
    const float quad[8] = {0.1f, 0.1f, 0.5f, 0.1f, 0.5f, 0.5f, 0.1f, 0.5f};
    for (int k = 0; k < 8; ++k) loc[8 + k] = quad[k];
    auto r = decode_structure(probs.data(), loc.data(), 3, v, dict, 488, 488,
                              244, 244);
    REQUIRE(r.cells.size() == 1);
    // padded=488, ori=244 → ratio=2.0 → h_scale=w_scale=244.
    const int expected[8] = {
        static_cast<int>(0.1f * 244.0f), static_cast<int>(0.1f * 244.0f),
        static_cast<int>(0.5f * 244.0f), static_cast<int>(0.1f * 244.0f),
        static_cast<int>(0.5f * 244.0f), static_cast<int>(0.5f * 244.0f),
        static_cast<int>(0.1f * 244.0f), static_cast<int>(0.5f * 244.0f),
    };
    for (int k = 0; k < 8; ++k) REQUIRE(r.cells[0].bbox[k] == expected[k]);
}

TEST_CASE("structure score is mean of kept tokens", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td_idx = any_td(dict);
    std::vector<float> probs(4 * v, 0.0f);
    probs[dict.sos_idx()] = 1.0f;
    probs[v + td_idx] = 0.6f;
    probs[2 * v + td_idx] = 0.8f;
    probs[3 * v + dict.eos_idx()] = 1.0f;
    std::vector<float> loc(4 * 8, 0.0f);
    auto r = decode_structure(probs.data(), loc.data(), 4, v, dict, 488, 488,
                              244, 244);
    REQUIRE(std::fabs(r.structure_score - 0.7f) < 1e-5f);
    // Both quads are all-zero (blank); RapidAI filter_blank_bbox drops them,
    // while the structure tokens still count toward the score.
    REQUIRE(r.cells.empty());
}
