#include <catch_amalgamated.hpp>

#include "turbo_ocr/table/cell_matcher.h"

using turbo_ocr::table::match_cells_to_ocr;
using turbo_ocr::table::OcrLine;
using turbo_ocr::table::quad_to_bbox;

namespace {

OcrLine line(float x1, float y1, float x2, float y2, const char* t) {
    return OcrLine{{x1, y1, x2, y2}, t};
}

std::array<int, 8> quad(int x1, int y1, int x2, int y2) {
    return {x1, y1, x2, y1, x2, y2, x1, y2};
}

} // namespace

TEST_CASE("quad_to_bbox picks extremes", "[cell_matcher]") {
    std::array<int, 8> q = {10, 20, 100, 22, 110, 80, 8, 78};
    auto b = quad_to_bbox(q);
    REQUIRE(b[0] == 8.0f);
    REQUIRE(b[1] == 20.0f);
    REQUIRE(b[2] == 110.0f);
    REQUIRE(b[3] == 80.0f);
}

TEST_CASE("matches only when intersect exceeds threshold", "[cell_matcher]") {
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 100, 100)};
    std::vector<OcrLine> ocr = {
        line(10.0f, 10.0f, 90.0f, 90.0f, "inside"),
        line(95.0f, 95.0f, 195.0f, 195.0f, "barely"),
    };
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m.size() == 1);
    REQUIRE(m[0].ocr_indices == std::vector<std::size_t>{0});
}

TEST_CASE("unmatched cell returns empty indices", "[cell_matcher]") {
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 50, 50)};
    std::vector<OcrLine> ocr = {line(200.0f, 200.0f, 300.0f, 300.0f, "far")};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m[0].ocr_indices.empty());
}

TEST_CASE("rtree handles many lines efficiently", "[cell_matcher]") {
    std::vector<OcrLine> ocr;
    ocr.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const float x = static_cast<float>(i % 50) * 20.0f;
        const float y = static_cast<float>(i / 50) * 20.0f;
        ocr.push_back(line(x, y, x + 10.0f, y + 10.0f, "x"));
    }
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 100, 100)};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m[0].ocr_indices.size() == 25);
}
