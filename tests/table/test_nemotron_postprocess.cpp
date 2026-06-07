#include <catch_amalgamated.hpp>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "turbo_ocr/table/nemotron_postprocess.h"

using turbo_ocr::table::assemble_nemotron;
using turbo_ocr::table::decode_nemotron;
using turbo_ocr::table::kNemotronStride;
using turbo_ocr::table::NemotronAssembly;
using turbo_ocr::table::NemotronBox;
using turbo_ocr::table::NemotronCell;
using turbo_ocr::table::NemotronDecode;
using turbo_ocr::table::nemotron_to_html;

namespace {

// Helper: build one raw query row in the layout the model emits.
// `obj_logit` / `cls_logits` should be raw (pre-sigmoid).
std::array<float, 10> raw_query(float cx, float cy, float w, float h,
                                float obj_logit,
                                std::array<float, 5> cls_logits) {
    return {cx, cy, w, h, obj_logit,
            cls_logits[0], cls_logits[1], cls_logits[2], cls_logits[3], cls_logits[4]};
}

void push_query(std::vector<float>& buf, const std::array<float, 10>& q) {
    buf.insert(buf.end(), q.begin(), q.end());
}

// All-zero logits everywhere except where we explicitly set them.
constexpr std::array<float, 5> kNoCls{-10, -10, -10, -10, -10};

float logit(float p) {
    return std::log(p / (1.0f - p));
}

} // namespace

TEST_CASE("decode filters by conf threshold", "[nemotron]") {
    std::vector<float> buf;

    // High confidence cell at canvas center → should survive.
    auto cls_cell = kNoCls; cls_cell[1] = logit(0.95f);
    push_query(buf, raw_query(512, 512, 100, 40, logit(0.95f), cls_cell));

    // Low confidence cell → should be dropped (0.95 * 0.02 = 0.019 < 0.05).
    auto cls_low = kNoCls; cls_low[1] = logit(0.02f);
    push_query(buf, raw_query(700, 200, 50, 50, logit(0.95f), cls_low));

    // 1024-canvas → original image of size 512×512 (ratio = 2.0).
    auto d = decode_nemotron(buf.data(), buf.size() / kNemotronStride, 512, 512);

    REQUIRE(d.cells.size() == 1);
    REQUIRE(d.rows.empty());
    REQUIRE(d.columns.empty());
    // (cx-w/2)/ratio = (512-50)/2 = 231; (cx+w/2)/ratio = (512+50)/2 = 281.
    REQUIRE(d.cells[0].bbox[0] == Approx(231.0f));
    REQUIRE(d.cells[0].bbox[2] == Approx(281.0f));
}

TEST_CASE("decode drops border (cls 0) and header (cls 4)", "[nemotron]") {
    std::vector<float> buf;

    auto cls_border = kNoCls; cls_border[0] = logit(0.95f);
    push_query(buf, raw_query(512, 512, 100, 40, logit(0.95f), cls_border));

    auto cls_header = kNoCls; cls_header[4] = logit(0.95f);
    push_query(buf, raw_query(400, 400, 200, 60, logit(0.95f), cls_header));

    auto d = decode_nemotron(buf.data(), buf.size() / kNemotronStride, 512, 512);
    REQUIRE(d.cells.empty());
    REQUIRE(d.rows.empty());
    REQUIRE(d.columns.empty());
}

TEST_CASE("per-class NMS dedupes overlapping cells", "[nemotron]") {
    std::vector<float> buf;
    auto cls_cell = kNoCls; cls_cell[1] = logit(0.9f);
    // Two near-identical cell predictions → NMS keeps the higher-score one.
    push_query(buf, raw_query(512, 512, 100, 40, logit(0.95f), cls_cell));
    auto cls_cell2 = kNoCls; cls_cell2[1] = logit(0.8f);
    push_query(buf, raw_query(515, 510, 100, 40, logit(0.95f), cls_cell2));
    auto d = decode_nemotron(buf.data(), buf.size() / kNemotronStride, 1024, 1024);
    REQUIRE(d.cells.size() == 1);
}

TEST_CASE("assemble single row two columns", "[nemotron]") {
    NemotronDecode d;
    // One row band, two columns, two cells.
    d.rows.push_back({{0, 0, 1000, 50}, 0.9f, 2});
    d.columns.push_back({{0, 0, 500, 50}, 0.9f, 3});
    d.columns.push_back({{500, 0, 1000, 50}, 0.9f, 3});
    d.cells.push_back({{10, 5, 490, 45}, 0.9f, 1});
    d.cells.push_back({{510, 5, 990, 45}, 0.9f, 1});

    auto a = assemble_nemotron(d);
    REQUIRE(a.num_rows == 1);
    REQUIRE(a.num_cols == 2);
    REQUIRE(a.cells.size() == 2);
    REQUIRE(a.cells[0].row_idx == 0);
    REQUIRE(a.cells[0].col_idx == 0);
    REQUIRE(a.cells[0].colspan == 1);
    REQUIRE(a.cells[1].col_idx == 1);

    auto html = nemotron_to_html(a, {"A", "B"});
    REQUIRE(html.find("<tr><td>A</td><td>B</td></tr>") != std::string::npos);
}

TEST_CASE("assemble three rows two columns with colspan header",
          "[nemotron]") {
    NemotronDecode d;
    // Three row bands.
    d.rows.push_back({{0, 0,   1000, 40}, 0.9f, 2});
    d.rows.push_back({{0, 50,  1000, 90}, 0.9f, 2});
    d.rows.push_back({{0, 100, 1000, 140}, 0.9f, 2});
    d.columns.push_back({{0,   0, 500,  140}, 0.9f, 3});
    d.columns.push_back({{500, 0, 1000, 140}, 0.9f, 3});
    // Row 0: a single header cell spanning both columns.
    d.cells.push_back({{10, 5, 990, 35}, 0.9f, 1});
    // Rows 1, 2 have two cells each.
    d.cells.push_back({{10,  55, 490,  85}, 0.9f, 1});
    d.cells.push_back({{510, 55, 990,  85}, 0.9f, 1});
    d.cells.push_back({{10, 105, 490, 135}, 0.9f, 1});
    d.cells.push_back({{510,105, 990, 135}, 0.9f, 1});

    auto a = assemble_nemotron(d);
    REQUIRE(a.num_rows == 3);
    REQUIRE(a.num_cols == 2);
    REQUIRE(a.cells.size() == 5);
    REQUIRE(a.cells[0].colspan == 2);
    REQUIRE(a.cells[0].row_idx == 0);

    std::vector<std::string> texts(5, "x");
    texts[0] = "Header";
    auto html = nemotron_to_html(a, texts);
    REQUIRE(html.find("colspan=\"2\"") != std::string::npos);
    REQUIRE(html.find("Header") != std::string::npos);
}

TEST_CASE("missing cell detection falls back to row x column grid",
          "[nemotron]") {
    NemotronDecode d;
    d.rows.push_back({{0, 0, 1000, 50}, 0.9f, 2});
    d.columns.push_back({{0,   0, 500,  50}, 0.9f, 3});
    d.columns.push_back({{500, 0, 1000, 50}, 0.9f, 3});
    // Only one cell detected (left side); right slot must still appear.
    d.cells.push_back({{10, 5, 490, 45}, 0.9f, 1});

    auto a = assemble_nemotron(d);
    REQUIRE(a.cells.size() == 2);
    REQUIRE(a.cells[1].col_idx == 1);
    // Synthetic cell uses the row×column intersection bbox.
    REQUIRE(a.cells[1].bbox[0] == Approx(500.0f));
    REQUIRE(a.cells[1].bbox[2] == Approx(1000.0f));
}

TEST_CASE("html escaping protects against angle brackets in OCR text",
          "[nemotron]") {
    NemotronDecode d;
    d.rows.push_back({{0, 0, 100, 50}, 0.9f, 2});
    d.columns.push_back({{0, 0, 100, 50}, 0.9f, 3});
    d.cells.push_back({{0, 0, 100, 50}, 0.9f, 1});
    auto a = assemble_nemotron(d);
    auto html = nemotron_to_html(a, {"<script>&amp;"});
    REQUIRE(html.find("&lt;script&gt;&amp;amp;") != std::string::npos);
}

TEST_CASE("empty assembly emits well-formed empty table", "[nemotron]") {
    NemotronDecode d;
    auto a = assemble_nemotron(d);
    REQUIRE(a.cells.empty());
    auto html = nemotron_to_html(a, {});
    REQUIRE(html == "<html><body><table></table></body></html>");
}
