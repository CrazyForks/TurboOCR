#include <catch_amalgamated.hpp>

#include <array>
#include <string>
#include <vector>

#include "turbo_ocr/router/router_types.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"

using turbo_ocr::Box;
using turbo_ocr::router::FormulaResult;
using turbo_ocr::router::FormulaWrap;
using turbo_ocr::table::MatchedCell;
using turbo_ocr::table::reconstruct_html;

namespace {

std::vector<std::string> structure(std::vector<std::string> body) {
    std::vector<std::string> out = {"<html>", "<body>", "<table>"};
    for (auto& b : body) out.push_back(std::move(b));
    out.insert(out.end(), {"</table>", "</body>", "</html>"});
    return out;
}

MatchedCell make_cell(std::array<float, 4> bbox,
                      std::vector<std::size_t> indices) {
    return MatchedCell{bbox, std::move(indices)};
}

Box box_from_aabb(float x0, float y0, float x1, float y1) {
    Box b{};
    b[0] = {static_cast<int>(x0), static_cast<int>(y0)};
    b[1] = {static_cast<int>(x1), static_cast<int>(y0)};
    b[2] = {static_cast<int>(x1), static_cast<int>(y1)};
    b[3] = {static_cast<int>(x0), static_cast<int>(y1)};
    return b;
}

FormulaResult make_formula(float x0, float y0, float x1, float y1,
                           std::string latex, FormulaWrap wrap) {
    FormulaResult f{};
    f.box = box_from_aabb(x0, y0, x1, y1);
    f.latex = std::move(latex);
    f.wrap = wrap;
    return f;
}

} // namespace

TEST_CASE("inline formula absorbed into single cell", "[html_reconstruct][formula]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 20}, {0})};
    std::vector<std::string> texts = {"x ="};
    std::vector<std::array<float, 4>> ocr_aabbs = {{0, 0, 30, 20}};
    std::vector<FormulaResult> formulas = {
        make_formula(40, 0, 100, 20, "y^2", FormulaWrap::Inline)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("x =") != std::string::npos);
    REQUIRE(html.find("$y^2$") != std::string::npos);
    REQUIRE(formulas[0].owned_by_cell == true);
}

TEST_CASE("display formula in cell wraps with $$", "[html_reconstruct][formula]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 200, 80}, {})};
    std::vector<std::string> texts;
    std::vector<std::array<float, 4>> ocr_aabbs;
    std::vector<FormulaResult> formulas = {
        make_formula(10, 10, 190, 70, "\\int f(x) dx", FormulaWrap::Display)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("$$\\int f(x) dx$$") != std::string::npos);
    REQUIRE(formulas[0].owned_by_cell == true);
}

TEST_CASE("partial overlap above threshold absorbs formula",
          "[html_reconstruct][formula]") {
    // Cell area 100x100 = 10000; formula 80x80 inset by (10,10) = 6400.
    // Intersection 80*80 = 6400; union = 10000+6400-6400 = 10000; IoU = 0.64.
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 100}, {})};
    std::vector<std::string> texts;
    std::vector<std::array<float, 4>> ocr_aabbs;
    std::vector<FormulaResult> formulas = {
        make_formula(10, 10, 90, 90, "a+b", FormulaWrap::Inline)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("$a+b$") != std::string::npos);
    REQUIRE(formulas[0].owned_by_cell == true);
}

TEST_CASE("partial overlap below threshold is NOT absorbed",
          "[html_reconstruct][formula]") {
    // Cell 100x100; formula 50x50 starting at (60,60) -> intersection 40x40=1600;
    // union 10000 + 2500 - 1600 = 10900; IoU ~= 0.147 (< 0.5).
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 100}, {0})};
    std::vector<std::string> texts = {"hello"};
    std::vector<std::array<float, 4>> ocr_aabbs = {{0, 0, 50, 20}};
    std::vector<FormulaResult> formulas = {
        make_formula(60, 60, 110, 110, "z", FormulaWrap::Inline)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("hello") != std::string::npos);
    REQUIRE(html.find("$z$") == std::string::npos);
    REQUIRE(formulas[0].owned_by_cell == false);
}

TEST_CASE("formula outside any cell is not marked owned_by_cell",
          "[html_reconstruct][formula]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 20}, {0})};
    std::vector<std::string> texts = {"caption"};
    std::vector<std::array<float, 4>> ocr_aabbs = {{0, 0, 80, 20}};
    std::vector<FormulaResult> formulas = {
        make_formula(500, 500, 600, 540, "p", FormulaWrap::Inline)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("caption") != std::string::npos);
    REQUIRE(html.find("$p$") == std::string::npos);
    REQUIRE(formulas[0].owned_by_cell == false);
}

TEST_CASE("OCR line covered by absorbed formula is dropped from cell text",
          "[html_reconstruct][formula]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 20}, {0, 1})};
    // OCR 0 lives inside the formula; OCR 1 sits to the right.
    std::vector<std::string> texts = {"y^2", "result"};
    std::vector<std::array<float, 4>> ocr_aabbs = {{10, 5, 50, 18},
                                                    {60, 0, 95, 20}};
    std::vector<FormulaResult> formulas = {
        make_formula(0, 0, 60, 20, "y^2", FormulaWrap::Inline)};

    auto html = reconstruct_html(s, cells, texts, ocr_aabbs, formulas);

    REQUIRE(html.find("$y^2$") != std::string::npos);
    REQUIRE(html.find("result") != std::string::npos);
    // The first OCR line ("y^2" as raw det text) is covered by the formula.
    // It should appear inside the formula wrapping, not as a standalone
    // segment before/after it.
    auto pos = html.find("y^2");
    REQUIRE(pos != std::string::npos);
    // The non-formula "y^2" appearance (without $) should not be in the cell.
    auto count_substr = [&](const std::string& s) {
        size_t cnt = 0, p = 0;
        while ((p = html.find(s, p)) != std::string::npos) { ++cnt; ++p; }
        return cnt;
    };
    REQUIRE(count_substr("y^2") == 1);
}

TEST_CASE("legacy 3-arg overload still works", "[html_reconstruct]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<std::string> texts = {"hello"};
    std::vector<MatchedCell> cells = {make_cell({0, 0, 100, 20}, {0})};
    auto html = reconstruct_html(s, cells, texts);
    REQUIRE(html ==
            "<html><body><table><tr><td>hello</td></tr></table></body></html>");
}
