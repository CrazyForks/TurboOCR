#include <catch_amalgamated.hpp>

#include "turbo_ocr/table/html_reconstruct.h"

using turbo_ocr::table::MatchedCell;
using turbo_ocr::table::reconstruct_html;

namespace {

std::vector<std::string> structure(std::vector<std::string> body) {
    std::vector<std::string> out = {"<html>", "<body>", "<table>"};
    for (auto& b : body) out.push_back(std::move(b));
    out.insert(out.end(), {"</table>", "</body>", "</html>"});
    return out;
}

MatchedCell cell(std::vector<std::size_t> indices) {
    MatchedCell c{};
    c.ocr_indices = std::move(indices);
    return c;
}

} // namespace

TEST_CASE("empty cells round trip structure only", "[html_reconstruct]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<std::string> texts;
    std::vector<MatchedCell> cells = {cell({})};
    auto html = reconstruct_html(s, cells, texts);
    REQUIRE(html ==
            "<html><body><table><tr><td></td></tr></table></body></html>");
}

TEST_CASE("merged td inserts text", "[html_reconstruct]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<std::string> texts = {"hello"};
    std::vector<MatchedCell> cells = {cell({0})};
    auto html = reconstruct_html(s, cells, texts);
    REQUIRE(html ==
            "<html><body><table><tr><td>hello</td></tr></table></body></html>");
}

TEST_CASE("split td with attributes preserved", "[html_reconstruct]") {
    auto s = structure({"<tr>", "<td", " colspan=\"2\"", ">", "</td>", "</tr>"});
    std::vector<std::string> texts = {"hi"};
    std::vector<MatchedCell> cells = {cell({0})};
    auto html = reconstruct_html(s, cells, texts);
    REQUIRE(html.find("<td colspan=\"2\">hi</td>") != std::string::npos);
}

TEST_CASE("multi match concatenates with space and strips b",
          "[html_reconstruct]") {
    auto s = structure({"<tr>", "<td></td>", "</tr>"});
    std::vector<std::string> texts = {"<b>foo", "bar</b>"};
    std::vector<MatchedCell> cells = {cell({0, 1})};
    auto html = reconstruct_html(s, cells, texts);
    REQUIRE(html.find("<b>foo bar</b>") != std::string::npos);
}
