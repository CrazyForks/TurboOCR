#include "turbo_ocr/table/slanext_dict.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace turbo_ocr::table {

// Verbatim PaddleX table_structure_dict_ch.txt — 48 base tokens.
// Whitespace and quoting are byte-for-byte; alteration breaks SLANeXt
// vocab alignment.
const std::string_view DEFAULT_DICT_TEXT =
    "<thead>\n"
    "</thead>\n"
    "<tr>\n"
    "</tr>\n"
    "<td>\n"
    "</td>\n"
    "<td\n"
    ">\n"
    " colspan=\"2\"\n"
    " colspan=\"3\"\n"
    " colspan=\"4\"\n"
    " colspan=\"5\"\n"
    " colspan=\"6\"\n"
    " colspan=\"7\"\n"
    " colspan=\"8\"\n"
    " colspan=\"9\"\n"
    " colspan=\"10\"\n"
    " colspan=\"11\"\n"
    " colspan=\"12\"\n"
    " colspan=\"13\"\n"
    " colspan=\"14\"\n"
    " colspan=\"15\"\n"
    " colspan=\"16\"\n"
    " colspan=\"17\"\n"
    " colspan=\"18\"\n"
    " colspan=\"19\"\n"
    " colspan=\"25\"\n"
    " rowspan=\"2\"\n"
    " rowspan=\"3\"\n"
    " rowspan=\"4\"\n"
    " rowspan=\"5\"\n"
    " rowspan=\"6\"\n"
    " rowspan=\"7\"\n"
    " rowspan=\"8\"\n"
    " rowspan=\"9\"\n"
    " rowspan=\"10\"\n"
    " rowspan=\"11\"\n"
    " rowspan=\"12\"\n"
    " rowspan=\"13\"\n"
    " rowspan=\"14\"\n"
    " rowspan=\"15\"\n"
    " rowspan=\"16\"\n"
    " rowspan=\"17\"\n"
    " rowspan=\"18\"\n"
    " rowspan=\"19\"\n"
    " rowspan=\"20\"\n"
    "<tbody>\n"
    "</tbody>\n";

CharDict CharDict::build(std::vector<std::string> base) {
    if (std::none_of(base.begin(), base.end(),
                     [](const std::string& s) { return s == "<td></td>"; })) {
        base.emplace_back("<td></td>");
    }
    base.erase(std::remove(base.begin(), base.end(), std::string("<td>")),
               base.end());

    CharDict d;
    d.chars_.reserve(base.size() + 2);
    d.chars_.emplace_back("sos");
    for (auto& t : base) d.chars_.emplace_back(std::move(t));
    d.chars_.emplace_back("eos");
    return d;
}

CharDict CharDict::from_text(std::string_view text) {
    std::vector<std::string> base;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t nl = text.find('\n', i);
        std::string line(text.substr(i, (nl == std::string_view::npos
                                             ? text.size()
                                             : nl) - i));
        if (!line.empty()) base.push_back(std::move(line));
        if (nl == std::string_view::npos) break;
        i = nl + 1;
    }
    return build(std::move(base));
}

CharDict CharDict::from_path(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("CharDict: could not open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return from_text(ss.str());
}

std::string_view CharDict::token(std::size_t idx) const {
    if (idx >= chars_.size()) return {};
    return chars_[idx];
}

bool CharDict::is_td_token(std::size_t idx) const {
    if (idx >= chars_.size()) return false;
    const std::string& t = chars_[idx];
    return t == "<td>" || t == "<td" || t == "<td></td>";
}

CharDict default_dict() {
    if (const char* override = std::getenv("TURBO_OCR_TABLE_DICT_PATH")) {
        if (*override) return CharDict::from_path(override);
    }
    return CharDict::from_text(DEFAULT_DICT_TEXT);
}

} // namespace turbo_ocr::table
