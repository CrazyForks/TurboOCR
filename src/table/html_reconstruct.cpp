#include "turbo_ocr/table/html_reconstruct.h"

#include <algorithm>

namespace turbo_ocr::table {

namespace {

bool ends_with(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size() &&
           std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

bool contains(const std::string& s, std::string_view needle) {
    return s.find(needle) != std::string::npos;
}

} // namespace

std::string reconstruct_html(
    const std::vector<std::string>& structure,
    const std::vector<MatchedCell>& cells,
    const std::vector<std::string>& ocr_texts) {
    std::string out;
    const std::size_t n = structure.size();
    if (n < 6) {
        // Without the 3-tag wrapper on each side we can't slice — caller has
        // an unwrapped or trivially short stream; return concatenation.
        for (const auto& t : structure) out += t;
        return out;
    }

    for (std::size_t i = 0; i < 3; ++i) out += structure[i];

    std::size_t td_index = 0;
    for (std::size_t k = 3; k + 3 < n; ++k) {
        const std::string& tag = structure[k];
        const bool is_close = contains(tag, "</td>");
        const bool is_open_close = (tag == "<td></td>");
        if (is_close || is_open_close) {
            if (is_open_close) out += "<td>";
            if (td_index < cells.size()) {
                const MatchedCell& matched = cells[td_index];
                const bool multi = matched.ocr_indices.size() > 1;
                bool wrap_b = false;
                if (multi && !matched.ocr_indices.empty()) {
                    const std::size_t first = matched.ocr_indices.front();
                    if (first < ocr_texts.size() && contains(ocr_texts[first], "<b>")) {
                        wrap_b = true;
                        out += "<b>";
                    }
                }
                const std::size_t nm = matched.ocr_indices.size();
                for (std::size_t i_match = 0; i_match < nm; ++i_match) {
                    const std::size_t idx = matched.ocr_indices[i_match];
                    std::string content =
                        idx < ocr_texts.size() ? ocr_texts[idx] : std::string{};
                    if (multi) {
                        if (content.empty()) continue;
                        if (!content.empty() && content.front() == ' ') {
                            content.erase(content.begin());
                        }
                        if (content.rfind("<b>", 0) == 0) {
                            content.erase(0, 3);
                        }
                        if (ends_with(content, "</b>")) {
                            content.resize(content.size() - 4);
                        }
                        if (content.empty()) continue;
                        if (i_match + 1 != nm &&
                            (content.empty() || content.back() != ' ')) {
                            content.push_back(' ');
                        }
                    }
                    out += content;
                }
                if (wrap_b) out += "</b>";
            }
            if (is_open_close) {
                out += "</td>";
            } else {
                out += tag;
            }
            ++td_index;
        } else {
            out += tag;
        }
    }

    for (std::size_t i = n - 3; i < n; ++i) out += structure[i];
    return out;
}

} // namespace turbo_ocr::table
