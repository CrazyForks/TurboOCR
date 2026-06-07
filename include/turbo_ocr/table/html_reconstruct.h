#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/table/cell_matcher.h"

namespace turbo_ocr::table {

// Build the final <html>...</html> string by walking `structure` (already
// wrapped with <html><body><table> ... by SLANeXt) and substituting OCR text
// into each <td> slot in order.
//
// `ocr_texts` is the original OCR strings. `cells[i]` corresponds to the i-th
// <td>-family token in the structure stream.
std::string reconstruct_html(
    const std::vector<std::string>& structure,
    const std::vector<MatchedCell>& cells,
    const std::vector<std::string>& ocr_texts);

} // namespace turbo_ocr::table
