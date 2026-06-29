#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr {

// Unified, extensible per-region result schema (design #17). A flat region
// stream — a common header + a discriminated-union content payload — modeled
// on docling's typed-modality items flattened like OmniDocBench/MinerU's
// content_list. Native per-modality content (table HTML+cells, formula LaTeX)
// instead of a re-parsed string blob; std::visit gives compile-time
// exhaustiveness. Adding a modality = one RegionType value + one variant arm.
//
// This is an ADDITIVE external schema. The internal router::TableResult /
// router::FormulaResult + the layout_id-keyed tables[]/formulas[] JSON arrays
// remain the load-bearing contract (omnidoc_to_md.py joins on them) and are
// unchanged. The regions[] emitter is flag-gated (REGIONS_OUTPUT) and built
// lazily, so existing consumers see zero change.
enum class RegionType : uint8_t {
  Text, Title, Table, Formula, Figure, Caption, List, Code, Header, Footer, Other
};

struct TextContent    { std::string text; int level = 0; };  // 0=body, >=1 heading
struct TableCellRO    { int row = 0, col = 0, row_span = 1, col_span = 1;
                        std::string text; bool is_header = false; };
struct TableContent   { std::string html; std::vector<TableCellRO> cells;
                        int num_rows = 0, num_cols = 0; };
struct FormulaContent { std::string latex; };
struct FigureContent  { std::string image_ref; std::optional<uint32_t> caption_id; };

struct Region {
  uint32_t       id = 0;
  RegionType     type = RegionType::Other;  // discriminant
  Box            bbox{};                     // existing 4-corner Box (carries poly)
  uint32_t       page_index = 0;
  int32_t        reading_order = -1;
  float          confidence = 0.0f;
  std::optional<uint32_t> parent_id;         // caption<->figure etc.
  std::variant<TextContent, TableContent, FormulaContent, FigureContent> content;
};

// Map a layout class label (layout_types.h) to a RegionType.
inline RegionType region_type_from_label(std::string_view label) {
  if (label == "table") return RegionType::Table;
  if (label == "display_formula" || label == "inline_formula" ||
      label == "formula" || label == "formula_number")
    return RegionType::Formula;
  if (label == "doc_title" || label == "paragraph_title" || label == "figure_title")
    return RegionType::Title;
  if (label == "image" || label == "chart" || label == "figure")
    return RegionType::Figure;
  if (label == "abstract" || label == "text" || label == "content" ||
      label == "vertical_text")
    return RegionType::Text;
  return RegionType::Other;
}

} // namespace turbo_ocr
