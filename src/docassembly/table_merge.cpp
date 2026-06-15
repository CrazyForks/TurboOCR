#include "turbo_ocr/docassembly/table_merge.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <set>

namespace turbo_ocr::docassembly {

namespace {

struct Cell {
  int colspan = 1;
  int rowspan = 1;
  std::string text;  // inner text, tags stripped
};
using Row = std::vector<Cell>;

bool ieq_at(const std::string& s, std::size_t i, const char* lit) {
  for (std::size_t k = 0; lit[k]; ++k) {
    if (i + k >= s.size()) return false;
    if (std::tolower(static_cast<unsigned char>(s[i + k])) != lit[k]) return false;
  }
  return true;
}

int attr_int(const std::string& tag, const char* name, int def) {
  std::size_t p = tag.find(name);
  if (p == std::string::npos) return def;
  p += std::string(name).size();
  while (p < tag.size() &&
         (tag[p] == ' ' || tag[p] == '=' || tag[p] == '"' || tag[p] == '\''))
    ++p;
  int val = 0;
  bool any = false;
  while (p < tag.size() && std::isdigit(static_cast<unsigned char>(tag[p]))) {
    val = val * 10 + (tag[p] - '0');
    ++p;
    any = true;
  }
  return any ? val : def;
}

std::string strip_tags(const std::string& html) {
  std::string out;
  bool in_tag = false;
  for (char c : html) {
    if (c == '<') in_tag = true;
    else if (c == '>') in_tag = false;
    else if (!in_tag) out.push_back(c);
  }
  return out;
}

// Raw "<tr>...</tr>" spans (top-level rows of a table HTML string).
std::vector<std::string> extract_tr_raw(const std::string& html) {
  std::vector<std::string> rows;
  std::size_t i = 0;
  while (i < html.size()) {
    if (html[i] == '<' && ieq_at(html, i, "<tr")) {
      std::size_t close = html.find("</tr>", i);
      if (close == std::string::npos) break;
      std::size_t end = close + 5;
      rows.push_back(html.substr(i, end - i));
      i = end;
    } else {
      ++i;
    }
  }
  return rows;
}

Row extract_cells(const std::string& tr_html) {
  Row cells;
  std::size_t i = 0;
  while (i < tr_html.size()) {
    bool td = ieq_at(tr_html, i, "<td");
    bool th = ieq_at(tr_html, i, "<th");
    if (tr_html[i] == '<' && (td || th)) {
      std::size_t tag_end = tr_html.find('>', i);
      if (tag_end == std::string::npos) break;
      std::string open_tag = tr_html.substr(i, tag_end - i);
      const char* close_lit = td ? "</td>" : "</th>";
      std::size_t close = tr_html.find(close_lit, tag_end);
      std::size_t inner_end = (close == std::string::npos) ? tr_html.size() : close;
      std::string inner = tr_html.substr(tag_end + 1, inner_end - tag_end - 1);
      Cell c;
      c.colspan = attr_int(open_tag, "colspan", 1);
      c.rowspan = attr_int(open_tag, "rowspan", 1);
      c.text = strip_tags(inner);
      cells.push_back(std::move(c));
      i = (close == std::string::npos) ? tr_html.size() : close + 5;
    } else {
      ++i;
    }
  }
  return cells;
}

std::vector<Row> parse_rows(const std::string& html) {
  std::vector<Row> rows;
  for (const auto& tr : extract_tr_raw(html)) rows.push_back(extract_cells(tr));
  return rows;
}

int calc_total_columns(const std::vector<Row>& rows) {
  if (rows.empty()) return 0;
  int max_cols = 0;
  std::vector<std::set<int>> occupied(rows.size());
  for (std::size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
    int col_idx = 0;
    for (const auto& cell : rows[row_idx]) {
      while (occupied[row_idx].count(col_idx)) ++col_idx;
      for (int r = static_cast<int>(row_idx);
           r < static_cast<int>(row_idx) + cell.rowspan; ++r) {
        if (r >= static_cast<int>(occupied.size())) occupied.resize(r + 1);
        for (int c = col_idx; c < col_idx + cell.colspan; ++c) occupied[r].insert(c);
      }
      col_idx += cell.colspan;
      max_cols = std::max(max_cols, col_idx);
    }
  }
  return max_cols;
}

int calc_row_columns(const Row& row) {
  int s = 0;
  for (const auto& c : row) s += c.colspan;
  return s;
}

int calc_visual_columns(const Row& row) { return static_cast<int>(row.size()); }

std::string norm_text(const std::string& s) {
  std::string fh = full_to_half(s);
  std::string out;
  for (char c : fh)
    if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
  return out;
}

// Returns (header_rows, headers_match).
std::pair<int, bool> detect_table_headers(const std::vector<Row>& r1,
                                          const std::vector<Row>& r2,
                                          int max_header_rows = 5) {
  int min_rows = std::min({static_cast<int>(r1.size()), static_cast<int>(r2.size()),
                           max_header_rows});
  int header_rows = 0;
  bool headers_match = true;
  for (int i = 0; i < min_rows; ++i) {
    if (r1[i].size() != r2[i].size()) {
      headers_match = header_rows > 0;
      break;
    }
    bool match = true;
    for (std::size_t k = 0; k < r1[i].size(); ++k) {
      if (norm_text(r1[i][k].text) != norm_text(r2[i][k].text) ||
          r1[i][k].colspan != r2[i][k].colspan) {
        match = false;
        break;
      }
    }
    if (match) ++header_rows;
    else { headers_match = header_rows > 0; break; }
  }
  if (header_rows == 0) headers_match = false;
  return {header_rows, headers_match};
}

bool check_rows_match(const std::vector<Row>& r1, const std::vector<Row>& r2) {
  if (r1.empty() || r2.empty()) return false;
  const Row& last_row = r1.back();
  int header_count = detect_table_headers(r1, r2).first;
  if (static_cast<int>(r2.size()) <= header_count) return false;
  const Row& first_data_row = r2[header_count];
  return calc_row_columns(last_row) == calc_row_columns(first_data_row) ||
         calc_visual_columns(last_row) == calc_visual_columns(first_data_row);
}

bool contains_ci(const std::string& hay, const std::string& needle) {
  if (needle.empty()) return false;
  std::string h = hay;
  for (char& c : h) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return h.find(needle) != std::string::npos;
}

bool is_skippable(const ParsingBlock& block,
                  const std::vector<std::string>& allowed_labels) {
  if (std::find(allowed_labels.begin(), allowed_labels.end(), block.label) !=
      allowed_labels.end())
    return true;
  // PaddleX inspects text/title attrs; our block carries text in `content`.
  static const char* kw[] = {"continue", "continued", "cont'd",
                             "\xe7\xbb\xad" /*续*/};
  for (const char* k : kw)
    if (contains_ci(block.content, k)) return true;
  return false;
}

int box_width(const ParsingBlock& b) { return b.bbox[2] - b.bbox[0]; }

bool can_merge_tables(const std::vector<ParsingBlock>& prev_page, std::size_t prev_idx,
                      const std::vector<ParsingBlock>& curr_page, std::size_t curr_idx) {
  const ParsingBlock& prev = prev_page[prev_idx];
  const ParsingBlock& curr = curr_page[curr_idx];
  int pw = box_width(prev), cw = box_width(curr);
  if (pw == 0 || cw == 0) return false;
  if (static_cast<double>(std::abs(cw - pw)) / std::min(cw, pw) >= 0.1) return false;

  static const std::vector<std::string> follow_ok = {
      "footer", "vision_footnote", "number", "footnote", "footer_image", "seal"};
  for (std::size_t i = prev_idx + 1; i < prev_page.size(); ++i)
    if (std::find(follow_ok.begin(), follow_ok.end(), prev_page[i].label) ==
        follow_ok.end())
      return false;

  static const std::vector<std::string> before_ok = {"header", "header_image",
                                                     "number", "seal"};
  for (std::size_t i = 0; i < curr_idx; ++i)
    if (!is_skippable(curr_page[i], before_ok)) return false;

  if (prev.content.empty() || curr.content.empty()) return false;
  auto rows_prev = parse_rows(prev.content);
  auto rows_curr = parse_rows(curr.content);
  bool tables_match = calc_total_columns(rows_prev) == calc_total_columns(rows_curr);
  bool rows_match = check_rows_match(rows_prev, rows_curr);
  return tables_match || rows_match;
}

std::string perform_table_merge(const std::string& prev_html,
                                const std::string& curr_html) {
  auto rows_prev = parse_rows(prev_html);
  auto rows_curr = parse_rows(curr_html);
  int header_count = detect_table_headers(rows_prev, rows_curr).first;
  auto curr_raw = extract_tr_raw(curr_html);

  std::string appended;
  for (std::size_t i = static_cast<std::size_t>(header_count); i < curr_raw.size(); ++i)
    appended += curr_raw[i];
  if (appended.empty()) return prev_html;

  std::size_t pos = prev_html.rfind("</tr>");
  std::size_t insert_at = (pos != std::string::npos)
                              ? pos + 5
                              : prev_html.find("</table>");
  if (insert_at == std::string::npos) return prev_html + appended;
  return prev_html.substr(0, insert_at) + appended + prev_html.substr(insert_at);
}

} // namespace

std::string full_to_half(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if ((c >> 4) == 0xE && i + 2 < text.size()) {
      std::uint32_t cp = ((c & 0x0Fu) << 12) |
                         ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6) |
                         (static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
      if (cp >= 0xFF01 && cp <= 0xFF5E) {
        out.push_back(static_cast<char>(cp - 0xFEE0));
        i += 3;
        continue;
      }
      out.append(text, i, 3);
      i += 3;
    } else {
      out.push_back(text[i]);
      ++i;
    }
  }
  return out;
}

void merge_tables_across_pages(std::vector<std::vector<ParsingBlock>>& pages) {
  // Assign flattened-order ids (restructure_pages does this before merge).
  int gid = 0;
  std::vector<ParsingBlock*> all_blocks;
  for (auto& page : pages)
    for (auto& b : page) {
      b.global_block_id = gid;
      b.global_group_id = gid;
      all_blocks.push_back(&b);
      ++gid;
    }

  for (int i = static_cast<int>(pages.size()) - 1; i > 0; --i) {
    auto& page_curr = pages[i];
    auto& page_prev = pages[i - 1];

    int curr_idx = -1;
    for (std::size_t j = 0; j < page_curr.size(); ++j)
      if (page_curr[j].label == "table") { curr_idx = static_cast<int>(j); break; }
    int prev_idx = -1;
    for (int j = static_cast<int>(page_prev.size()) - 1; j >= 0; --j)
      if (page_prev[j].label == "table") { prev_idx = j; break; }

    if (curr_idx < 0 || prev_idx < 0) continue;
    if (!can_merge_tables(page_prev, prev_idx, page_curr, curr_idx)) continue;

    ParsingBlock& prev_block = page_prev[prev_idx];
    ParsingBlock& curr_block = page_curr[curr_idx];
    prev_block.content = perform_table_merge(prev_block.content, curr_block.content);
    curr_block.content.clear();
    curr_block.global_group_id = prev_block.global_block_id;
  }

  // Resolve chains: a block folded into another inherits that host's group.
  for (auto& page : pages)
    for (auto& b : page)
      if (b.global_block_id != b.global_group_id)
        b.global_group_id = all_blocks[b.global_group_id]->global_group_id;
}

} // namespace turbo_ocr::docassembly
