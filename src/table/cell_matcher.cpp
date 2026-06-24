#include "turbo_ocr/table/cell_matcher.h"

#include <algorithm>
#include <cmath>

#ifdef __has_include
#  if __has_include(<boost/geometry.hpp>)
#    define TURBO_OCR_HAS_BOOST_GEOMETRY 1
#  endif
#endif

#ifdef TURBO_OCR_HAS_BOOST_GEOMETRY
#  include <boost/geometry.hpp>
#  include <boost/geometry/index/rtree.hpp>
#endif

namespace turbo_ocr::table {

namespace {
// Row-band height for within-cell reading order. Quantizing y into bands of
// this size yields a TOTAL order over (band, x), which std::sort requires —
// the raw tolerance comparison is non-transitive (strict-weak-ordering UB).
constexpr float READING_ORDER_BAND_PX = 8.0f;
} // namespace

float compute_inter(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x_left = std::max(a[0], b[0]);
    const float y_top = std::max(a[1], b[1]);
    const float x_right = std::min(a[2], b[2]);
    const float y_bottom = std::min(a[3], b[3]);
    const float iw = std::max(x_right - x_left, 0.0f);
    const float ih = std::max(y_bottom - y_top, 0.0f);
    const float inter = iw * ih;
    const float area_b = std::max(b[2] - b[0], 0.0f) * std::max(b[3] - b[1], 0.0f);
    return area_b > 0.0f ? inter / area_b : 0.0f;
}

std::array<float, 4> quad_to_bbox(const std::array<int, 8>& quad) {
    const int xs[4] = {quad[0], quad[2], quad[4], quad[6]};
    const int ys[4] = {quad[1], quad[3], quad[5], quad[7]};
    int xmin = xs[0], xmax = xs[0];
    int ymin = ys[0], ymax = ys[0];
    for (int i = 1; i < 4; ++i) {
        xmin = std::min(xmin, xs[i]);
        xmax = std::max(xmax, xs[i]);
        ymin = std::min(ymin, ys[i]);
        ymax = std::max(ymax, ys[i]);
    }
    return {static_cast<float>(xmin), static_cast<float>(ymin),
            static_cast<float>(xmax), static_cast<float>(ymax)};
}

#ifdef TURBO_OCR_HAS_BOOST_GEOMETRY

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using BPoint = bg::model::point<float, 2, bg::cs::cartesian>;
using BBox = bg::model::box<BPoint>;
using BValue = std::pair<BBox, std::size_t>;
using RTree = bgi::rtree<BValue, bgi::rstar<16>>;

std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr) {
    std::vector<BValue> entries;
    entries.reserve(ocr.size());
    for (std::size_t i = 0; i < ocr.size(); ++i) {
        const auto& b = ocr[i].bbox;
        entries.emplace_back(
            BBox(BPoint(b[0], b[1]), BPoint(b[2], b[3])), i);
    }
    RTree tree(entries.begin(), entries.end());

    std::vector<MatchedCell> out;
    out.reserve(cells.size());
    std::vector<char> assigned(ocr.size(), 0);  // global: did this line land in any cell?
    for (const auto& quad : cells) {
        const auto cell_bbox = quad_to_bbox(quad);
        const BBox query(BPoint(cell_bbox[0], cell_bbox[1]),
                         BPoint(cell_bbox[2], cell_bbox[3]));
        std::vector<std::size_t> indices;
        for (auto it = tree.qbegin(bgi::intersects(query));
             it != tree.qend(); ++it) {
            const std::size_t i = it->second;
            if (compute_inter(cell_bbox, ocr[i].bbox) > MATCH_INTER_THRESHOLD) {
                indices.push_back(i);
                assigned[i] = 1;
            }
        }
        // Reading order within the cell: top-to-bottom by row band (8px
        // tolerance), then left-to-right — so multi-line cell text joins correctly
        // regardless of page-global OCR order.
        std::sort(indices.begin(), indices.end(),
                  [&ocr](std::size_t a, std::size_t b) {
                      const auto& ba = ocr[a].bbox; const auto& bb = ocr[b].bbox;
                      const int ra = static_cast<int>(std::floor(ba[1] / READING_ORDER_BAND_PX));
                      const int rb = static_cast<int>(std::floor(bb[1] / READING_ORDER_BAND_PX));
                      if (ra != rb) return ra < rb;
                      return ba[0] < bb[0];
                  });
        out.push_back(MatchedCell{cell_bbox, std::move(indices)});
    }
    // Fallback: never silently drop a detected OCR line. A line straddling cell
    // boundaries (dense numeric grids) can miss the threshold for every cell — assign
    // each still-unassigned line to the cell whose quad it overlaps most.
    auto cell_order = [&ocr](std::size_t a, std::size_t b) {
        const auto& ba = ocr[a].bbox; const auto& bb = ocr[b].bbox;
        const int ra = static_cast<int>(std::floor(ba[1] / READING_ORDER_BAND_PX));
        const int rb = static_cast<int>(std::floor(bb[1] / READING_ORDER_BAND_PX));
        if (ra != rb) return ra < rb;
        return ba[0] < bb[0];
    };
    for (std::size_t i = 0; i < ocr.size(); ++i) {
        if (assigned[i]) continue;
        float best = 0.0f;
        std::size_t bestc = out.size();
        for (std::size_t c = 0; c < out.size(); ++c) {
            const float r = compute_inter(out[c].bbox, ocr[i].bbox);
            if (r > best) { best = r; bestc = c; }
        }
        // Only rescue lines genuinely straddling internal cell boundaries (>=15% of
        // the line in its best cell); a line mostly outside the table stays unassigned.
        if (bestc < out.size() && best > 0.15f) {
            out[bestc].ocr_indices.push_back(i);
            std::sort(out[bestc].ocr_indices.begin(),
                      out[bestc].ocr_indices.end(), cell_order);
            assigned[i] = 1;
        }
    }
    return out;
}

#else

// 16-bucket grid fallback when Boost.Geometry isn't available. Bucket count is
// fixed; cell→bucket assignment is by axis-aligned envelope. Sub-ms either
// way for the expected ≤500 cells × ≤200 OCR lines.
namespace {
constexpr int GRID_N = 16;

struct Grid {
    float x0{0}, y0{0}, x1{0}, y1{0};
    std::vector<std::size_t> buckets[GRID_N][GRID_N];

    void build(const std::vector<OcrLine>& ocr) {
        if (ocr.empty()) return;
        x0 = ocr[0].bbox[0]; y0 = ocr[0].bbox[1];
        x1 = ocr[0].bbox[2]; y1 = ocr[0].bbox[3];
        for (const auto& l : ocr) {
            x0 = std::min(x0, l.bbox[0]);
            y0 = std::min(y0, l.bbox[1]);
            x1 = std::max(x1, l.bbox[2]);
            y1 = std::max(y1, l.bbox[3]);
        }
        for (std::size_t i = 0; i < ocr.size(); ++i) {
            int bx0, by0, bx1, by1;
            range(ocr[i].bbox, bx0, by0, bx1, by1);
            for (int gy = by0; gy <= by1; ++gy)
                for (int gx = bx0; gx <= bx1; ++gx)
                    buckets[gy][gx].push_back(i);
        }
    }

    void range(const std::array<float, 4>& b,
               int& bx0, int& by0, int& bx1, int& by1) const {
        const float dx = std::max(x1 - x0, 1.0f);
        const float dy = std::max(y1 - y0, 1.0f);
        auto bucket_x = [&](float v) {
            int g = static_cast<int>(((v - x0) / dx) * GRID_N);
            return std::clamp(g, 0, GRID_N - 1);
        };
        auto bucket_y = [&](float v) {
            int g = static_cast<int>(((v - y0) / dy) * GRID_N);
            return std::clamp(g, 0, GRID_N - 1);
        };
        bx0 = bucket_x(b[0]);
        by0 = bucket_y(b[1]);
        bx1 = bucket_x(b[2]);
        by1 = bucket_y(b[3]);
    }
};
} // namespace

std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr) {
    Grid grid;
    grid.build(ocr);

    std::vector<MatchedCell> out;
    out.reserve(cells.size());
    std::vector<char> assigned(ocr.size(), 0);  // global: did this line land in any cell?
    for (const auto& quad : cells) {
        const auto cell_bbox = quad_to_bbox(quad);
        std::vector<std::size_t> indices;
        if (ocr.empty()) {
            out.push_back(MatchedCell{cell_bbox, std::move(indices)});
            continue;
        }
        int bx0, by0, bx1, by1;
        grid.range(cell_bbox, bx0, by0, bx1, by1);
        std::vector<char> seen(ocr.size(), 0);
        for (int gy = by0; gy <= by1; ++gy) {
            for (int gx = bx0; gx <= bx1; ++gx) {
                for (std::size_t i : grid.buckets[gy][gx]) {
                    if (seen[i]) continue;
                    seen[i] = 1;
                    if (compute_inter(cell_bbox, ocr[i].bbox) > MATCH_INTER_THRESHOLD) {
                        indices.push_back(i);
                        assigned[i] = 1;
                    }
                }
            }
        }
        // Reading order within the cell: top-to-bottom by row band (8px
        // tolerance), then left-to-right — so multi-line cell text joins correctly
        // regardless of page-global OCR order.
        std::sort(indices.begin(), indices.end(),
                  [&ocr](std::size_t a, std::size_t b) {
                      const auto& ba = ocr[a].bbox; const auto& bb = ocr[b].bbox;
                      const int ra = static_cast<int>(std::floor(ba[1] / READING_ORDER_BAND_PX));
                      const int rb = static_cast<int>(std::floor(bb[1] / READING_ORDER_BAND_PX));
                      if (ra != rb) return ra < rb;
                      return ba[0] < bb[0];
                  });
        out.push_back(MatchedCell{cell_bbox, std::move(indices)});
    }
    // Fallback: never silently drop a detected OCR line (see rtree variant above).
    auto cell_order = [&ocr](std::size_t a, std::size_t b) {
        const auto& ba = ocr[a].bbox; const auto& bb = ocr[b].bbox;
        const int ra = static_cast<int>(std::floor(ba[1] / READING_ORDER_BAND_PX));
        const int rb = static_cast<int>(std::floor(bb[1] / READING_ORDER_BAND_PX));
        if (ra != rb) return ra < rb;
        return ba[0] < bb[0];
    };
    for (std::size_t i = 0; i < ocr.size(); ++i) {
        if (assigned[i]) continue;
        float best = 0.0f;
        std::size_t bestc = out.size();
        for (std::size_t c = 0; c < out.size(); ++c) {
            const float r = compute_inter(out[c].bbox, ocr[i].bbox);
            if (r > best) { best = r; bestc = c; }
        }
        // Only rescue lines genuinely straddling internal cell boundaries (>=15% of
        // the line in its best cell); a line mostly outside the table stays unassigned.
        if (bestc < out.size() && best > 0.15f) {
            out[bestc].ocr_indices.push_back(i);
            std::sort(out[bestc].ocr_indices.begin(),
                      out[bestc].ocr_indices.end(), cell_order);
            assigned[i] = 1;
        }
    }
    return out;
}

#endif

} // namespace turbo_ocr::table
