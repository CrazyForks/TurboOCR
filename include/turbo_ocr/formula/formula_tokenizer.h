#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace turbo_ocr::formula {

// HuggingFace-compatible byte-level BPE decoder for the PP-FormulaNet-S
// tokenizer. Only the decode path is implemented — encoding is not needed
// at inference.
class FormulaTokenizer {
public:
    static std::optional<FormulaTokenizer> load(const std::string& json_path);

    // Decode token ids into a normalised LaTeX string.
    // Trims at first EOS, skips specials, looks up id_to_token, concatenates,
    // remaps the BPE space marker 'Ġ' (UTF-8 0xC4 0xA0) to ' ', strips
    // BOS/EOS/PAD literals, trims, then applies latex_post_process.
    std::string decode(std::span<const int64_t> ids) const;

    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

    // Exposed for tests and the routing pipeline post-step.
    static std::string latex_post_process(const std::string& s);

private:
    FormulaTokenizer() = default;

    std::vector<std::string> id_to_token_;
    std::unordered_set<int64_t> special_ids_;
    int64_t eos_id_ = 2;  // "</s>"
};

}  // namespace turbo_ocr::formula
