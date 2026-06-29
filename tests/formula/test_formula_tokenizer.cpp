// Hand-curated parity tests for FormulaTokenizer::latex_post_process.
//
// We can't link the Rust tokenizers crate from C++, so the BPE decode itself
// is exercised only through the optional `loads_real_tokenizer_if_present`
// path. The 30+ cases below cover the documented behavior of the Rust
// `tokenizer.rs::latex_post_process`: whitespace collapsing around
// non-letter symbols, backslash-space preservation, operatorname/mathrm/
// text/mathbf wrapper stash-and-restore, and CJK \text{...} unwrapping.

#include "catch_amalgamated.hpp"
#include "turbo_ocr/formula/formula_tokenizer.h"

using turbo_ocr::formula::FormulaTokenizer;

namespace {
std::string pp(const std::string& s) {
    return FormulaTokenizer::latex_post_process(s);
}
}  // namespace

TEST_CASE("latex_post_process: arithmetic collapses spaces", "[formula_tokenizer]") {
    REQUIRE(pp("a + b = c") == "a+b=c");
    REQUIRE(pp("a + b") == "a+b");
    REQUIRE(pp("x = y") == "x=y");
    REQUIRE(pp("a , b") == "a,b");
    REQUIRE(pp("a ; b") == "a;b");
    REQUIRE(pp("1 + 2") == "1+2");
    REQUIRE(pp("1 - 2 + 3") == "1-2+3");
}

TEST_CASE("latex_post_process: braces, parens, brackets", "[formula_tokenizer]") {
    REQUIRE(pp("\\frac { a } { b }") == "\\frac{a}{b}");
    REQUIRE(pp("\\sqrt { x + 1 }") == "\\sqrt{x+1}");
    REQUIRE(pp("( a + b )") == "(a+b)");
    REQUIRE(pp("[ x , y ]") == "[x,y]");
    REQUIRE(pp("{ x }") == "{x}");
    REQUIRE(pp("\\frac{a}{b}") == "\\frac{a}{b}");  // already-tight stays tight
}

TEST_CASE("latex_post_process: backslash-space stays", "[formula_tokenizer]") {
    REQUIRE(pp("a\\ b") == "a\\ b");
    REQUIRE(pp("x +\\ y") == "x+\\ y");
    REQUIRE(pp("\\ ") == "\\ ");  // bare backslash-space stays
}

TEST_CASE("latex_post_process: superscript and subscript", "[formula_tokenizer]") {
    REQUIRE(pp("x ^ 2") == "x^2");
    REQUIRE(pp("x _ i") == "x_i");
    REQUIRE(pp("a ^ { 2 }") == "a^{2}");
    REQUIRE(pp("a _ { i + 1 }") == "a_{i+1}");
}

TEST_CASE("latex_post_process: wrapper-macro stash strips internal whitespace",
          "[formula_tokenizer]") {
    REQUIRE(pp("\\mathrm {x}") == "\\mathrm{x}");
    REQUIRE(pp("\\mathrm {a b}") == "\\mathrm{ab}");
    REQUIRE(pp("\\operatorname {sin} x") == "\\operatorname{sin}x");
    REQUIRE(pp("\\operatorname* {lim}") == "\\operatorname*{lim}");
    REQUIRE(pp("\\mathbf {v} = 0") == "\\mathbf{v}=0");
    REQUIRE(pp("\\text {hi} + 1") == "\\text{hi}+1");
}

TEST_CASE("latex_post_process: CJK text wrapper unwrapped", "[formula_tokenizer]") {
    REQUIRE(pp("\\text{中文} a") == "中文a");
    REQUIRE(pp("\\text {中文}+1") == "中文+1");
    REQUIRE(pp("\\text{中文}") == "中文");
}

TEST_CASE("latex_post_process: literal double-quotes are dropped",
          "[formula_tokenizer]") {
    // remove_chinese_text_wrapping unconditionally strips '"' after CJK fixup.
    REQUIRE(pp("\"a\"") == "a");
    REQUIRE(pp("a \"b\" c") == "a b c");  // letter-ws-letter is not collapsed
}

TEST_CASE("latex_post_process: idempotent on clean input", "[formula_tokenizer]") {
    REQUIRE(pp("a+b") == "a+b");
    REQUIRE(pp("\\sin(x)") == "\\sin(x)");
    REQUIRE(pp("") == "");
    REQUIRE(pp("a") == "a");
}

TEST_CASE("latex_post_process: nested fraction collapses", "[formula_tokenizer]") {
    REQUIRE(pp("\\frac { \\frac { 1 } { 2 } } { 3 }") == "\\frac{\\frac{1}{2}}{3}");
}

TEST_CASE("FormulaTokenizer::load loads real tokenizer.json when present",
          "[formula_tokenizer][.integration]") {
    // Skipped by default tag — the real tokenizer.json lives in the Rust
    // workspace, not in this repo. Run with `-Rinteg` to opt in.
    const std::string path =
        "models/formula/ppformulanet_s/tokenizer.json";
    auto tok = FormulaTokenizer::load(path);
    if (!tok) {
        WARN("tokenizer.json not found at " << path << " — skipping");
        return;
    }
    REQUIRE(tok->vocab_size() >= 50000);

    // eos_id() exposes the learned EOS so callers stop hard-coding the literal.
    REQUIRE(tok->eos_id() == 2);

    // Decoding only the BOS+EOS frame returns the empty string.
    std::vector<int64_t> ids = {0, 2};  // <s>, </s>
    REQUIRE(tok->decode(ids) == "");

    // Garbage past EOS is trimmed off.
    ids = {0, 2, 999, 1000};
    REQUIRE(tok->decode(ids) == "");
}
