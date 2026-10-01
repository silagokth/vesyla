#ifndef __VESYLA_TM_CONSTRAINT_HPP__
#define __VESYLA_TM_CONSTRAINT_HPP__

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Support/Anchor.hpp"
#include "vesyla/Support/Common.hpp"
#include <optional>
#include <regex>
#include <string>
#include <vector>

using namespace std;

namespace vesyla {
namespace tm {

struct Constraint {
  using Anchor = ::vesyla::Anchor;

  std::string src_id;
  std::string dst_id;
  std::optional<Anchor> src_anchor;
  std::optional<Anchor> dst_anchor;
  std::optional<int> min_delay;
  std::optional<int> max_delay;
  bool is_neq = false;

  string kind;
  // Raw expression text used by the legacy string-based constructors
  // (Constraint(string), Constraint(string,string)). For structured
  // Constraints (the new constructor / from_cstr_op) this stays empty and
  // to_string() builds the expression from the structured fields.
  std::string expr;

  Constraint() {}
  Constraint(string kind_, string expr_) : kind(kind_) {
    expr_.erase(remove_if(expr_.begin(), expr_.end(), ::isspace), expr_.end());
    expr = std::move(expr_);
  }
  Constraint(string expr_);

  Constraint(std::string src_id_, std::string dst_id_,
             std::optional<int> min_delay_, std::optional<int> max_delay_,
             std::optional<Anchor> src_anchor_ = std::nullopt,
             std::optional<Anchor> dst_anchor_ = std::nullopt)
      : src_id(std::move(src_id_)), dst_id(std::move(dst_id_)),
        src_anchor(std::move(src_anchor_)), dst_anchor(std::move(dst_anchor_)),
        min_delay(std::move(min_delay_)), max_delay(std::move(max_delay_)) {}

  ~Constraint() {}
  // Bare expression text (e.g. "read_b_seq.e0[0][29] == read_a_seq.e0[0][29]"),
  // built from the structured fields when set, otherwise the stored legacy
  // string. No "constraint" prefix or trailing semicolon — callers add those.
  string to_string();

  // Expand a pasm.cstr op's index range into one Constraint per
  // (src_idx, dst_idx) atom. Each returned Constraint has src_id/dst_id,
  // min/max_delay, is_neq, and (optional) src_anchor/dst_anchor populated.
  static std::vector<Constraint> from_cstr_op(vesyla::pasm::CstrOp cstr_op);
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_CONSTRAINT_HPP__
