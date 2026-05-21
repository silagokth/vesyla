#include "Constraint.hpp"

namespace vesyla {
namespace tm {
Constraint::Constraint(string expr_str) {
  // it has to start with "constraint"
  // followed by the kind of the constraint
  // followed by the expression
  // e.g. constraint linear op0.e0 > op1.e0

  // remove the leading and trailing spaces
  const char *WhiteSpace = " \t\v\r\n";
  std::size_t start = expr_str.find_first_not_of(WhiteSpace);
  std::size_t end = expr_str.find_last_not_of(WhiteSpace);
  expr_str =
      start == end ? std::string() : expr_str.substr(start, end - start + 1);

  string pattern = "^constraint\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s+(.*)$";
  std::smatch match;
  if (std::regex_match(expr_str, match, std::regex(pattern))) {
    kind = match[1];
    expr = match[2];
  } else {
    LOG_FATAL << "Invalid constraint string: " << expr_str;
    std::exit(EXIT_FAILURE);
  }
}

string Constraint::to_string() {
  // Legacy path: a raw string was supplied at construction (e.g., cop != rop
  // anchor constraints). Return it as-is.
  if (!expr.empty())
    return expr;

  // Flat MZN-friendly form: "read_a_seq_e0_0_29". Matches the names
  // tm::Anchor produces in extractAnchors so to_mzn can emit directly.
  auto flat_ref = [](const std::string &id,
                     const std::optional<Anchor> &anchor) -> std::string {
    if (!anchor)
      return id;
    std::string s = id + "_" + anchor->event_id;
    for (int i : anchor->idx)
      s += "_" + std::to_string(i);
    return s;
  };
  std::string sr = flat_ref(src_id, src_anchor);
  std::string dr = flat_ref(dst_id, dst_anchor);

  if (is_neq) {
    int m = min_delay.value_or(0);
    if (m == 0)
      return dr + " != " + sr;
    return dr + " != " + sr + " + " + std::to_string(m);
  }
  if (min_delay && max_delay && *min_delay == *max_delay) {
    int m = *min_delay;
    if (m == 0)
      return dr + " == " + sr;
    if (m > 0)
      return dr + " == " + sr + " + " + std::to_string(m);
    return dr + " == " + sr + " - " + std::to_string(-m);
  }
  std::string s;
  if (min_delay)
    s = dr + " - " + sr + " >= " + std::to_string(*min_delay);
  if (max_delay) {
    if (!s.empty())
      s += " /\\ ";
    s += dr + " - " + sr + " <= " + std::to_string(*max_delay);
  }
  return s;
}

std::vector<Constraint> Constraint::from_cstr_op(vesyla::pasm::CstrOp cstr_op) {
  std::vector<Constraint> result;

  auto src_ar = cstr_op.getSrc();
  auto dst_ar = cstr_op.getDst();
  llvm::ArrayRef<uint32_t> src_lo = src_ar.getIdxLo();
  llvm::ArrayRef<uint32_t> src_hi = src_ar.getIdxHi();
  llvm::ArrayRef<uint32_t> dst_lo = dst_ar.getIdxLo();
  llvm::ArrayRef<uint32_t> dst_hi = dst_ar.getIdxHi();
  std::string src_id = src_ar.getInstr().getValue().str();
  std::string dst_id = dst_ar.getInstr().getValue().str();
  std::string src_event = src_ar.getEvent().str();
  std::string dst_event = dst_ar.getEvent().str();
  std::optional<int> min_delay;
  std::optional<int> max_delay;
  auto d = cstr_op.getDelay();
  if (auto m = d.getMin()) {
    min_delay = *m;
  }
  if (auto m = d.getMax()) {
    max_delay = *m;
  }
  bool is_neq = cstr_op.getIsNeq();

  auto make_anchor =
      [](const std::string &event,
         const std::vector<uint32_t> &idx) -> std::optional<Anchor> {
    if (event.empty()) {
      return std::nullopt;
    }
    Anchor a;
    a.event_id = event;
    a.idx.assign(idx.begin(), idx.end());
    return a;
  };
  auto advance = [](std::vector<uint32_t> &cur, llvm::ArrayRef<uint32_t> lo,
                    llvm::ArrayRef<uint32_t> hi) {
    for (size_t i = cur.size(); i-- > 0;) {
      if (cur[i] < hi[i]) {
        cur[i]++;
        return true;
      }
      cur[i] = lo[i];
    }
    return false;
  };

  std::vector<uint32_t> src_cur(src_lo.begin(), src_lo.end());
  std::vector<uint32_t> dst_cur(dst_lo.begin(), dst_lo.end());
  auto emit = [&]() {
    Constraint c(src_id, dst_id, min_delay, max_delay,
                 make_anchor(src_event, src_cur),
                 make_anchor(dst_event, dst_cur));
    c.is_neq = is_neq;
    c.kind = "linear";
    result.push_back(std::move(c));
  };
  emit();
  while (advance(src_cur, src_lo, src_hi)) {
    advance(dst_cur, dst_lo, dst_hi);
    emit();
  }
  return result;
}

} // namespace tm
} // namespace vesyla
