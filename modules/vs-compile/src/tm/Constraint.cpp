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
    exprs.push_back(match[2]);
  } else {
    LOG_FATAL << "Invalid constraint string: " << expr_str;
    std::exit(EXIT_FAILURE);
  }
}
string Constraint::to_string() {
  string s;
  for (auto &e : exprs)
    s += "constraint " + kind + " " + e + ";\n";
  return s;
}

Constraint::Constraint(vesyla::pasm::CstrOp cstr_op) {
  auto fmt_ref = [](llvm::StringRef rop_id, llvm::StringRef event,
                    llvm::ArrayRef<int32_t> indices) {
    std::string s = event.empty() ? rop_id.str() : (rop_id + "." + event).str();
    for (int32_t i : indices) {
      s += "[" + std::to_string(i) + "]";
    }
    return s;
  };

  llvm::ArrayRef<int32_t> src_lo = cstr_op.getSrcIdxLo();
  llvm::ArrayRef<int32_t> src_hi = cstr_op.getSrcIdxHi();
  llvm::ArrayRef<int32_t> dst_lo = cstr_op.getDstIdxLo();
  llvm::ArrayRef<int32_t> dst_hi = cstr_op.getDstIdxHi();
  int32_t min_delay = cstr_op.getMinDelay();
  int32_t max_delay = cstr_op.getMaxDelay();
  bool is_neq = cstr_op.getIsNeq();

  kind = "linear";

  auto build_atom = [&](llvm::ArrayRef<int32_t> src_idx,
                        llvm::ArrayRef<int32_t> dst_idx) -> std::string {
    std::string src_ref =
        fmt_ref(cstr_op.getSrc(), cstr_op.getSrcEvent(), src_idx);
    std::string dst_ref =
        fmt_ref(cstr_op.getDst(), cstr_op.getDstEvent(), dst_idx);
    if (is_neq) {
      if (min_delay == 0) {
        return dst_ref + " != " + src_ref;
      }
      return dst_ref + " != " + src_ref + " + " + std::to_string(min_delay);
    }
    if (min_delay == max_delay) {
      if (min_delay == 0) {
        return dst_ref + " == " + src_ref;
      }
      if (min_delay > 0) {
        return dst_ref + " == " + src_ref + " + " + std::to_string(min_delay);
      }
      return dst_ref + " == " + src_ref + " - " + std::to_string(-min_delay);
    }
    std::string s =
        dst_ref + " - " + src_ref + " >= " + std::to_string(min_delay);
    // 10000000 matches MAX_LATENCY in tm/TimingModel.cpp; sentinel = no upper
    // bound.
    if (max_delay != 10000000) {
      s += " /\\ " + dst_ref + " - " + src_ref +
           " <= " + std::to_string(max_delay);
    }
    return s;
  };

  auto advance = [](std::vector<int32_t> &cur, llvm::ArrayRef<int32_t> lo,
                    llvm::ArrayRef<int32_t> hi) {
    for (size_t i = cur.size(); i-- > 0;) {
      if (cur[i] < hi[i]) {
        cur[i]++;
        return true;
      }
      cur[i] = lo[i];
    }
    return false;
  };

  std::vector<int32_t> src_cur(src_lo.begin(), src_lo.end());
  std::vector<int32_t> dst_cur(dst_lo.begin(), dst_lo.end());
  exprs.push_back(build_atom(src_cur, dst_cur));
  while (advance(src_cur, src_lo, src_hi)) {
    advance(dst_cur, dst_lo, dst_hi);
    exprs.push_back(build_atom(src_cur, dst_cur));
  }

  for (auto &e : exprs) {
    e.erase(remove_if(e.begin(), e.end(), ::isspace), e.end());
  }
}

} // namespace tm
} // namespace vesyla
