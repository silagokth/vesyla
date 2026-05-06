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
string Constraint::to_string() { return "constraint " + kind + " " + expr; }

Constraint::Constraint(vesyla::pasm::CstrOp cstr_op) {
  auto fmt_ref = [](llvm::StringRef rop_id, llvm::StringRef event,
                    llvm::ArrayRef<int32_t> indices) {
    std::string s = event.empty() ? rop_id.str() : (rop_id + "." + event).str();
    for (int32_t i : indices) {
      s += "[" + std::to_string(i) + "]";
    }
    return s;
  };

  std::string src_ref =
      fmt_ref(cstr_op.getSrc(), cstr_op.getSrcEvent(), cstr_op.getSrcIdxLo());
  std::string dst_ref =
      fmt_ref(cstr_op.getDst(), cstr_op.getDstEvent(), cstr_op.getDstIdxLo());

  int32_t min_delay = cstr_op.getMinDelay();
  int32_t max_delay = cstr_op.getMaxDelay();

  kind = "linear";

  if (min_delay == max_delay) {
    if (min_delay == 0) {
      expr = dst_ref + " == " + src_ref;
    } else if (min_delay > 0) {
      expr = dst_ref + " == " + src_ref + " + " + std::to_string(min_delay);
    } else {
      expr = dst_ref + " == " + src_ref + " - " + std::to_string(-min_delay);
    }
  } else {
    expr = dst_ref + " - " + src_ref + " >= " + std::to_string(min_delay);
    // 10000000 matches MAX_LATENCY in tm/TimingModel.cpp; sentinel = no upper
    // bound.
    if (max_delay != 10000000) {
      expr += " /\\ " + dst_ref + " - " + src_ref +
              " <= " + std::to_string(max_delay);
    }
  }

  expr.erase(remove_if(expr.begin(), expr.end(), ::isspace), expr.end());
}

} // namespace tm
} // namespace vesyla
