#include "vesyla/Parser/BisonUtil.hpp"

#include <algorithm>
#include <regex>

namespace vesyla {
namespace schedule {
mlir::ModuleOp *module;
mlir::Operation *temp_epoch_op;

/*! This function reports the error according to the current file, source line,
 * and the given message. */
void print_error(const char *message) {
  LOG_DEBUG << "Error in \"" << global_input_file_name << "\" "
            << "around line (" << global_source_line << "): " << message;
}

void print_grammar(const std::string &grammar_) {
  LOG_DEBUG << "BISON: "
            << "source line: " + std::to_string(global_source_line) + ", "
            << "grammar: " << grammar_;
}

} // namespace schedule
} // namespace vesyla

mlir::Operation *build_cstr(rop_ref_t *lhs, rop_ref_t *rhs,
                            const std::string &cmp) {
  auto epoch_op =
      llvm::dyn_cast<vesyla::pasm::EpochOp>(vesyla::schedule::temp_epoch_op);
  if (!epoch_op) {
    vesyla::schedule::print_error("EpochOp not found");
    exit(1);
  }

  mlir::OpBuilder builder(epoch_op.getBody());
  auto loc = builder.getUnknownLoc();

  int alpha = lhs->offset;
  int beta = rhs->offset;

  ::vesyla::AnchorRange src, dst;
  std::optional<int> min_delay;
  std::optional<int> max_delay;

  if (cmp == "<") {
    src = lhs->range;
    dst = rhs->range;
    min_delay = 1 + alpha - beta;
    // no upper bound
  } else if (cmp == ">") {
    src = rhs->range;
    dst = lhs->range;
    min_delay = 1 + beta - alpha;
    // no upper bound
  } else { // "==" or "!="
    src = lhs->range;
    dst = rhs->range;
    min_delay = alpha - beta;
    max_delay = alpha - beta;
  }
  bool is_neq = (cmp == "!=");

  // Same-sign post-rule: if both bounds are present and negative, swap.
  if (min_delay && max_delay && *min_delay < 0 && *max_delay < 0) {
    std::swap(src, dst);
    int new_min = -*max_delay;
    int new_max = -*min_delay;
    min_delay = new_min;
    max_delay = new_max;
  }

  std::optional<int32_t> min_v;
  std::optional<int32_t> max_v;
  if (min_delay) {
    min_v = static_cast<int32_t>(*min_delay);
  }
  if (max_delay) {
    max_v = static_cast<int32_t>(*max_delay);
  }
  auto delay_attr =
      vesyla::pasm::DelayAttr::get(builder.getContext(), min_v, max_v);

  auto to_u = [](const std::vector<int> &v) {
    return std::vector<uint32_t>(v.begin(), v.end());
  };
  auto make_ar = [&](const ::vesyla::AnchorRange &r) {
    return vesyla::pasm::AnchorRangeAttr::get(
        builder.getContext(),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), r.lo.name),
        to_u(r.lo.or_idx), static_cast<uint32_t>(r.lo.mt_idx), to_u(r.lo.ir_idx),
        to_u(r.hi.or_idx), static_cast<uint32_t>(r.hi.mt_idx),
        to_u(r.hi.ir_idx));
  };

  auto cstr_op = vesyla::pasm::CstrOp::create(builder, loc, make_ar(src),
                                              make_ar(dst), delay_attr,
                                              builder.getBoolAttr(is_neq));

  return cstr_op.getOperation();
}

static rop_ref_t *parse_rop_ref(const std::string &s) {
  // Anchor text (name + OR.MT.IR with an optional innermost-IR range) followed
  // by an optional +N / -N cycle offset. The OR/MT/IR grammar and the
  // "range only on the innermost IR" rule are enforced by AnchorRange::parse.
  static const std::regex re(
      R"(^\s*([A-Za-z_]\w*(?:\.|\[[0-9:]*\])*)\s*([+-]\s*\d+)?\s*$)");
  LOG_DEBUG << "parse_rop_ref input: [" << s << "]";
  std::smatch m;
  if (!std::regex_match(s, m, re)) {
    vesyla::schedule::print_error(
        ("cstr: invalid rop reference: " + s).c_str());
    exit(1);
  }

  auto *ref = new rop_ref_t();
  auto range = ::vesyla::AnchorRange::parse(m[1].str());
  if (!range) {
    vesyla::schedule::print_error(
        ("cstr: invalid anchor: " + m[1].str()).c_str());
    exit(1);
  }
  ref->range = *range;
  ref->offset = 0;

  if (m[2].matched) {
    std::string off_str = m[2];
    off_str.erase(std::remove_if(off_str.begin(), off_str.end(), ::isspace),
                  off_str.end());
    ref->offset = std::stoi(off_str);
  }

  return ref;
}

std::vector<mlir::Operation *> parse_and_build_cstr(const std::string &expr) {
  std::string s = expr;
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
  }

  size_t pos;
  std::string cmp;
  size_t cmp_len;
  if ((pos = s.find("!=")) != std::string::npos) {
    cmp = "!=";
    cmp_len = 2;
  } else if ((pos = s.find("==")) != std::string::npos) {
    cmp = "==";
    cmp_len = 2;
  } else if ((pos = s.find('<')) != std::string::npos) {
    cmp = "<";
    cmp_len = 1;
  } else if ((pos = s.find('>')) != std::string::npos) {
    cmp = ">";
    cmp_len = 1;
  } else {
    vesyla::schedule::print_error("cstr: no comparison operator found");
    exit(1);
  }

  std::string lhs_str = s.substr(0, pos);
  std::string rhs_str = s.substr(pos + cmp_len);

  rop_ref_t *lhs = parse_rop_ref(lhs_str);
  rop_ref_t *rhs = parse_rop_ref(rhs_str);

  if ((cmp == "==" || cmp == "!=") && lhs->offset < rhs->offset) {
    std::swap(lhs, rhs);
  }

  std::vector<mlir::Operation *> ops;
  ops.push_back(build_cstr(lhs, rhs, cmp));

  delete lhs;
  delete rhs;

  return ops;
}
