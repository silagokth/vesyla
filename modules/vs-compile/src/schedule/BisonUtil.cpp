#include "BisonUtil.hpp"

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

  std::string src_id, src_event;
  std::vector<int32_t> src_indices;
  std::string dst_id, dst_event;
  std::vector<int32_t> dst_indices;
  int min_delay = 0;
  int max_delay = 0;

  if (cmp == "<") {
    src_id = lhs->id;
    src_event = lhs->event;
    src_indices = lhs->indices;
    dst_id = rhs->id;
    dst_event = rhs->event;
    dst_indices = rhs->indices;
    min_delay = 1 + alpha - beta;
    max_delay = 10000000;
  } else if (cmp == ">") {
    src_id = rhs->id;
    src_event = rhs->event;
    src_indices = rhs->indices;
    dst_id = lhs->id;
    dst_event = lhs->event;
    dst_indices = lhs->indices;
    min_delay = 1 + beta - alpha;
    max_delay = 10000000;
  } else { // "=="
    src_id = lhs->id;
    src_event = lhs->event;
    src_indices = lhs->indices;
    dst_id = rhs->id;
    dst_event = rhs->event;
    dst_indices = rhs->indices;
    min_delay = alpha - beta;
    max_delay = alpha - beta;
  }

  // Same-sign post-rule: if both bounds are negative, swap direction.
  if (min_delay < 0 && max_delay < 0) {
    std::swap(src_id, dst_id);
    std::swap(src_event, dst_event);
    std::swap(src_indices, dst_indices);
    int new_min = -max_delay;
    int new_max = -min_delay;
    min_delay = new_min;
    max_delay = new_max;
  }

  auto cstr_op = builder.create<vesyla::pasm::CstrOp>(
      loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), src_id),
      builder.getStringAttr(src_event),
      builder.getDenseI32ArrayAttr(src_indices),
      builder.getDenseI32ArrayAttr(src_indices),
      mlir::FlatSymbolRefAttr::get(builder.getContext(), dst_id),
      builder.getStringAttr(dst_event),
      builder.getDenseI32ArrayAttr(dst_indices),
      builder.getDenseI32ArrayAttr(dst_indices),
      builder.getI32IntegerAttr(min_delay),
      builder.getI32IntegerAttr(max_delay));

  return cstr_op.getOperation();
}

static rop_ref_t *parse_rop_ref(const std::string &s) {
  static const std::regex re(
      R"(^\s*(\w+)(?:\.(\w+))?((?:\s*\[\s*[+-]?\d+\s*\])*)\s*([+-]\s*\d+)?\s*$)");
  LOG_DEBUG << "parse_rop_ref input: [" << s << "]";
  std::smatch m;
  if (!std::regex_match(s, m, re)) {
    vesyla::schedule::print_error(
        ("cstr: invalid rop reference: " + s).c_str());
    exit(1);
  }

  auto *ref = new rop_ref_t();
  ref->id = m[1];
  ref->event = m[2].matched ? m[2].str() : std::string();
  ref->offset = 0;

  std::string idx_str = m[3];
  static const std::regex idx_re(R"(\[\s*([+-]?\d+)\s*\])");
  for (auto it = std::sregex_iterator(idx_str.begin(), idx_str.end(), idx_re);
       it != std::sregex_iterator(); ++it) {
    ref->indices.push_back(std::stoi((*it)[1]));
  }

  if (m[4].matched) {
    std::string off_str = m[4];
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

  std::vector<mlir::Operation *> ops;
  if (cmp == "!=") {
    // lhs != rhs is encoded as two "<" edges:
    //   lhs < rhs  (lhs precedes rhs by at least 1 cycle)
    //   rhs < lhs  (rhs precedes lhs by at least 1 cycle)
    ops.push_back(build_cstr(lhs, rhs, "<"));
    ops.push_back(build_cstr(rhs, lhs, "<"));
  } else {
    ops.push_back(build_cstr(lhs, rhs, cmp));
  }

  delete lhs;
  delete rhs;

  return ops;
}
