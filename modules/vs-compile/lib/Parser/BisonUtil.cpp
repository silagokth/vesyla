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

namespace {
struct RepInfo {
  mlir::Attribute iter;
  mlir::Attribute delay;
};

std::vector<RepInfo> get_rop_reps(vesyla::pasm::EpochOp epoch_op,
                                  llvm::StringRef rop_id) {
  std::vector<RepInfo> reps;
  if (epoch_op.getBody().empty())
    return reps;
  for (auto &op : epoch_op.getBody().front().getOperations()) {
    auto rop = llvm::dyn_cast<vesyla::pasm::RopOp>(&op);
    if (!rop || rop.getSymName() != rop_id)
      continue;
    if (rop.getBody().empty())
      break;
    for (auto &child : rop.getBody().front().getOperations()) {
      auto instr = llvm::dyn_cast<vesyla::pasm::InstrOp>(&child);
      if (!instr || instr.getType() != "rep")
        continue;
      auto param = instr.getParam();
      reps.push_back({param.get("iter"), param.get("delay")});
    }
    break;
  }
  return reps;
}

} // namespace

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
  std::vector<idx_entry_t> src_indices;
  std::string dst_id, dst_event;
  std::vector<idx_entry_t> dst_indices;
  std::optional<int> min_delay;
  std::optional<int> max_delay;

  if (cmp == "<") {
    src_id = lhs->id;
    src_event = lhs->event;
    src_indices = lhs->indices;
    dst_id = rhs->id;
    dst_event = rhs->event;
    dst_indices = rhs->indices;
    min_delay = 1 + alpha - beta;
    // no upper bound
  } else if (cmp == ">") {
    src_id = rhs->id;
    src_event = rhs->event;
    src_indices = rhs->indices;
    dst_id = lhs->id;
    dst_event = lhs->event;
    dst_indices = lhs->indices;
    min_delay = 1 + beta - alpha;
    // no upper bound
  } else { // "==" or "!="
    src_id = lhs->id;
    src_event = lhs->event;
    src_indices = lhs->indices;
    dst_id = rhs->id;
    dst_event = rhs->event;
    dst_indices = rhs->indices;
    min_delay = alpha - beta;
    max_delay = alpha - beta;
  }
  bool is_neq = (cmp == "!=");

  // Same-sign post-rule: if both bounds are present and negative, swap.
  if (min_delay && max_delay && *min_delay < 0 && *max_delay < 0) {
    std::swap(src_id, dst_id);
    std::swap(src_event, dst_event);
    std::swap(src_indices, dst_indices);
    int new_min = -*max_delay;
    int new_max = -*min_delay;
    min_delay = new_min;
    max_delay = new_max;
  }

  // If event is missing and the rop has rep instructions, default to e0
  // pinned at iteration 0 (one zero index per rep level).
  auto src_reps = get_rop_reps(epoch_op, src_id);
  if (src_event.empty() && !src_reps.empty()) {
    src_event = "e0";
    if (src_indices.empty()) {
      src_indices.assign(src_reps.size(), idx_entry_t{false, false, 0, 0});
    }
  }
  auto dst_reps = get_rop_reps(epoch_op, dst_id);
  if (dst_event.empty() && !dst_reps.empty()) {
    dst_event = "e0";
    if (dst_indices.empty()) {
      dst_indices.assign(dst_reps.size(), idx_entry_t{false, false, 0, 0});
    }
  }

  auto lower_indices = [&](const std::vector<idx_entry_t> &entries,
                           const std::vector<RepInfo> &reps,
                           std::vector<int32_t> &out_lo,
                           std::vector<int32_t> &out_hi) {
    if (entries.size() > reps.size()) {
      vesyla::schedule::print_error(
          ("cstr: more indices (" + std::to_string(entries.size()) +
           ") than rep levels (" + std::to_string(reps.size()) + ")")
              .c_str());
      exit(1);
    }
    // entries[k] pairs with reps[k] (instruction order): the first rep
    // written is index [0], next is [1], ...
    for (size_t k = 0; k < entries.size(); ++k) {
      const auto &e = entries[k];
      int32_t lo = e.lo_default ? 0 : e.lo;
      int32_t hi = e.hi;
      if (e.hi_default) {
        if (auto iter_int =
                llvm::dyn_cast_or_null<mlir::IntegerAttr>(reps[k].iter)) {
          hi = static_cast<int32_t>(iter_int.getInt()) - 1;
        } else {
          hi = lo;
        }
      }
      out_lo.push_back(lo);
      out_hi.push_back(hi);
    }
  };

  std::vector<int32_t> src_idx_lo, src_idx_hi, dst_idx_lo, dst_idx_hi;
  lower_indices(src_indices, src_reps, src_idx_lo, src_idx_hi);
  lower_indices(dst_indices, dst_reps, dst_idx_lo, dst_idx_hi);

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

  std::vector<uint32_t> src_lo_u(src_idx_lo.begin(), src_idx_lo.end());
  std::vector<uint32_t> src_hi_u(src_idx_hi.begin(), src_idx_hi.end());
  std::vector<uint32_t> dst_lo_u(dst_idx_lo.begin(), dst_idx_lo.end());
  std::vector<uint32_t> dst_hi_u(dst_idx_hi.begin(), dst_idx_hi.end());

  auto src_ar = vesyla::pasm::AnchorRangeAttr::get(
      builder.getContext(),
      mlir::FlatSymbolRefAttr::get(builder.getContext(), src_id), src_event,
      src_lo_u, src_hi_u);
  auto dst_ar = vesyla::pasm::AnchorRangeAttr::get(
      builder.getContext(),
      mlir::FlatSymbolRefAttr::get(builder.getContext(), dst_id), dst_event,
      dst_lo_u, dst_hi_u);

  auto cstr_op = vesyla::pasm::CstrOp::create(builder, loc, src_ar, dst_ar,
                                              delay_attr,
                                              builder.getBoolAttr(is_neq));

  return cstr_op.getOperation();
}

static int32_t parse_nonneg_int(const std::string &s, const std::string &ctx) {
  static const std::regex int_re(R"(^\d+$)");
  if (!std::regex_match(s, int_re)) {
    vesyla::schedule::print_error(
        ("cstr: index must be a non-negative integer in " + ctx + ": " + s)
            .c_str());
    exit(1);
  }
  return static_cast<int32_t>(std::stoi(s));
}

static idx_entry_t parse_idx_entry(const std::string &raw) {
  std::string content = raw;
  content.erase(std::remove_if(content.begin(), content.end(), ::isspace),
                content.end());
  idx_entry_t e{false, false, 0, 0};
  auto colon = content.find(':');
  if (colon == std::string::npos) {
    e.lo = e.hi = parse_nonneg_int(content, "[" + raw + "]");
    return e;
  }
  std::string left = content.substr(0, colon);
  std::string right = content.substr(colon + 1);
  if (left.empty()) {
    e.lo_default = true;
  } else {
    e.lo = parse_nonneg_int(left, "[" + raw + "]");
  }
  if (right.empty()) {
    e.hi_default = true;
  } else {
    e.hi = parse_nonneg_int(right, "[" + raw + "]");
  }
  return e;
}

static rop_ref_t *parse_rop_ref(const std::string &s) {
  static const std::regex re(
      R"(^\s*(\w+)(?:\.(\w+))?((?:\s*\[[^\]]*\])*)\s*([+-]\s*\d+)?\s*$)");
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
  static const std::regex idx_re(R"(\[([^\]]*)\])");
  for (auto it = std::sregex_iterator(idx_str.begin(), idx_str.end(), idx_re);
       it != std::sregex_iterator(); ++it) {
    ref->indices.push_back(parse_idx_entry((*it)[1].str()));
  }

  for (size_t k = 0; k + 1 < ref->indices.size(); ++k) {
    const auto &e = ref->indices[k];
    if (e.lo_default || e.hi_default || e.lo != e.hi) {
      vesyla::schedule::print_error(
          ("cstr: range allowed only on the innermost index in: " + s).c_str());
      exit(1);
    }
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

  if ((cmp == "==" || cmp == "!=") && lhs->offset < rhs->offset) {
    std::swap(lhs, rhs);
  }

  std::vector<mlir::Operation *> ops;
  ops.push_back(build_cstr(lhs, rhs, cmp));

  delete lhs;
  delete rhs;

  return ops;
}
