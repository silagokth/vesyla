#include "vesyla/Analysis/TimingModel/Constraint.hpp"

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

  // Flat MZN-friendly form (see ::vesyla::Anchor::flat_name). Matches the names
  // tm::Anchor produces in extractAnchors so to_mzn can emit directly.
  auto flat_ref = [](const std::string &id,
                     const std::optional<Anchor> &anchor) -> std::string {
    if (!anchor)
      return id;
    return anchor->flat_name(id);
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
  std::string src_id = src_ar.getInstr().getValue().str();
  std::string dst_id = dst_ar.getInstr().getValue().str();
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

  // A point anchor for endpoint `a` with the given IR indices. A fully bare
  // endpoint (no OR, MT 0, no IR) references the operation's start directly
  // rather than a specific event, so it yields no anchor (nullopt) — matching
  // the pre-OR/MT/IR "no event" behavior.
  auto point = [](vesyla::pasm::AnchorRangeAttr a,
                  const std::vector<uint32_t> &ir) -> std::optional<Anchor> {
    if (a.getOrLo().empty() && a.getMtLo() == 0 && a.getIrLo().empty()) {
      return std::nullopt;
    }
    Anchor anc;
    anc.or_idx.assign(a.getOrLo().begin(), a.getOrLo().end());
    anc.mt_idx = static_cast<int>(a.getMtLo());
    anc.ir_idx.assign(ir.begin(), ir.end());
    return anc;
  };
  // Odometer over an IR range: advance the innermost dimension first.
  auto advance = [](std::vector<uint32_t> &cur, llvm::ArrayRef<uint32_t> lo,
                    llvm::ArrayRef<uint32_t> hi) {
    for (std::size_t i = cur.size(); i-- > 0;) {
      if (cur[i] < hi[i]) {
        cur[i]++;
        return true;
      }
      cur[i] = lo[i];
    }
    return false;
  };

  // The range spans the IR cross-product; src and dst element counts match
  // (checked by CstrOp::verify), so they step in lockstep.
  std::vector<uint32_t> src_cur(src_ar.getIrLo().begin(),
                                src_ar.getIrLo().end());
  std::vector<uint32_t> dst_cur(dst_ar.getIrLo().begin(),
                                dst_ar.getIrLo().end());
  auto emit = [&]() {
    Constraint c(src_id, dst_id, min_delay, max_delay,
                 point(src_ar, src_cur), point(dst_ar, dst_cur));
    c.is_neq = is_neq;
    c.kind = "linear";
    result.push_back(std::move(c));
  };

  // How many indices each dimension of an end holds, and how many points that
  // is in total.
  auto extents = [](llvm::ArrayRef<uint32_t> lo, llvm::ArrayRef<uint32_t> hi) {
    std::vector<uint64_t> ext(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) {
      ext[i] = static_cast<uint64_t>(hi[i]) - lo[i] + 1;
    }
    return ext;
  };
  auto volume = [](const std::vector<uint64_t> &ext) {
    uint64_t n = 1;
    for (uint64_t e : ext) {
      n *= e;
    }
    return n;
  };
  std::vector<uint64_t> src_ext =
      extents(src_ar.getIrLo(), src_ar.getIrHi());
  std::vector<uint64_t> dst_ext =
      extents(dst_ar.getIrLo(), dst_ar.getIrHi());

  // Which points of the range the pair actually has to be stated at.
  //
  // The two ends step in lockstep, so the distance between them is affine in
  // the position along the range: each dimension adds its index times a stride,
  // and that stride is whatever the dimension's repetition delay works out to,
  // the same on every pass. Writing the pair out at every point is therefore
  // redundant. An exact distance is pinned by the origin and one step along
  // each dimension -- the origin fixes the constant, each step fixes one
  // stride. A bounded distance needs the corners instead: an affine distance
  // over a box reaches its extremes at a vertex, and the sign of each stride is
  // the solver's to choose, so the far end of every dimension has to be named
  // too. Neither set is an approximation -- what it leaves out is implied by
  // what it states -- and both are a handful of points where the full range is
  // the whole product, which is the difference between a model the solver reads
  // in milliseconds and one it cannot hold in memory.
  //
  // A disequality states nothing affine: it excludes single points rather than
  // a half-space, and every one of them has to be excluded by name. Ends whose
  // volumes disagree are left alone too, the correspondence between them being
  // whatever the lockstep walk makes it.
  bool exact = min_delay && max_delay && *min_delay == *max_delay;
  uint64_t count = volume(src_ext);
  if (is_neq || count != volume(dst_ext)) {
    emit();
    while (advance(src_cur, src_ar.getIrLo(), src_ar.getIrHi())) {
      advance(dst_cur, dst_ar.getIrLo(), dst_ar.getIrHi());
      emit();
    }
    return result;
  }

  std::vector<std::vector<uint64_t>> samples;
  samples.push_back(std::vector<uint64_t>(src_ext.size(), 0));
  for (std::size_t i = 0; i < src_ext.size(); ++i) {
    if (src_ext[i] < 2) {
      continue;
    }
    if (exact) {
      std::vector<uint64_t> at(src_ext.size(), 0);
      at[i] = 1;
      samples.push_back(at);
      continue;
    }
    std::size_t stated = samples.size();
    for (std::size_t k = 0; k < stated; ++k) {
      std::vector<uint64_t> at = samples[k];
      at[i] = src_ext[i] - 1;
      samples.push_back(at);
    }
  }

  // A sample is a point of the src range; the point it pairs with is the one
  // the lockstep walk would have reached at the same position, so it is found
  // by flattening in src's shape and unflattening in dst's.
  auto flatten = [](const std::vector<uint64_t> &ext,
                    const std::vector<uint64_t> &at) {
    uint64_t flat = 0;
    for (std::size_t i = 0; i < ext.size(); ++i) {
      flat = flat * ext[i] + at[i];
    }
    return flat;
  };
  auto unflatten = [](const std::vector<uint64_t> &ext, uint64_t flat,
                      llvm::ArrayRef<uint32_t> lo) {
    std::vector<uint32_t> at(ext.size());
    for (std::size_t i = ext.size(); i-- > 0;) {
      at[i] = static_cast<uint32_t>(lo[i] + flat % ext[i]);
      flat /= ext[i];
    }
    return at;
  };
  for (const std::vector<uint64_t> &at : samples) {
    uint64_t flat = flatten(src_ext, at);
    src_cur = unflatten(src_ext, flat, src_ar.getIrLo());
    dst_cur = unflatten(dst_ext, flat, dst_ar.getIrLo());
    emit();
  }
  return result;
}

} // namespace tm
} // namespace vesyla
