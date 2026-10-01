#ifndef __VESYLA_TM_ANCHOR_HPP__
#define __VESYLA_TM_ANCHOR_HPP__

#include "vesyla/Support/Anchor.hpp"
#include "vesyla/Support/Common.hpp"
#include <string>

using namespace std;

namespace vesyla {
namespace tm {

struct Anchor {
  // Flat MiniZinc variable identifier (see ::vesyla::Anchor::flat_name).
  string name;
  // The parsed anchor: owning op name, MT event id, and OR/IR indices.
  ::vesyla::Anchor anchor;
  // MiniZinc expression computing this anchor's absolute cycle.
  string timing_expr;

  Anchor() {}
  Anchor(string expr_str_);
  Anchor(::vesyla::Anchor anchor_);
  ~Anchor();
  string to_string();
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_ANCHOR_HPP__
