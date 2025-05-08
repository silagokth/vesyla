#ifndef __VESYLA_TM_ANCHOR_HPP__
#define __VESYLA_TM_ANCHOR_HPP__

#include "util/Common.hpp"
#include <regex>
#include <string>
#include <vector>

using namespace std;

namespace vesyla {
namespace tm {

struct AnchorExpr {
  string op_name;
  int event_id;
  std::vector<int> indices;
  AnchorExpr() {}
  AnchorExpr(string str);
  ~AnchorExpr();
  string to_string();
};

struct Anchor {
  string name;
  AnchorExpr expr;

  Anchor() {}
  Anchor(string expr_str_);
  Anchor(AnchorExpr expr_);
  ~Anchor();
  string to_string();
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_ANCHOR_HPP__