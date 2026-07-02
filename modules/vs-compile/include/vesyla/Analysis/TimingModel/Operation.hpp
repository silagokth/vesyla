#ifndef __VESYLA_TM_OPERATION_HPP__
#define __VESYLA_TM_OPERATION_HPP__

#include "vesyla/Support/Anchor.hpp"
#include "vesyla/Support/Common.hpp"
#include <regex>
#include <string>
#include <vector>

using namespace std;

namespace vesyla {
namespace tm {

struct OperationExpr {
  enum Kind { UNDEFINED, EVENT, REPEAT, TRANSIT };
  Kind kind;
  unordered_map<string, string> parameters;
  std::vector<OperationExpr> children;
  OperationExpr() {}
  OperationExpr(string str);
  ~OperationExpr();
  string to_string();
  // Enumerate every event anchor of this expression. MT holds the event id;
  // OR / IR hold the enclosing repeat iteration indices split by whether the
  // repeat sits above (outer) or below (inner) the nearest transition. The
  // returned anchors have an empty name (the owning operation fills it in).
  std::vector<::vesyla::Anchor> get_all_anchors();
};

struct Operation {
  string name;
  OperationExpr expr;
  string duration_expr;

  int col = -1;
  int row = -1;
  int slot = -1;
  int port = -1;

  Operation() {}
  Operation(string name_, string expr_) : name(name_), expr(expr_) {}
  Operation(string name_, OperationExpr expr_) : name(name_), expr(expr_) {}
  Operation(string expr_);
  ~Operation() {}
  string to_string();
  std::vector<std::string> get_all_anchors();
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_OPERATION_HPP__
