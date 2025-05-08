#ifndef __VESYLA_TM_OPERATION_HPP__
#define __VESYLA_TM_OPERATION_HPP__

#include "util/Common.hpp"
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
};

struct Operation {
  string name;
  OperationExpr expr;

  Operation() {}
  Operation(string name_, string expr_) : name(name_), expr(expr_) {}
  Operation(string name_, OperationExpr expr_) : name(name_), expr(expr_) {}
  ~Operation() {}
  string to_string();
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_OPERATION_HPP__