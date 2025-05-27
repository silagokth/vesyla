#ifndef __VESYLA_TM_CONSTRAINT_HPP__
#define __VESYLA_TM_CONSTRAINT_HPP__

#include "util/Common.hpp"
#include <regex>
#include <string>
#include <vector>

using namespace std;

namespace vesyla {
namespace tm {

struct Constraint {
  string kind;
  string expr;

  Constraint() {}
  Constraint(string kind_, string expr_) : kind(kind_), expr(expr_) {
    // remove all white spaces
    expr.erase(remove_if(expr.begin(), expr.end(), ::isspace), expr.end());
  }
  Constraint(string expr_);
  ~Constraint() {}
  string to_string();
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_CONSTRAINT_HPP__