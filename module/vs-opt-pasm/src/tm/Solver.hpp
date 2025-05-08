#ifndef __VESYLA_TM_SOLVER_HPP__
#define __VESYLA_TM_SOLVER_HPP__

#include "TimingModel.hpp"

namespace vesyla {
namespace tm {
class Solver {
public:
  Solver() {}
  ~Solver() {}
  unordered_map<string, string> solve(TimingModel &tm);
};
} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_SOLVER_HPP__