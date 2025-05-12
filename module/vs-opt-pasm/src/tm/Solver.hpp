#ifndef __VESYLA_TM_SOLVER_HPP__
#define __VESYLA_TM_SOLVER_HPP__

#include "TimingModel.hpp"

namespace vesyla {
namespace tm {
class Solver {
private:
  std::string tmp_path;

public:
  Solver() { tmp_path = "."; }
  Solver(std::string tmp_path_) : tmp_path(tmp_path_) {}
  ~Solver() {}
  std::unordered_map<std::string, std::string> solve(TimingModel &tm);
};
} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_SOLVER_HPP__