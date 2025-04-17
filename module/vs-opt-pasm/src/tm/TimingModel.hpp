#ifndef __VESYLA_TM_TIMINGMODEL_HPP__
#define __VESYLA_TM_TIMINGMODEL_HPP__

#include "tm/Anchor.hpp"
#include "tm/Constraint.hpp"
#include "tm/Operation.hpp"
#include <string>

using namespace std;

namespace vesyla {
namespace tm {
class TimingModel {
public:
  TimingModel();
  ~TimingModel();
};
} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_TIMINGMODEL_HPP__