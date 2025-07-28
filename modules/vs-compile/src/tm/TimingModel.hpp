#ifndef __VESYLA_TM_TIMINGMODEL_HPP__
#define __VESYLA_TM_TIMINGMODEL_HPP__

#include "Anchor.hpp"
#include "BinaryTree.hpp"
#include "Constraint.hpp"
#include "Operation.hpp"
#include <string>

using namespace std;

namespace vesyla {
namespace tm {

struct BinaryTreeData {
  OperationExpr expr;
  string start;
  string duration;
};

class TimingModel {
public:
  unordered_map<string, Operation> operations;
  unordered_map<string, Anchor> anchors;
  std::vector<Constraint> constraints;
  std::set<string> variables;

  enum state_t { INITIALIZED, COMPILED };
  state_t state = INITIALIZED;

public:
  TimingModel();
  ~TimingModel();

public:
  void compile();
  string to_string();
  int to_mzn(std::ostream &mzn_file, std::ostream &dzn_file);
  string add_operation(Operation op) {
    operations[op.name] = op;
    return op.name;
  }
  void add_constraint(Constraint constraint) {
    constraints.push_back(constraint);
  }
  void from_string(string str);
  Operation get_operation(string name);

private:
  BinaryTree<BinaryTreeData> *build_binary_tree(OperationExpr &expr);
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_TIMINGMODEL_HPP__
