#ifndef __VESYLA_TM_BINARY_TREE_HPP__
#define __VESYLA_TM_BINARY_TREE_HPP__

#include "util/Common.hpp"

namespace vesyla {
namespace tm {

class BinaryTree;

// define a CALLBACK function for traversing the tree
typedef void (*CALLBACK)(void *data, BinaryTree *left, BinaryTree *right);

class BinaryTree {
public:
  BinaryTree *left;
  BinaryTree *right;
  void *data;

public:
  BinaryTree(void *data_, BinaryTree *left_, BinaryTree *right_);
  ~BinaryTree();
  void traverse_LRC(std::function<void(BinaryTree *node)> f_);
  void traverse_LCR(std::function<void(BinaryTree *node)> f_);
  void traverse_CLR(std::function<void(BinaryTree *node)> f_);
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_BINARY_TREE_HPP__