#ifndef __VESYLA_TM_BINARY_TREE_HPP__
#define __VESYLA_TM_BINARY_TREE_HPP__

#include "util/Common.hpp"

namespace vesyla {
namespace tm {

template <typename T> class BinaryTree {
public:
  BinaryTree *left;
  BinaryTree *right;
  T *data;

public:
  BinaryTree() : left(nullptr), right(nullptr), data(nullptr) {}
  BinaryTree(T *data_, BinaryTree *left_, BinaryTree *right_)
      : left(left_), right(right_), data(data_) {}
  ~BinaryTree() {
    if (left != nullptr) {
      delete left;
    }
    if (right != nullptr) {
      delete right;
    }
    if (data != nullptr) {
      delete data;
    }
    data = nullptr;
    left = nullptr;
    right = nullptr;
  }
  void traverse_LRC(std::function<void(BinaryTree<T> *node)> f_) {
    if (left != nullptr) {
      left->traverse_LRC(f_);
    }
    if (right != nullptr) {
      right->traverse_LRC(f_);
    }
    f_(this);
  }
  void traverse_LCR(std::function<void(BinaryTree<T> *node)> f_) {

    if (left != nullptr) {
      left->traverse_LCR(f_);
    }
    f_(this);
    if (right != nullptr) {
      right->traverse_LCR(f_);
    }
  }
  void traverse_CLR(std::function<void(BinaryTree<T> *node)> f_) {
    f_(this);
    if (left != nullptr) {
      left->traverse_CLR(f_);
    }
    if (right != nullptr) {
      right->traverse_CLR(f_);
    }
  }
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_BINARY_TREE_HPP__