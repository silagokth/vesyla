#include "BinaryTree.hpp"

namespace vesyla {
namespace tm {
BinaryTree::BinaryTree(void *data_ = nullptr, BinaryTree *left_ = nullptr,
                       BinaryTree *right_ = nullptr) {
  data = data_;
  left = left_;
  right = right_;
}
BinaryTree::~BinaryTree() {
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
void BinaryTree::traverse_LRC(std::function<void(BinaryTree *)> f_) {
  if (left != nullptr) {
    left->traverse_LRC(f_);
  }
  if (right != nullptr) {
    right->traverse_LRC(f_);
  }
  f_(this);
}
void BinaryTree::traverse_LCR(std::function<void(BinaryTree *)> f_) {

  if (left != nullptr) {
    left->traverse_LCR(f_);
  }
  f_(this);
  if (right != nullptr) {
    right->traverse_LCR(f_);
  }
}
void BinaryTree::traverse_CLR(std::function<void(BinaryTree *)> f_) {
  f_(this);
  if (left != nullptr) {
    left->traverse_CLR(f_);
  }
  if (right != nullptr) {
    right->traverse_CLR(f_);
  }
}
} // namespace tm
} // namespace vesyla