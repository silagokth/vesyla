#ifndef __VESYLA_SUPPORT_AGU_MERGE_HPP__
#define __VESYLA_SUPPORT_AGU_MERGE_HPP__

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <vector>

namespace vesyla::util {

// How rops on one port nest into one AGU program, from their loop chains (one
// per rop in program order, outermost loop first; null for a loop without an
// id). Node 0 is the root; every other node is one loop, holding the rops and
// loops nested in it in program order. A loop held by one rop only is that
// rop's own repetition; a loop held by several repeats them as a group (a
// range level of the AGU, around all of them an outer level).
struct LoopTree {
  struct Child {
    bool is_rop;
    size_t index; // rop index, or node index
  };
  struct Node {
    const void *loop = nullptr;
    size_t depth = 0; // loops from the root, the root being 0
    std::vector<Child> children;
    size_t rops = 0;  // rops in the subtree
    size_t first = 0; // first rop of the subtree
  };
  std::vector<Node> nodes;
};

// Nothing when the rops cannot form one AGU program: a loop that would sit in
// two places of the tree, or rops of one loop not contiguous in program order.
inline std::optional<LoopTree>
loop_tree(llvm::ArrayRef<llvm::SmallVector<const void *>> chains) {
  LoopTree tree;
  tree.nodes.emplace_back();
  std::map<const void *, size_t> node_of;
  for (size_t k = 0; k < chains.size(); ++k) {
    size_t node = 0;
    for (const void *loop : chains[k]) {
      size_t next = 0;
      for (const LoopTree::Child &c : tree.nodes[node].children) {
        if (!c.is_rop && loop && tree.nodes[c.index].loop == loop) {
          next = c.index;
        }
      }
      if (!next) {
        if (loop && node_of.count(loop)) {
          return std::nullopt;
        }
        next = tree.nodes.size();
        LoopTree::Node n;
        n.loop = loop;
        n.depth = tree.nodes[node].depth + 1;
        tree.nodes.push_back(n);
        tree.nodes[node].children.push_back({false, next});
        if (loop) {
          node_of[loop] = next;
        }
      }
      node = next;
    }
    tree.nodes[node].children.push_back({true, k});
  }
  // Rops in depth-first order must be the program order.
  size_t expect = 0;
  bool ordered = true;
  auto visit = [&](auto &self, size_t node) -> void {
    LoopTree::Node &n = tree.nodes[node];
    n.first = expect;
    for (const LoopTree::Child &c : n.children) {
      if (c.is_rop) {
        ordered = ordered && c.index == expect;
        ++expect;
      } else {
        self(self, c.index);
      }
    }
    n.rops = expect - n.first;
  };
  visit(visit, 0);
  if (!ordered) {
    return std::nullopt;
  }
  return tree;
}

} // namespace vesyla::util

#endif // __VESYLA_SUPPORT_AGU_MERGE_HPP__
