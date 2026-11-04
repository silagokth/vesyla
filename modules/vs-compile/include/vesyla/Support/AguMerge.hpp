#ifndef __VESYLA_SUPPORT_AGU_MERGE_HPP__
#define __VESYLA_SUPPORT_AGU_MERGE_HPP__

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <set>

namespace vesyla::util {

// Whether rops on one port can form one AGU program, from their loop chains
// (one per rop, outermost loop first; null for a loop without an id). The
// outer loops all of them share become the outer repetitions around the
// merged events; returns how many those are. Nothing when a loop is shared by
// some rops but not all, which would need a merge inside the merge.
inline std::optional<size_t>
shared_loops(llvm::ArrayRef<llvm::SmallVector<const void *>> chains) {
  size_t shared = 0;
  while (!chains.empty()) {
    const auto &head = chains.front();
    bool all = shared < head.size() && head[shared];
    for (const auto &chain : chains) {
      if (!all || chain.size() <= shared || chain[shared] != head[shared]) {
        all = false;
        break;
      }
    }
    if (!all) {
      break;
    }
    ++shared;
  }
  std::set<const void *> seen;
  for (const auto &chain : chains) {
    for (size_t i = shared; i < chain.size(); ++i) {
      if (chain[i] && !seen.insert(chain[i]).second) {
        return std::nullopt;
      }
    }
  }
  return shared;
}

} // namespace vesyla::util

#endif // __VESYLA_SUPPORT_AGU_MERGE_HPP__
