#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <set>
#include <tuple>

#include "InterconnectPass.hpp"
#include "RoutingDepGraph.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_INTERCONNECTPASS
#include "pasm/Passes.hpp.inc"

namespace {

struct RangeRef {
  mlir::FlatSymbolRefAttr instr_id;
  std::string event;
  std::vector<uint32_t> lo;
  std::vector<uint32_t> hi;
  // True iff every constraint traversed to reach this frame had min_delay == 0.
  bool min_delay_zero;
};

bool ranges_intersect(llvm::ArrayRef<uint32_t> a_lo,
                      llvm::ArrayRef<uint32_t> a_hi,
                      llvm::ArrayRef<uint32_t> b_lo,
                      llvm::ArrayRef<uint32_t> b_hi) {
  if (a_lo.size() != a_hi.size() || a_lo.size() != b_lo.size() ||
      b_lo.size() != b_hi.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a_lo.size(); ++i) {
    if (std::max(a_lo[i], b_lo[i]) > std::min(a_hi[i], b_hi[i])) {
      return false;
    }
  }
  return true;
}

bool range_strictly_after(llvm::ArrayRef<uint32_t> a_lo,
                          llvm::ArrayRef<uint32_t> b_hi) {
  if (a_lo.size() != b_hi.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a_lo.size(); ++i) {
    if (a_lo[i] <= b_hi[i]) {
      return false;
    }
  }
  return true;
}

void populate_routes(RoutingDepGraph &graph, EpochOp epoch,
                     llvm::StringRef kind) {
  auto log_node = [](int id, NodeKind kind, const Anchor &a) {
    llvm::errs() << "node " << id << " "
                 << (kind == NodeKind::First ? "first" : "last") << ": ";
    if (a.instr_id) {
      llvm::errs() << a.instr_id.getValue();
    } else {
      llvm::errs() << "<sentinel>";
    }
    llvm::errs() << " [" << a.event << ", [";
    for (std::size_t i = 0; i < a.indices.size(); ++i) {
      if (i > 0) {
        llvm::errs() << ",";
      }
      llvm::errs() << a.indices[i];
    }
    llvm::errs() << "], delay=" << a.delay << "]\n";
  };

  // insert start and end node first
  Anchor sentinel{};
  graph.insert_node(sentinel, 0, NodeKind::First);
  graph.insert_node(sentinel, 0, NodeKind::Last);

  int current_id = 1;

  // create two nodes for each datadependency one for the first use and one for
  // the last use
  for (mlir::Operation &op : epoch.getBody().front()) {
    auto icdep = mlir::dyn_cast<IcDepOp>(op);
    if (!icdep) {
      continue;
    }
    if (icdep.getKind() != kind) {
      continue;
    }

    llvm::ArrayRef<int32_t> first_idx = icdep.getFirstIdx();
    std::vector<uint32_t> first_idx_v(first_idx.begin(), first_idx.end());
    Anchor first_anchor{icdep.getFirstInstrAttr(), icdep.getFirstEvent().str(),
                        std::move(first_idx_v),
                        static_cast<int32_t>(icdep.getFirstDelay())};

    llvm::ArrayRef<int32_t> last_idx = icdep.getLastIdx();
    std::vector<uint32_t> last_idx_v(last_idx.begin(), last_idx.end());
    Anchor last_anchor{icdep.getLastInstrAttr(), icdep.getLastEvent().str(),
                       std::move(last_idx_v),
                       static_cast<int32_t>(icdep.getLastDelay())};

    graph.insert_node(first_anchor, current_id, NodeKind::First);
    graph.insert_node(last_anchor, current_id, NodeKind::Last);
    ++current_id;
  }
  // DFS each non-sentinel node to discover edges to other nodes.
  for (const Node &node : graph) {
    if (!node.anchor.instr_id) {
      continue;
    }
    log_node(node.id, node.kind, node.anchor);

    // DFS stack of range frames yet to expand for this node.
    std::vector<RangeRef> stack;
    stack.push_back(RangeRef{node.anchor.instr_id, node.anchor.event,
                             node.anchor.indices, node.anchor.indices,
                             /*min_delay_zero=*/true});

    // Cycle guard: delay-[0,0] reverse traversal can otherwise loop forever.
    std::set<std::tuple<std::string, std::string, std::vector<uint32_t>,
                        std::vector<uint32_t>>>
        visited;

    while (!stack.empty()) {
      RangeRef current = std::move(stack.back());
      stack.pop_back();

      auto visit_key = std::make_tuple(current.instr_id.getValue().str(),
                                       current.event, current.lo, current.hi);
      if (!visited.insert(visit_key).second) {
        continue;
      }

      // Follow constraints onward to reach further nodes.
      for (mlir::Operation &op : epoch.getBody().front()) {
        auto cstr = mlir::dyn_cast<CstrOp>(op);
        if (!cstr) {
          continue;
        }
        // delay==[0,0] also enables reverse direction (dst -> src).
        auto delay = cstr.getDelay();
        bool delay_zero =
            delay.getMin().value_or(1) == 0 && delay.getMax().value_or(1) == 0;
        int num_dirs = delay_zero ? 2 : 1;
        for (int dir = 0; dir < num_dirs; ++dir) {
          auto src_ar = (dir == 0) ? cstr.getSrc() : cstr.getDst();
          auto dst_ar = (dir == 0) ? cstr.getDst() : cstr.getSrc();
          if (src_ar.getInstr() != current.instr_id) {
            continue;
          }
          std::string src_event = src_ar.getEvent().str();
          if (src_event != current.event) {
            continue;
          }

          // Event-less constraints carry no indices; just propagate dst.
          if (src_event.empty()) {
            llvm::ArrayRef<uint32_t> dst_lo = dst_ar.getIdxLo();
            llvm::ArrayRef<uint32_t> dst_hi = dst_ar.getIdxHi();
            bool new_flag = cstr.getDelay().getMin().value_or(0) == 0;
            stack.push_back(RangeRef{
                dst_ar.getInstr(), dst_ar.getEvent().str(),
                std::vector<uint32_t>(dst_lo.begin(), dst_lo.end()),
                std::vector<uint32_t>(dst_hi.begin(), dst_hi.end()), new_flag});
            continue;
          }

          // src matches if its range overlaps current or starts strictly after
          // it.
          llvm::ArrayRef<uint32_t> src_lo = src_ar.getIdxLo();
          llvm::ArrayRef<uint32_t> src_hi = src_ar.getIdxHi();
          bool intersects =
              ranges_intersect(current.lo, current.hi, src_lo, src_hi);
          bool src_after = range_strictly_after(src_lo, current.hi);
          if (!intersects && !src_after) {
            continue;
          }

          llvm::ArrayRef<uint32_t> dst_lo = dst_ar.getIdxLo();
          llvm::ArrayRef<uint32_t> dst_hi = dst_ar.getIdxHi();

          // Trace: matched src -> dst hop with full index ranges.
          llvm::errs() << "\t" << src_ar.getInstr().getValue() << " ["
                       << src_ar.getEvent() << ", [";
          for (std::size_t i = 0; i < src_lo.size(); ++i) {
            if (i > 0) {
              llvm::errs() << ",";
            }
            llvm::errs() << src_lo[i] << ":" << src_hi[i];
          }
          llvm::errs() << "]] -> " << dst_ar.getInstr().getValue() << " ["
                       << dst_ar.getEvent() << ", [";
          for (std::size_t i = 0; i < dst_lo.size(); ++i) {
            if (i > 0) {
              llvm::errs() << ",";
            }
            llvm::errs() << dst_lo[i] << ":" << dst_hi[i];
          }
          llvm::errs() << "]]\n";

          // src is a single element: no delta to apply, propagate dst as-is.
          bool src_single = true;
          for (std::size_t i = 0; i < src_lo.size(); ++i) {
            if (src_lo[i] != src_hi[i]) {
              src_single = false;
              break;
            }
          }
          if (src_single) {
            bool new_flag = current.min_delay_zero &&
                            cstr.getDelay().getMin().value_or(0) == 0;
            stack.push_back(RangeRef{
                dst_ar.getInstr(), dst_ar.getEvent().str(),
                std::vector<uint32_t>(dst_lo.begin(), dst_lo.end()),
                std::vector<uint32_t>(dst_hi.begin(), dst_hi.end()), new_flag});
            continue;
          }

          // delta = how far inside src the current frame's lower bound sits.
          std::vector<uint32_t> new_lo(src_lo.size());
          std::vector<uint32_t> delta(src_lo.size());
          bool delta_nonzero = false;
          for (std::size_t i = 0; i < src_lo.size(); ++i) {
            new_lo[i] = std::max(current.lo[i], src_lo[i]);
            delta[i] = new_lo[i] - src_lo[i];
            if (delta[i] != 0) {
              delta_nonzero = true;
            }
          }

          // Apply delta to dst's lower bound, only for as many dims as delta
          // has.
          std::vector<uint32_t> dst_new_lo(dst_lo.begin(), dst_lo.end());
          for (std::size_t i = 0; i < delta.size() && i < dst_new_lo.size();
               ++i) {
            dst_new_lo[i] += delta[i];
          }

          // Bilaterality survives only if delta is zero AND this hop's min
          // delay is zero.
          bool new_flag = current.min_delay_zero && !delta_nonzero &&
                          cstr.getDelay().getMin().value_or(0) == 0;

          stack.push_back(RangeRef{
              dst_ar.getInstr(), dst_ar.getEvent().str(), std::move(dst_new_lo),
              std::vector<uint32_t>(dst_hi.begin(), dst_hi.end()), new_flag});
        }
      }

      // Direct match: another graph node lies at or past current's lower
      // bound in the same (instr, event) space.
      for (const Node &candidate : graph) {
        if (candidate.key() == node.key()) {
          continue;
        }
        if (candidate.anchor.instr_id != current.instr_id) {
          continue;
        }
        if (candidate.anchor.event != current.event) {
          continue;
        }
        llvm::ArrayRef<uint32_t> point = candidate.anchor.indices;
        if (point.size() != current.lo.size()) {
          continue;
        }
        bool ge_lo = true;
        for (std::size_t i = 0; i < point.size(); ++i) {
          if (point[i] < current.lo[i]) {
            ge_lo = false;
            break;
          }
        }
        if (!ge_lo) {
          continue;
        }
        graph.insert_edge(node.key(), candidate.key());
      }
    }
  }

  // Wire orphan nodes (no incoming/outgoing edges) to the start/end sentinels.
  NodeKey start_key{0, NodeKind::First};
  NodeKey end_key{0, NodeKind::Last};
  for (const Node &n : graph) {
    if (n.id == 0) {
      continue;
    }
    if (!graph.has_incoming(n)) {
      graph.insert_edge(start_key, n.key());
    }
    if (!graph.has_outgoing(n)) {
      graph.insert_edge(n.key(), end_key);
    }
  }
}

//===----------------------------------------------------------------------===//
class InterconnectPassRewriter : public OpRewritePattern<EpochOp> {
public:
  using OpRewritePattern<EpochOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EpochOp op,
                                PatternRewriter &rewriter) const final {
    auto build_and_dump = [&](llvm::StringRef kind, llvm::StringRef prefix) {
      llvm::errs() << "\n========================================\n"
                   << "=== " << kind << " graph for epoch " << op.getId()
                   << " ===\n"
                   << "========================================\n";
      RoutingDepGraph graph;
      populate_routes(graph, op, kind);
      // graph.transitive_reduce();

      std::string dot_path = (prefix + "_" + op.getId().str() + ".dot").str();
      std::string png_path = (prefix + "_" + op.getId().str() + ".png").str();
      graph.dump_dot(dot_path);

      // Best-effort PNG rendering via graphviz. Any failure is reported but
      // does not abort the pass — the .dot file is always available.
      std::string cmd = "dot -Tpng " + dot_path + " -o " + png_path;
      int rc = std::system(cmd.c_str());
      if (rc != 0) {
        llvm::errs() << "graphviz rendering failed (rc=" << rc << "): " << cmd
                     << "\n";
      }
    };

    build_and_dump("bulk", "routes");
    build_and_dump("word", "swb");

    return failure();
  }
};

class InterconnectPass : public impl::InterconnectPassBase<InterconnectPass> {
public:
  using impl::InterconnectPassBase<InterconnectPass>::InterconnectPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<InterconnectPassRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
