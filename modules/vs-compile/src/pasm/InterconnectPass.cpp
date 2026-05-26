#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <set>
#include <tuple>

#include "InterconnectBinding.hpp"
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

// Maps `idx` (a coord inside [src_lo, src_hi]) to the same-position coord
// inside [dst_lo, dst_hi] using row-major flat-index correspondence. Caller
// must guarantee equal element count between the two ranges.
std::vector<uint32_t> map_index(llvm::ArrayRef<uint32_t> idx,
                                llvm::ArrayRef<uint32_t> src_lo,
                                llvm::ArrayRef<uint32_t> src_hi,
                                llvm::ArrayRef<uint32_t> dst_lo,
                                llvm::ArrayRef<uint32_t> dst_hi) {
  uint64_t flat = 0;
  uint64_t stride = 1;
  for (int i = static_cast<int>(src_lo.size()) - 1; i >= 0; --i) {
    flat += static_cast<uint64_t>(idx[i] - src_lo[i]) * stride;
    stride *= static_cast<uint64_t>(src_hi[i] - src_lo[i] + 1);
  }
  std::vector<uint32_t> out(dst_lo.size());
  stride = 1;
  for (int i = static_cast<int>(dst_lo.size()) - 1; i >= 0; --i) {
    uint64_t dim = static_cast<uint64_t>(dst_hi[i] - dst_lo[i] + 1);
    out[i] = dst_lo[i] + static_cast<uint32_t>((flat / stride) % dim);
    stride *= dim;
  }
  return out;
}

void populate_routes(RoutingDepGraph &graph, mlir::Block &icdep_block,
                     mlir::Block &cstr_block, llvm::StringRef kind) {
  int current_id = 1;

  // create two nodes for each datadependency one for the first use and one for
  // the last use
  for (mlir::Operation &op : icdep_block) {
    auto icdep = mlir::dyn_cast<IcDepOp>(op);
    if (!icdep) {
      continue;
    }
    if (icdep.getKind() != kind) {
      continue;
    }

    AnchorAttr first = icdep.getFirst();
    llvm::ArrayRef<int32_t> first_idx = first.getIdx();
    std::vector<uint32_t> first_idx_v(first_idx.begin(), first_idx.end());
    Anchor first_anchor{first.getInstr(), first.getEvent().str(),
                        std::move(first_idx_v), first.getDelay()};

    AnchorAttr last = icdep.getLast();
    llvm::ArrayRef<int32_t> last_idx = last.getIdx();
    std::vector<uint32_t> last_idx_v(last_idx.begin(), last_idx.end());
    Anchor last_anchor{last.getInstr(), last.getEvent().str(),
                       std::move(last_idx_v), last.getDelay()};

    llvm::StringRef dir = icdep.getDir().value_or(llvm::StringRef());
    graph.insert_node(first_anchor, current_id, NodeKind::First, dir,
                      icdep.getSrc(), icdep.getDst());
    graph.insert_node(last_anchor, current_id, NodeKind::Last, dir,
                      icdep.getSrc(), icdep.getDst());
    ++current_id;
  }

  // DFS each non-sentinel node to discover edges to other nodes.
  for (const Node &node : graph) {
    if (!node.anchor.instr_id) {
      continue;
    }

    // DFS stack of range frames yet to expand for this node.
    std::vector<RangeRef> stack;
    stack.push_back(RangeRef{node.anchor.instr_id, node.anchor.event,
                             node.anchor.indices, node.anchor.indices});

    // Cycle guard: delay-[0,0] reverse traversal can otherwise loop forever.
    std::set<std::tuple<std::string, std::string, std::vector<uint32_t>,
                        std::vector<uint32_t>>>
        visited;

    // look for all possible paths to check which other nodes are reachable here
    while (!stack.empty()) {
      RangeRef current = std::move(stack.back());
      stack.pop_back();

      auto visit_key = std::make_tuple(current.instr_id.getValue().str(),
                                       current.event, current.lo, current.hi);
      if (!visited.insert(visit_key).second) {
        continue;
      }

      // check all constraints for possible matches
      for (mlir::Operation &op : cstr_block) {
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

          // constraints without indices should just be propagated
          if (src_event.empty()) {
            llvm::ArrayRef<uint32_t> dst_lo = dst_ar.getIdxLo();
            llvm::ArrayRef<uint32_t> dst_hi = dst_ar.getIdxHi();
            stack.push_back(
                RangeRef{dst_ar.getInstr(), dst_ar.getEvent().str(),
                         std::vector<uint32_t>(dst_lo.begin(), dst_lo.end()),
                         std::vector<uint32_t>(dst_hi.begin(), dst_hi.end())});
            continue;
          }

          // Only reject the cstr when src lies strictly in current's past on
          // every propagated dim — clamp + map_index handles every other case.
          // Surplus src dims (beyond dst's arity) are filter dims and aren't
          // gated here.
          llvm::ArrayRef<uint32_t> src_lo = src_ar.getIdxLo();
          llvm::ArrayRef<uint32_t> src_hi = src_ar.getIdxHi();
          llvm::ArrayRef<uint32_t> dst_lo_ar = dst_ar.getIdxLo();
          llvm::ArrayRef<uint32_t> current_lo_ar(current.lo);
          std::size_t n = std::min(src_lo.size(), dst_lo_ar.size());
          if (range_strictly_after(current_lo_ar.take_front(n),
                                   src_hi.take_front(n))) {
            continue;
          }

          llvm::ArrayRef<uint32_t> dst_lo = dst_ar.getIdxLo();
          llvm::ArrayRef<uint32_t> dst_hi = dst_ar.getIdxHi();

          // Clamp current.lo into src's rectangle, then map that src coord to
          // the corresponding dst coord by row-major flat-index correspondence.
          std::vector<uint32_t> matched_src(src_lo.size());
          for (std::size_t i = 0; i < src_lo.size(); ++i) {
            uint32_t cl = (i < current.lo.size()) ? current.lo[i] : src_lo[i];
            matched_src[i] = std::min(std::max(cl, src_lo[i]), src_hi[i]);
          }
          std::vector<uint32_t> dst_new_lo =
              map_index(matched_src, src_lo, src_hi, dst_lo, dst_hi);

          stack.push_back(RangeRef{
              dst_ar.getInstr(), dst_ar.getEvent().str(), std::move(dst_new_lo),
              std::vector<uint32_t>(dst_hi.begin(), dst_hi.end())});
        }
      }

      // check if there is a node in that range if yes create an edege between
      // them
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
    mlir::Block &epoch_block = op.getBody().front();

    bool any_processed = false;
    for (CellOp cell : epoch_block.getOps<CellOp>()) {
      if (cell->hasAttr("interconnect_done")) {
        continue;
      }
      any_processed = true;
      std::string cell_label = op.getId().str() + "_r" +
                               std::to_string(cell.getRow()) + "c" +
                               std::to_string(cell.getCol());

      auto build_and_dump = [&](llvm::StringRef kind, llvm::StringRef prefix) {
        llvm::errs() << "\n========================================\n"
                     << "=== " << kind << " graph for cell " << cell_label
                     << " ===\n"
                     << "========================================\n";
        RoutingDepGraph graph;
        populate_routes(graph, cell.getBody().front(), epoch_block, kind);

        std::string dot_path = (prefix + "_" + cell_label + ".dot").str();
        std::string png_path = (prefix + "_" + cell_label + ".png").str();
        RoutingDepGraph reduced = graph;
        reduced.transitive_reduce();
        reduced.dump_dot(dot_path);

        InterconnectBinding binding = bind_interconnect(graph, kind);
        llvm::errs() << "binding (" << kind << "):\n";
        dump_binding(binding, llvm::errs());

        RopOp rop = nullptr;
        if (kind == "word") {
          rop = emit_swb_instructions(binding, cell, rewriter);
        } else {
          rop = emit_route_instructions(binding, cell, rewriter);
        }
        if (rop) {
          emit_sequence_instructions(binding, rop, rewriter);
          emit_interconnect_constraints(binding, rop, rewriter);
        }

        // Best-effort PNG rendering via graphviz. Any failure is reported but
        // does not abort the pass — the .dot file is always available.
        std::string cmd = "dot -Tpng " + dot_path + " -o " + png_path;
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
          llvm::errs() << "graphviz rendering failed (rc=" << rc << "): " << cmd
                       << "\n";
        }
      };

      build_and_dump("bulk", "bulk");
      build_and_dump("word", "swb");
      cell->setAttr("interconnect_done", rewriter.getUnitAttr());
    }

    return any_processed ? success() : failure();
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
    module.dump();
  }
};

} // namespace
} // namespace vesyla::pasm
