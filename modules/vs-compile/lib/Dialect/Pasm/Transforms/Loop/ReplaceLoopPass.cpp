#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Support/Common.hpp"
#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/RandName.hpp"

#include "LoopLevelDetail.hpp"

#include <cstdlib>
#include <map>
#include <utility>

namespace vesyla::pasm {
#define GEN_PASS_DEF_REPLACELOOPOP
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {
using namespace loop_level_detail;

// Sequencer calc modes (fabric ISA calc verbo_map keys).
constexpr int CALC_MODE_ADD = 1;
constexpr int CALC_MODE_SUB = 2;
constexpr int CALC_MODE_LT = 21;
// operand2 source flag: 0 = static immediate, 1 = dynamic (register index).
constexpr int OPND_STATIC = 0;
constexpr int OPND_DYNAMIC = 1;

// Scalar registers reserved for loop control. Allocation is structural rather
// than liveness-based (see below): the controller register file is small and
// the live ranges of loop-control registers are determined by loop nesting, so
// a fixed reservation is both correct and simple.
//
// Register map assumed here:
//   r0        conventional zero (act-mode-2 prep in RegisterAllocator.cpp
//             relies on it; init() zeroes all registers at reset)
//   r1..r3    spare
//   r4..r11   act-mode-2 port-map prep (two contiguous 4-register blocks)
//   r12       loop flag (shared by all loops, see below)
//   r13..r15  loop counters, indexed by nesting depth
//
// A loop counter is live across its entire body, so nested loops need distinct
// counters; sibling loops never overlap and reuse the same register. The
// counter for a loop at nesting depth d is LOOP_COUNTER_REG_TOP - d, bounded by
// LOOP_COUNTER_REG_FLOOR. The flag register is written by the epilogue's bound
// test and consumed immediately by the branch, so its live range is a single
// epilogue and never overlaps another flag: one shared register suffices.
//
// Keeping the counters/flag disjoint from r0 and the act-mode-2 window means no
// coordination with RegisterAllocator.cpp is required. A liveness-based
// allocator can replace this later without changing the injection logic.
constexpr int LOOP_FLAG_REG = 12;
constexpr int LOOP_COUNTER_REG_TOP = 15;   // counter for nesting depth 0
constexpr int LOOP_COUNTER_REG_FLOOR = 13; // deepest counter (depth 2)

// The largest back-edge distance the brn offset field can encode. Derived from
// the ISA so a wider target field lifts the limit with no code change.
int brn_max_offset(const nlohmann::json &isa_json) {
  for (const auto &component : isa_json["components"]) {
    for (const auto &instr : component["instructions"]) {
      if (!instr.contains("name") || instr["name"] != "brn") {
        continue;
      }
      for (const auto &segment : instr["segments"]) {
        if (segment.contains("name") && segment["name"] == "target_true") {
          int bw = segment["bitwidth"].get<int>();
          bool is_signed = segment.value("is_signed", false);
          return is_signed ? (1 << (bw - 1)) - 1 : (1 << bw) - 1;
        }
      }
    }
  }
  llvm::outs() << "Error: brn.target_true segment not found in ISA.\n";
  std::exit(EXIT_FAILURE);
}

// Per-cell state gathered over a loop body: where to place the once-only
// prologue (front of the cell's first body slice), where to place the
// per-iteration epilogue (end of the cell's last body slice), and the number
// of instructions the cell contributes (the branch offset spans them).
struct CellLoopInfo {
  RawOp first_raw;
  RawOp last_raw;
  int body_len = 0;
};

mlir::Block *raw_op_block(RawOp raw_op, PatternRewriter &rewriter) {
  mlir::Region &region = raw_op.getBody();
  return region.empty() ? rewriter.createBlock(&region) : &region.front();
}

InstrOp make_instr(PatternRewriter &rewriter, mlir::Location loc,
                   llvm::StringRef type,
                   llvm::ArrayRef<mlir::NamedAttribute> params) {
  return rewriter.create<InstrOp>(
      loc, rewriter.getStringAttr(util::RandName::generate(8)),
      rewriter.getStringAttr(type), rewriter.getDictionaryAttr(params));
}

mlir::NamedAttribute i32_attr(PatternRewriter &rewriter, llvm::StringRef name,
                              int value) {
  return rewriter.getNamedAttr(name, rewriter.getI32IntegerAttr(value));
}

//===----------------------------------------------------------------------===//
class ReplaceLoopOpRewriter : public OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern<LoopOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getBody().empty()) {
      return mlir::failure();
    }
    int iter = op.getIter();
    mlir::Block &body_block = op.getBody().front();

    // Lower leaf loops only; the greedy driver (applyPatternsGreedily) then
    // peels nested loops inside-out; each lowered inner loop becoming plain
    // body instructions of its parent
    for (mlir::Operation &child : body_block) {
      if (llvm::isa<LoopOp>(&child)) {
        return mlir::failure();
      }
    }

    // Allocate the counter register from the loop's nesting depth (annotated by
    // the pass before rewriting). Depth 0 uses the top of the counter pool;
    // each enclosing loop takes the next register down. The flag register is
    // shared across all loops (single-epilogue live range).
    int depth = 0;
    if (auto depth_attr = op->getAttrOfType<mlir::IntegerAttr>("loop_depth")) {
      depth = depth_attr.getInt();
    }
    int counter_reg = LOOP_COUNTER_REG_TOP - depth;
    int flag_reg = LOOP_FLAG_REG;
    if (counter_reg < LOOP_COUNTER_REG_FLOOR) {
      llvm::outs() << "Error: loop nesting depth " << depth
                   << " exceeds the reserved counter registers (r"
                   << LOOP_COUNTER_REG_FLOOR << "..r" << LOOP_COUNTER_REG_TOP
                   << "). A liveness-based register allocator is needed.\n";
      std::exit(EXIT_FAILURE);
    }

    // Aggregate the loop body per cell across all of its epochs, in program
    // order, so the branch offset can be computed from the merged stream.
    std::map<std::pair<int, int>, CellLoopInfo> cell_info;
    llvm::SmallVector<mlir::Operation *> body_epochs;
    for (mlir::Operation &epoch_child : body_block) {
      auto epoch_op = llvm::dyn_cast<EpochOp>(&epoch_child);
      if (!epoch_op) {
        continue; // skip the yield terminator
      }
      body_epochs.push_back(&epoch_child);
      if (epoch_op.getBody().empty()) {
        continue;
      }
      for (mlir::Operation &raw_child : epoch_op.getBody().front()) {
        auto raw_op = llvm::dyn_cast<RawOp>(&raw_child);
        if (!raw_op) {
          continue;
        }
        int n = 0;
        if (!raw_op.getBody().empty()) {
          for (mlir::Operation &instr : raw_op.getBody().front()) {
            if (llvm::isa<InstrOp>(&instr)) {
              ++n;
            }
          }
        }
        CellLoopInfo &info = cell_info[{raw_op.getRow(), raw_op.getCol()}];
        if (!info.first_raw) {
          info.first_raw = raw_op;
        }
        info.last_raw = raw_op;
        info.body_len += n;
      }
    }

    vesyla::pasm::Config cfg;
    nlohmann::json isa_json = cfg.get_isa_json();
    int max_offset = brn_max_offset(isa_json);

    // Inject the control instructions into every participating cell. Each cell
    // has its own sequencer, so each needs its own counter and branch; the
    // identical per-cell cycle cost (1 prologue + 3 epilogue) keeps the cells
    // in lock-step, which ScheduleEpoch's per-cell latency verifier enforced.
    for (auto &entry : cell_info) {
      CellLoopInfo &info = entry.second;

      // brn targets are signed offsets relative to the branch's own PC
      // (fabric_arch.json). Local layout per cell after merging:
      //   [prologue calc]              runs once (before the back-edge target)
      //   [body_len instrs]            <- back-edge target (first body instr)
      //   [calc i = i + 1]
      //   [calc flag = (i < iter)]
      //   [brn]                        <- this instruction
      // So the jump back spans body_len + 2 instructions.
      int loop_start_offset = -(info.body_len + 2);
      int fall_through_offset = 1;
      if (info.body_len + 2 > max_offset) {
        llvm::outs() << "Error: loop body of cell (" << entry.first.first
                     << ", " << entry.first.second << ") has " << info.body_len
                     << " instructions; back-edge offset exceeds the brn "
                        "target range (+/-"
                     << max_offset << "). A branch trampoline is needed.\n";
        std::exit(EXIT_FAILURE);
      }

      mlir::Location loc = info.first_raw.getLoc();

      // Prologue: zero the counter once, at the very front of the cell stream.
      // Self-subtract avoids depending on a hardwired zero register (none
      // exists). The counter counts up 0..iter-1 and rides the act signal.
      mlir::Block *first_block = raw_op_block(info.first_raw, rewriter);
      rewriter.setInsertionPointToStart(first_block);
      make_instr(rewriter, loc, "calc",
                 {i32_attr(rewriter, "mode", CALC_MODE_SUB),
                  i32_attr(rewriter, "operand1", counter_reg),
                  i32_attr(rewriter, "operand2_sd", OPND_DYNAMIC),
                  i32_attr(rewriter, "operand2", counter_reg),
                  i32_attr(rewriter, "result", counter_reg)});

      // Epilogue: increment, test i < iter, branch back while looping. Placed
      // at the end of the cell's last body slice, before its terminator.
      mlir::Block *last_block = raw_op_block(info.last_raw, rewriter);
      rewriter.setInsertionPoint(last_block->getTerminator());
      make_instr(rewriter, loc, "calc",
                 {i32_attr(rewriter, "mode", CALC_MODE_ADD),
                  i32_attr(rewriter, "operand1", counter_reg),
                  i32_attr(rewriter, "operand2_sd", OPND_STATIC),
                  i32_attr(rewriter, "operand2", 1),
                  i32_attr(rewriter, "result", counter_reg)});
      make_instr(rewriter, loc, "calc",
                 {i32_attr(rewriter, "mode", CALC_MODE_LT),
                  i32_attr(rewriter, "operand1", counter_reg),
                  i32_attr(rewriter, "operand2_sd", OPND_STATIC),
                  i32_attr(rewriter, "operand2", iter),
                  i32_attr(rewriter, "result", flag_reg)});
      make_instr(rewriter, loc, "brn",
                 {i32_attr(rewriter, "reg", flag_reg),
                  i32_attr(rewriter, "target_true", loop_start_offset),
                  i32_attr(rewriter, "target_false", fall_through_offset)});
    }

    // Unwrap: hoist the body epochs (now carrying the loop control) into the
    // parent block, in order, then drop the loop. MergeRawOp later flattens
    // them with the surrounding epochs; the branch offsets are relative, so
    // the concatenation is transparent to them.
    for (mlir::Operation *epoch : body_epochs) {
      epoch->moveBefore(op);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ReplaceLoopOp : public impl::ReplaceLoopOpBase<ReplaceLoopOp> {
public:
  using impl::ReplaceLoopOpBase<ReplaceLoopOp>::ReplaceLoopOpBase;
  void runOnOperation() final {
    // Annotate every loop with its nesting depth before rewriting, so counter
    // allocation is stable even as the greedy driver unwraps loops (which moves
    // nested loops into the parent block). The attribute rides with the op.
    mlir::OpBuilder builder(&getContext());
    getOperation()->walk([&](LoopOp loop_op) {
      loop_op->setAttr("loop_depth",
                       builder.getI32IntegerAttr(enclosing_loops(loop_op)));
    });

    // Resolve an evt's "auto" loop_level to the innermost enclosing loop; an
    // explicit level written in pasm is kept. Must run before the loops are
    // unwrapped, while the ancestry is intact.
    getOperation()->walk([&](InstrOp instr_op) {
      if (instr_op.getType().str() != "evt")
        return;

      auto lv = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
          instr_op.getParam().get("loop_level"));
      if (lv && lv.getInt() != LOOP_LEVEL_AUTO)
        return;

      update_params(instr_op, builder, /*remove=*/{},
                    {{"loop_level", innermost_loop_level(instr_op)}});
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceLoopOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vesyla::pasm
