#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Support/Common.hpp"
#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/RandName.hpp"

#include <map>
#include <utility>

namespace vesyla::pasm {
#define GEN_PASS_DEF_REPLACELOOPOP
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Sequencer calc modes (fabric ISA calc verbo_map keys).
constexpr int CALC_MODE_ADD = 1;
constexpr int CALC_MODE_SUB = 2;
constexpr int CALC_MODE_LT = 21;
// operand2 source flag: 0 = static immediate, 1 = dynamic (register index).
constexpr int OPND_STATIC = 0;
constexpr int OPND_DYNAMIC = 1;

// Registers reserved for loop control. The counter also rides the act signal
// (it is what act mode 2 forwards to the resources), so it must sit in the act
// register window [r4..r11].
// TODO: source these from register allocation once loop-scope liveness exists,
// and reconcile with the act-mode-2 prep allocation in RegisterAllocator.cpp,
// which currently also claims r4.
constexpr int LOOP_COUNTER_REG = 4;
constexpr int LOOP_FLAG_REG = 1;

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
        llvm::outs() << "Error: loop body of cell (" << entry.first.first << ", "
                     << entry.first.second << ") has " << info.body_len
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
                  i32_attr(rewriter, "operand1", LOOP_COUNTER_REG),
                  i32_attr(rewriter, "operand2_sd", OPND_DYNAMIC),
                  i32_attr(rewriter, "operand2", LOOP_COUNTER_REG),
                  i32_attr(rewriter, "result", LOOP_COUNTER_REG)});

      // Epilogue: increment, test i < iter, branch back while looping. Placed
      // at the end of the cell's last body slice, before its terminator.
      mlir::Block *last_block = raw_op_block(info.last_raw, rewriter);
      rewriter.setInsertionPoint(last_block->getTerminator());
      make_instr(rewriter, loc, "calc",
                 {i32_attr(rewriter, "mode", CALC_MODE_ADD),
                  i32_attr(rewriter, "operand1", LOOP_COUNTER_REG),
                  i32_attr(rewriter, "operand2_sd", OPND_STATIC),
                  i32_attr(rewriter, "operand2", 1),
                  i32_attr(rewriter, "result", LOOP_COUNTER_REG)});
      make_instr(rewriter, loc, "calc",
                 {i32_attr(rewriter, "mode", CALC_MODE_LT),
                  i32_attr(rewriter, "operand1", LOOP_COUNTER_REG),
                  i32_attr(rewriter, "operand2_sd", OPND_STATIC),
                  i32_attr(rewriter, "operand2", iter),
                  i32_attr(rewriter, "result", LOOP_FLAG_REG)});
      make_instr(rewriter, loc, "brn",
                 {i32_attr(rewriter, "reg", LOOP_FLAG_REG),
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
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceLoopOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vesyla::pasm
