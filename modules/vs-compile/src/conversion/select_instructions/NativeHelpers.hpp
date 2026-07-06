#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

// Native match/rewrite helpers called from the inline code blocks in
// patterns.pdll. Keeping the real logic here (rather than in the .pdll) keeps
// it ordinary, testable C++; the .pdll only describes the match structure and
// forwards to these functions.
namespace vesyla {
namespace sel {

// Native constraint: true when `acc` is the loop-carried accumulator of an
// enclosing affine.for (i.e. a region iter_args block argument). Used to
// recognise a reduction accumulate so muli+addi fuses into a MAC.
bool is_iter_args_carry(mlir::Value acc);

// Native rewrite: fuse `mul` (arith.muli) and `add` (arith.addi) into one
// drra.rop kind="dpu" instr="conf" (mode=mac). Replaces `add` with the rop and
// erases `mul`. `a`/`b` are the multiply operands (become the rop operands);
// the accumulator operand of `add` is dropped (implicit in the DPU).
void emit_mac(mlir::PatternRewriter &rewriter, mlir::Operation *mul,
              mlir::Operation *add, mlir::Value a, mlir::Value b);

// Native rewrite: lower a memory read (affine.load / affine.vector_load) into a
// drra.rop kind="rf" instr="evt" with no operands, producing the same result.
void emit_rf_load(mlir::PatternRewriter &rewriter, mlir::Operation *load);

// Native rewrite: lower a memory write (affine.store / affine.vector_store) into
// a drra.rop kind="rf" instr="evt" consuming the stored value, no result.
void emit_rf_store(mlir::PatternRewriter &rewriter, mlir::Operation *store);

} // namespace sel
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__
