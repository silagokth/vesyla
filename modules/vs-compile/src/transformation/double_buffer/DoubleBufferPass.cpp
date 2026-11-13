#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "DoubleBufferPass.hpp"
#include "conversion/select_instructions/NativeHelpers.hpp"
#include "vesyla/Support/GlobalVar.hpp"

namespace vesyla::transformation::double_buffer {
#define GEN_PASS_DEF_DOUBLEBUFFERPASS
#include "transformation/double_buffer/Passes.hpp.inc"

namespace {

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::affine::AffineForOp;

// Off with vs-compile --no-double-buffer, and for the retry without it.
bool enabled() {
  std::string v;
  return !vesyla::util::GlobalVar::gets("__DOUBLE_BUFFER__", v) || v != "0";
}

bool is_access(mlir::Operation *op) {
  return llvm::isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                   mlir::affine::AffineVectorLoadOp,
                   mlir::affine::AffineVectorStoreOp>(op);
}

bool is_write(mlir::Operation *op) {
  return llvm::isa<mlir::affine::AffineStoreOp,
                   mlir::affine::AffineVectorStoreOp>(op);
}

AffineMap map_of(mlir::Operation *op) {
  if (auto r = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(op))
    return r.getAffineMap();
  return llvm::cast<mlir::affine::AffineWriteOpInterface>(op).getAffineMap();
}

llvm::SmallVector<mlir::Value> operands_of(mlir::Operation *op) {
  if (auto r = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(op))
    return llvm::to_vector(r.getMapOperands());
  return llvm::to_vector(
      llvm::cast<mlir::affine::AffineWriteOpInterface>(op).getMapOperands());
}

// `op` again with `memref`, `map` and `operands`, keeping its attributes.
void rebuild(mlir::Operation *op, mlir::Value memref, AffineMap map,
             llvm::ArrayRef<mlir::Value> operands) {
  mlir::OpBuilder b(op);
  mlir::Operation *n = nullptr;
  if (auto o = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op))
    n = mlir::affine::AffineLoadOp::create(b, op->getLoc(), memref, map,
                                           operands);
  else if (auto o = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op))
    n = mlir::affine::AffineStoreOp::create(b, op->getLoc(),
                                            o.getValueToStore(), memref, map,
                                            operands);
  else if (auto o = llvm::dyn_cast<mlir::affine::AffineVectorLoadOp>(op))
    n = mlir::affine::AffineVectorLoadOp::create(
        b, op->getLoc(), o.getVectorType(), memref, map, operands);
  else if (auto o = llvm::dyn_cast<mlir::affine::AffineVectorStoreOp>(op))
    n = mlir::affine::AffineVectorStoreOp::create(
        b, op->getLoc(), o.getValueToStore(), memref, map, operands);
  n->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  op->replaceAllUsesWith(n);
  op->erase();
}

mlir::Value memref_of(mlir::Operation *op) {
  return is_write(op) ? op->getOperand(1) : op->getOperand(0);
}

// A row fetch: a vector store of a vector read from an io buffer.
bool is_fetch(mlir::Operation *op) {
  auto st = llvm::dyn_cast<mlir::affine::AffineVectorStoreOp>(op);
  if (!st)
    return false;
  auto ld = st.getValueToStore().getDefiningOp<mlir::affine::AffineVectorLoadOp>();
  return ld && vesyla::sel::is_io_buffer(ld.getMemRef());
}

// The op directly in `block` that holds `op`, or null.
mlir::Operation *in_block(mlir::Block *block, mlir::Operation *op) {
  return block->findAncestorOpInBlock(*op);
}

// The loop a staging buffer is refilled in every pass: its writes are row
// fetches in the loop's body, ahead of a loop directly in that body that
// reads it, and nothing else touches it.
AffineForOp host_loop(mlir::memref::AllocOp alloc) {
  llvm::SmallVector<mlir::Operation *> writes, reads;
  for (mlir::Operation *user : alloc->getUsers()) {
    if (!is_access(user) || memref_of(user) != alloc.getResult())
      return {};
    (is_write(user) ? writes : reads).push_back(user);
  }
  if (writes.empty() || reads.empty() || !llvm::all_of(writes, is_fetch))
    return {};

  AffineForOp loop = writes.front()->getParentOfType<AffineForOp>();
  while (loop && !llvm::all_of(llvm::concat<mlir::Operation *>(writes, reads),
                               [&](mlir::Operation *op) {
                                 return loop->isProperAncestor(op);
                               }))
    loop = loop->getParentOfType<AffineForOp>();
  if (!loop || loop.getNumIterOperands() != 0 || !loop.hasConstantBounds() ||
      loop.getConstantLowerBound() != 0 || loop.getStepAsInt() != 1)
    return {};
  int64_t trips = loop.getConstantUpperBound();
  if (trips < 2 || trips % 2 != 0)
    return {};

  mlir::Block *body = loop.getBody();
  for (mlir::Operation *r : reads) {
    auto stream = llvm::dyn_cast<AffineForOp>(in_block(body, r));
    if (!stream)
      return {};
    for (mlir::Operation *w : writes) {
      mlir::Operation *at = in_block(body, w);
      if (at == stream.getOperation() || !at->isBeforeInBlock(stream))
        return {};
    }
  }
  // Every pass refills the same rows.
  for (mlir::Operation *w : writes)
    if (llvm::is_contained(operands_of(w), loop.getInductionVar()))
      return {};
  // The counter only addresses: it can be rewritten into the accesses.
  for (mlir::Operation *user : loop.getInductionVar().getUsers())
    if (!is_access(user) && !llvm::isa<mlir::affine::AffineApplyOp>(user))
      return {};
  return loop;
}

// Splits `loop` (0 to n) into jo (n/2) around it and itself (0 to 2), its
// counter becoming 2 * jo + ji in every access. Returns jo.
AffineForOp split(AffineForOp loop) {
  mlir::MLIRContext *ctx = loop.getContext();
  mlir::OpBuilder b(loop);
  int64_t trips = loop.getConstantUpperBound();
  auto outer = AffineForOp::create(b, loop.getLoc(), 0, trips / 2, 1);
  loop->moveBefore(outer.getBody()->getTerminator());
  loop.setConstantUpperBound(2);

  b.setInsertionPointToStart(loop.getBody());
  AffineExpr d0 = mlir::getAffineDimExpr(0, ctx);
  AffineExpr d1 = mlir::getAffineDimExpr(1, ctx);
  auto whole = mlir::affine::AffineApplyOp::create(
      b, loop.getLoc(), AffineMap::get(2, 0, d0 * 2 + d1),
      mlir::ValueRange{outer.getInductionVar(), loop.getInductionVar()});
  loop.getInductionVar().replaceAllUsesExcept(whole, whole);

  llvm::SmallVector<mlir::Operation *> accesses;
  loop.walk([&](mlir::Operation *op) {
    if (is_access(op))
      accesses.push_back(op);
  });
  for (mlir::Operation *op : accesses) {
    AffineMap map = map_of(op);
    llvm::SmallVector<mlir::Value> operands = operands_of(op);
    mlir::affine::fullyComposeAffineMapAndOperands(&map, &operands);
    mlir::affine::canonicalizeMapAndOperands(&map, &operands);
    rebuild(op, memref_of(op), map, operands);
  }
  if (whole->use_empty())
    whole->erase();
  return outer;
}

// Gives `alloc` a second slot: every access moves by a slot when `ji` is 1.
void double_slots(mlir::memref::AllocOp alloc, mlir::Value ji) {
  auto type = llvm::cast<mlir::MemRefType>(alloc.getType());
  int64_t words = type.getShape()[0];
  mlir::OpBuilder b(alloc);
  auto doubled = mlir::memref::AllocOp::create(
      b, alloc.getLoc(),
      mlir::MemRefType::get({2 * words}, type.getElementType()));
  doubled->setDiscardableAttrs(alloc->getDiscardableAttrDictionary());

  for (mlir::Operation *op : llvm::to_vector(alloc->getUsers())) {
    AffineMap map = map_of(op);
    llvm::SmallVector<mlir::Value> operands = operands_of(op);
    // A vector access addresses the buffer in rows.
    int64_t slot = words;
    mlir::Type value = is_write(op) ? op->getOperand(0).getType()
                                    : op->getResult(0).getType();
    if (auto vector = llvm::dyn_cast<mlir::VectorType>(value))
      slot = words / vector.getNumElements();
    unsigned dims = map.getNumDims();
    AffineExpr moved =
        map.getResult(0).shiftDims(dims, 0) +
        mlir::getAffineDimExpr(dims, alloc.getContext()) * slot;
    operands.insert(operands.begin() + dims, ji);
    rebuild(op, doubled,
            AffineMap::get(dims + 1, map.getNumSymbols(), moved), operands);
  }
  alloc->erase();
}

class DoubleBufferPass : public impl::DoubleBufferPassBase<DoubleBufferPass> {
public:
  using impl::DoubleBufferPassBase<DoubleBufferPass>::DoubleBufferPassBase;

  void runOnOperation() final {
    if (!enabled())
      return;
    mlir::ModuleOp module = getOperation();

    llvm::MapVector<mlir::Operation *,
                    llvm::SmallVector<mlir::memref::AllocOp>>
        by_loop;
    module.walk([&](mlir::memref::AllocOp alloc) {
      if (vesyla::sel::is_io_buffer(alloc.getResult()))
        return;
      if (AffineForOp loop = host_loop(alloc))
        by_loop[loop.getOperation()].push_back(alloc);
    });

    for (auto &[op, allocs] : by_loop) {
      auto loop = llvm::cast<AffineForOp>(op);
      split(loop);
      for (mlir::memref::AllocOp alloc : allocs)
        double_slots(alloc, loop.getInductionVar());
    }
  }
};

} // namespace
} // namespace vesyla::transformation::double_buffer
