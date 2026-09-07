#include "Matcher.hpp"
#include "NativeHelpers.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

namespace vesyla {
namespace conversion {
namespace select_instructions {

namespace {

enum class MemKind { None, Read, Write };

// Every read is interchangeable with every other read, and likewise for
// writes. Whether the address came from an affine map is the program's
// business, not the resource's.
MemKind memKind(mlir::Operation *op) {
  if (llvm::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                mlir::vector::LoadOp, mlir::affine::AffineVectorLoadOp>(op))
    return MemKind::Read;
  if (llvm::isa<mlir::memref::StoreOp, mlir::affine::AffineStoreOp,
                mlir::vector::StoreOp, mlir::affine::AffineVectorStoreOp>(op))
    return MemKind::Write;
  return MemKind::None;
}

// Operand order is the same in all four dialects: a read is
// (memref, indices...) and a write is (value, memref, indices...).
mlir::Value memrefOf(mlir::Operation *op) {
  return memKind(op) == MemKind::Read ? op->getOperand(0) : op->getOperand(1);
}

mlir::Value storedValueOf(mlir::Operation *op) { return op->getOperand(0); }

// What the access actually moves. This is what tells a word access from a bulk
// one -- i16 selects @word_read, vector<16xi16> selects @bulk_read.
mlir::Type transferredType(mlir::Operation *op) {
  return memKind(op) == MemKind::Read ? op->getResult(0).getType()
                                      : op->getOperand(0).getType();
}

// The program names its IO buffers with an id on the alloc, because MLIR drops
// SSA value names at parse time and %input_buffer would otherwise be gone by
// the time this pass runs.
bool isIoBuffer(mlir::Value memref) {
  auto alloc =
      llvm::dyn_cast_or_null<mlir::memref::AllocOp>(memref.getDefiningOp());
  if (!alloc)
    return false;
  auto id = alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  if (!id)
    return false;
  return id.getValue() == "input_buffer" || id.getValue() == "output_buffer";
}

} // namespace

bool Matcher::isAccumulatorRead(mlir::Value patternValue) const {
  if (!pattern_.accumulator)
    return false;
  auto load = llvm::dyn_cast_or_null<mlir::memref::LoadOp>(
      patternValue.getDefiningOp());
  return load && load->getOperand(0) == pattern_.accumulator;
}

bool Matcher::matchMemref(mlir::Value patternMemref, mlir::Value inputMemref,
                          MatchResult &result) const {
  // An io pattern only applies to a memref the program declared as an IO
  // buffer. rf carries no such guard, so it matches anything and picks up
  // whatever the higher tiers left -- which is what makes it the fallback.
  if (pattern_.guardedByBufferRole() && !isIoBuffer(inputMemref))
    return false;

  // The memref's own type is deliberately not compared: the pattern states the
  // resource's geometry (rf's 64 words, io's dynamic external buffer) and the
  // program states its own.
  auto [it, inserted] = result.bindings.try_emplace(patternMemref, inputMemref);
  return inserted || it->second == inputMemref;
}

bool Matcher::matchValue(mlir::Value patternValue, mlir::Value inputValue,
                         MatchResult &result) const {
  // The accumulator rule. In the pattern the running total comes out of a named
  // register; in the program it arrives as an affine.for loop carry, which is
  // an SSA block argument with no defining op at all. Recursing would compare a
  // memref.load against nothing.
  if (isAccumulatorRead(patternValue))
    return vesyla::sel::is_iter_args_carry(inputValue);

  if (llvm::isa<mlir::BlockArgument>(patternValue)) {
    auto [it, inserted] = result.bindings.try_emplace(patternValue, inputValue);
    return inserted || it->second == inputValue;
  }

  return matchOp(patternValue.getDefiningOp(), inputValue.getDefiningOp(),
                 result);
}

bool Matcher::matchOp(mlir::Operation *patternOp, mlir::Operation *inputOp,
                      MatchResult &result) const {
  if (!patternOp || !inputOp)
    return false;

  const MemKind patternKind = memKind(patternOp);
  const MemKind inputKind = memKind(inputOp);
  if (patternKind != MemKind::None || inputKind != MemKind::None) {
    if (patternKind != inputKind)
      return false;
    if (transferredType(patternOp) != transferredType(inputOp))
      return false;
    if (!matchMemref(memrefOf(patternOp), memrefOf(inputOp), result))
      return false;
    // A write also carries the value being stored, and that is real datapath,
    // so it is matched. The indices are not: the pattern's own address
    // arithmetic is descriptive, and we never walk into it.
    if (patternKind == MemKind::Write &&
        !matchValue(storedValueOf(patternOp), storedValueOf(inputOp), result))
      return false;
    result.cone.push_back(inputOp);
    return true;
  }

  if (patternOp->getName() != inputOp->getName())
    return false;
  if (patternOp->getNumResults() != inputOp->getNumResults())
    return false;
  for (auto [patternResult, inputResult] :
       llvm::zip(patternOp->getResults(), inputOp->getResults()))
    if (patternResult.getType() != inputResult.getType())
      return false;

  // Only the attributes the pattern states have to agree. The program carries
  // metadata of its own (id, resource) that says nothing about the shape, so
  // requiring equality both ways would never match anything.
  for (mlir::NamedAttribute attr : patternOp->getAttrs())
    if (inputOp->getAttr(attr.getName()) != attr.getValue())
      return false;

  if (patternOp->getNumOperands() != inputOp->getNumOperands())
    return false;
  for (auto [patternOperand, inputOperand] :
       llvm::zip(patternOp->getOperands(), inputOp->getOperands()))
    if (!matchValue(patternOperand, inputOperand, result))
      return false;

  result.cone.push_back(inputOp);
  return true;
}

bool Matcher::match(mlir::Operation *candidate, MatchResult &result) const {
  if (!pattern_.root)
    return false;
  if (!matchOp(pattern_.root, candidate, result))
    return false;

  // A pattern that writes the accumulator without reading it is a reset. On its
  // own that is a bare constant, which every zero in the program would satisfy,
  // so it only counts where the program uses it to seed a loop carry.
  if (pattern_.initialisesAccumulator() &&
      !vesyla::sel::feeds_iter_args_init(candidate))
    return false;

  return true;
}

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla
