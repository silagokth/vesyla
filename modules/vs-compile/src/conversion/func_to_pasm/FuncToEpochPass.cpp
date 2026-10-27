#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "FuncToEpochPass.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

namespace vesyla::conversion::func_to_pasm {
#define GEN_PASS_DEF_FUNCTOEPOCHPASS
#include "conversion/func_to_pasm/Passes.hpp.inc"

namespace {

// A dynamic io dimension spans the whole io address space: IO_ADDR_WIDTH = 16
// in io.mlir. Nothing downstream reads the size, only the rank and the access
// maps, so this only has to be a valid static shape.
constexpr int64_t kIoRows = int64_t{1} << 16;

// The memref an affine load or store goes through, or null for anything else.
mlir::Value accessed_memref(mlir::Operation *op) {
  if (auto read = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(op))
    return read.getMemRef();
  if (auto write = llvm::dyn_cast<mlir::affine::AffineWriteOpInterface>(op))
    return write.getMemRef();
  return nullptr;
}

bool has_id(mlir::Operation *op, llvm::StringRef id) {
  auto attr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  return attr && attr.getValue() == id;
}

class FuncToEpochPass : public impl::FuncToEpochPassBase<FuncToEpochPass> {
public:
  using impl::FuncToEpochPassBase<FuncToEpochPass>::FuncToEpochPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> funcs(
        module.getOps<mlir::func::FuncOp>());
    if (funcs.empty())
      return;

    // Every precondition is checked before anything is rewritten, so a
    // rejected kernel is reported against the IR as it was written.
    mlir::func::FuncOp fn;
    mlir::BlockArgument input, output;
    llvm::SmallVector<mlir::Operation *> allocs;
    if (mlir::failed(check_preconditions(module, funcs, fn, input, output,
                                         allocs)))
      return signalPassFailure();

    mlir::OpBuilder builder(fn);
    hoist_argument(input, "input_buffer", builder);
    hoist_argument(output, "output_buffer", builder);
    hoist_allocs(allocs, builder);
    move_body(fn, builder);
    fn.erase();

    if (mlir::failed(check_postconditions(module)))
      return signalPassFailure();
  }

private:
  mlir::LogicalResult
  check_preconditions(mlir::ModuleOp module,
                      llvm::ArrayRef<mlir::func::FuncOp> funcs,
                      mlir::func::FuncOp &fn, mlir::BlockArgument &input,
                      mlir::BlockArgument &output,
                      llvm::SmallVectorImpl<mlir::Operation *> &allocs) {
    // 1. One kernel function.
    if (funcs.size() != 1)
      return module.emitError("func-to-epoch: expected one kernel function, "
                              "found ")
             << funcs.size();
    fn = funcs.front();

    // 2. A body, and a return without values.
    if (fn.isExternal() || !fn.getBody().hasOneBlock())
      return fn.emitError("func-to-epoch: the kernel must have a body of one "
                          "block");
    if (fn.getNumResults() != 0)
      return fn.emitError("func-to-epoch: the kernel must not return a value; "
                          "its results go to the output buffer");

    // 3. Memref arguments used only by affine loads and stores, and
    // 4. exactly one of them read-only (the input) and one written (the
    //    output): the io resource has one buffer each way.
    llvm::SmallVector<mlir::BlockArgument> inputs, outputs;
    for (mlir::BlockArgument arg : fn.getArguments()) {
      if (!llvm::isa<mlir::MemRefType>(arg.getType()))
        return fn.emitError("func-to-epoch: argument #")
               << arg.getArgNumber()
               << " is not a memref; only buffer arguments are supported";
      bool written = false;
      for (mlir::Operation *user : arg.getUsers()) {
        if (accessed_memref(user) != arg)
          return user->emitError("func-to-epoch: a buffer argument may only "
                                 "be used by affine loads and stores; was the "
                                 "code raised to affine?");
        written |= llvm::isa<mlir::affine::AffineWriteOpInterface>(user);
      }
      (written ? outputs : inputs).push_back(arg);
    }
    if (inputs.size() != 1 || outputs.size() != 1)
      return fn.emitError("func-to-epoch: expected one buffer that is only "
                          "read (the input buffer) and one that is written "
                          "(the output buffer), found ")
             << inputs.size() << " read-only and " << outputs.size()
             << " written";
    input = inputs.front();
    output = outputs.front();

    // 5. Local buffers outside any loop: each becomes one register file for
    //    the whole program. cgeist hoists a C array declared in a loop body
    //    itself, so this only guards hand-written input.
    mlir::Block &body = fn.getBody().front();
    mlir::WalkResult result = fn.walk([&](mlir::Operation *op) {
      if (!llvm::isa<mlir::memref::AllocaOp, mlir::memref::AllocOp>(op))
        return mlir::WalkResult::advance();
      if (op->getBlock() != &body) {
        op->emitError("func-to-epoch: a local buffer must be allocated "
                      "outside any loop");
        return mlir::WalkResult::interrupt();
      }
      allocs.push_back(op);
      return mlir::WalkResult::advance();
    });
    return mlir::failure(result.wasInterrupted());
  }

  // The argument becomes a module-level alloc with the io buffer's id. The
  // accesses only check the rank, so they take the static type as is.
  void hoist_argument(mlir::BlockArgument arg, llvm::StringRef name,
                      mlir::OpBuilder &builder) {
    auto type = llvm::cast<mlir::MemRefType>(arg.getType());
    llvm::SmallVector<int64_t> shape(type.getShape());
    for (int64_t &size : shape)
      if (mlir::ShapedType::isDynamic(size))
        size = kIoRows;
    auto alloc = mlir::memref::AllocOp::create(
        builder, arg.getLoc(),
        mlir::MemRefType::get(shape, type.getElementType()));
    alloc->setAttr("id", mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                      name));
    arg.replaceAllUsesWith(alloc.getResult());
  }

  // Local arrays become module-level allocs, like the register files of a
  // hand-written program.
  void hoist_allocs(llvm::ArrayRef<mlir::Operation *> allocs,
                    mlir::OpBuilder &builder) {
    for (mlir::Operation *op : allocs) {
      auto alloc = mlir::memref::AllocOp::create(
          builder, op->getLoc(),
          llvm::cast<mlir::MemRefType>(op->getResult(0).getType()),
          op->getOperands());
      if (auto id = op->getAttr("id"))
        alloc->setAttr("id", id);
      op->getResult(0).replaceAllUsesWith(alloc.getResult());
      op->erase();
    }
  }

  // The rest of the body goes into the module. A body that already has
  // epochs keeps them; otherwise it all becomes one epoch.
  void move_body(mlir::func::FuncOp fn, mlir::OpBuilder &builder) {
    mlir::Block &body = fn.getBody().front();
    llvm::SmallVector<mlir::Operation *> deallocs;
    fn.walk([&](mlir::memref::DeallocOp op) { deallocs.push_back(op); });
    for (mlir::Operation *op : deallocs)
      op->erase();

    bool has_epochs = llvm::any_of(
        body, [](mlir::Operation &op) { return llvm::isa<pasm::EpochOp>(op); });
    mlir::Block *target = builder.getInsertionBlock();
    mlir::Block::iterator before = builder.getInsertionPoint();
    if (!has_epochs) {
      auto epoch = pasm::EpochOp::create(builder, fn.getLoc(),
                                         builder.getStringAttr("rb0"));
      target = builder.createBlock(&epoch.getBody());
      pasm::YieldOp::create(builder, fn.getLoc());
      before = target->getTerminator()->getIterator();
    }
    for (mlir::Operation &op : llvm::make_early_inc_range(body)) {
      if (llvm::isa<mlir::func::ReturnOp>(op))
        continue;
      op.moveBefore(target, before);
    }
  }

  mlir::LogicalResult check_postconditions(mlir::ModuleOp module) {
    auto internal_error = [&](llvm::StringRef what) {
      return module.emitError("func-to-epoch: internal error, ") << what;
    };

    // 1. No func ops left.
    bool func_left = false;
    module.walk([&](mlir::Operation *op) {
      func_left |= llvm::isa<mlir::func::FuncDialect>(op->getDialect());
    });
    if (func_left)
      return internal_error("a func.func or func.return is left");

    // 2. One input and one output buffer, and
    // 4. only allocs and epochs at module level.
    int inputs = 0, outputs = 0;
    for (mlir::Operation &op : *module.getBody()) {
      if (!llvm::isa<mlir::memref::AllocOp, pasm::EpochOp>(op))
        return internal_error("the module holds an op other than "
                              "memref.alloc and pasm.epoch: ")
               << op.getName();
      inputs += has_id(&op, "input_buffer");
      outputs += has_id(&op, "output_buffer");
    }
    if (inputs != 1 || outputs != 1)
      return internal_error("expected one @input_buffer and one "
                            "@output_buffer, found ")
             << inputs << " and " << outputs;

    // 3. No alloca or dealloc left.
    bool local_left = false;
    module.walk([&](mlir::Operation *op) {
      local_left |= llvm::isa<mlir::memref::AllocaOp, mlir::memref::DeallocOp>(op);
    });
    if (local_left)
      return internal_error("a memref.alloca or memref.dealloc is left");
    return mlir::success();
  }
};

} // namespace
} // namespace vesyla::conversion::func_to_pasm
