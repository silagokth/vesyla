#include "plog/Log.h"
#include "schedule/Parser.hpp"
#include "schedule/Scheduler.hpp"
#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "conversion/affine_to_pasm/AffineToInstrPass.hpp"
#include "pasm/Dialect.hpp"
#include "pasm/ExtractCellsPass.hpp"
#include "pasm/FlattenCellsPass.hpp"
#include "pasm/InterconnectPass.hpp"
#include "pasm/Passes.hpp"

namespace {

mlir::OwningOpRef<mlir::ModuleOp> run_mlir_mode(const std::string &mlir_file,
                                                mlir::MLIRContext &context) {
  if (!std::filesystem::exists(mlir_file)) {
    LOG_FATAL << "Error: MLIR file does not exist: " << mlir_file;
    return nullptr;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_file, &context);
  if (!module) {
    LOG_FATAL << "Error: Failed to parse MLIR file: " << mlir_file;
    return nullptr;
  }

  mlir::PassManager affine_pm(&context);
  affine_pm.addPass(
      vesyla::conversion::affine_to_pasm::createAffineToInstrPass());
  if (mlir::failed(affine_pm.run(*module))) {
    LOG_FATAL << "Error: AffineToInstrPass failed.";
    return nullptr;
  }
  module->print(llvm::errs());

  mlir::PassManager extract_pm(&context);
  extract_pm.addPass(vesyla::pasm::createExtractCellsPass());
  if (mlir::failed(extract_pm.run(*module))) {
    LOG_FATAL << "Error: ExtractCellsPass failed.";
    return nullptr;
  }
  module->print(llvm::errs());

  mlir::PassManager pm(&context);
  pm.addPass(vesyla::pasm::createInterconnectPass());
  if (mlir::failed(pm.run(*module))) {
    LOG_FATAL << "Error: InterconnectPass failed.";
    return nullptr;
  }

  mlir::PassManager flatten_pm(&context);
  flatten_pm.addPass(vesyla::pasm::createFlattenCellsPass());
  if (mlir::failed(flatten_pm.run(*module))) {
    LOG_FATAL << "Error: FlattenCellsPass failed.";
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> run_pasm_mode(const std::string &pasm_file,
                                                mlir::MLIRContext &context) {
  if (!std::filesystem::exists(pasm_file)) {
    LOG_FATAL << "Error: PASM file does not exist: " << pasm_file;
    return nullptr;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  vesyla::schedule::Parser parser;
  std::string pasm_file_copy = pasm_file;
  mlir::ModuleOp module_ref = *module;
  parser.parse(pasm_file_copy, &module_ref);
  return module;
}

} // namespace

int main(int argc, char **argv) {

  // seed the random number generator
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(static_cast<unsigned int>(seed));

  // Set up logging system
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::debug).addAppender(&consoleAppender);

  // Parsing command line arguments
  vesyla::util::MiniArgs args;
  args.parse(argc, argv);

  if (args.flag("h") || args.flag("help")) {
    LOG_INFO << "Usage: vesyla compile --arch FILE --isa FILE --pasm FILE "
                "[--output DIR] [--allow-unsafe] [-d|--debug]";
    LOG_INFO << "Or";
    LOG_INFO << "vesyla compile --arch FILE --isa FILE --mlir FILE "
                "[--output DIR]";
    return 0;
  }

  std::string arch_file = args.get("arch", args.get("a"));
  std::string isa_file = args.get("isa", args.get("i"));
  std::string pasm_file = args.get("pasm", args.get("p"));
  std::string cpp_file = args.get("cpp", args.get("c"));
  std::string mlir_file = args.get("mlir", args.get("m"));
  std::string output_dir = args.get("output", args.get("o", "."));
  bool allow_unsafe = args.flag("allow-unsafe");
  bool keep_debug = args.flag("d") || args.flag("debug");

  if (arch_file.empty() || isa_file.empty()) {
    LOG_FATAL << "Required arguments missing, see --help for usage.";
    return -1;
  }

  // File existence checks
  if (!std::filesystem::exists(arch_file)) {
    LOG_FATAL << "Error: Architecture file does not exist: ";
    return -1;
  }
  if (!std::filesystem::exists(isa_file)) {
    LOG_FATAL << "Error: ISA file does not exist: " << isa_file;
    return -1;
  }

  // Create output directory if it doesn't exist
  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  vesyla::util::GlobalVar::puts("__OUTPUT_DIR__", output_dir);

  if (!cpp_file.empty()) {
    LOG_FATAL << "Compilation from C++ model is not supported right now!";
    return -1;
  }

  if (mlir_file.empty() && pasm_file.empty()) {
    LOG_FATAL << "No input file provided. Pass --mlir FILE or --pasm FILE.";
    return -1;
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<vesyla::pasm::PasmDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (!mlir_file.empty()) {
    LOG_INFO << "Running mlir mode";
    module = run_mlir_mode(mlir_file, context);
  } else {
    LOG_INFO << "Running pasm mode";
    module = run_pasm_mode(pasm_file, context);
  }
  if (!module) {
    return -1;
  }

  vesyla::pasm::Config cfg;
  cfg.set_arch_json(arch_file);
  cfg.set_isa_json(isa_file);

  mlir::ModuleOp module_op = *module;
  vesyla::schedule::Scheduler scheduler;
  scheduler.run(module_op, output_dir, allow_unsafe);

  // clean up debug intermediates unless -d/--debug was passed
  if (!keep_debug) {
    std::string mzn_dir = output_dir + "/debug/minizinc";
    if (std::filesystem::exists(mzn_dir)) {
      std::filesystem::remove_all(mzn_dir);
    }
  }

  return 0;
}
