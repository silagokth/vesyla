#include "plog/Log.h"
#include "schedule/Scheduler.hpp"
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "pasm/Dialect.hpp"
#include "pasm/ExtractCellsPass.hpp"
#include "pasm/InterconnectPass.hpp"
#include "pasm/Passes.hpp"

namespace {

bool module_uses_only_pasm_dialect(mlir::ModuleOp module) {
  llvm::StringRef pasm_ns = vesyla::pasm::PasmDialect::getDialectNamespace();
  bool ok = true;
  module.walk([&](mlir::Operation *op) {
    if (op == module.getOperation()) {
      return mlir::WalkResult::advance();
    }
    mlir::Dialect *dialect = op->getDialect();
    if (!dialect || dialect->getNamespace() != pasm_ns) {
      LOG_FATAL << "Error: op '" << op->getName().getStringRef().str()
                << "' is not in the 'pasm' dialect.";
      ok = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return ok;
}

int run_mlir_mode(const std::string &mlir_file) {
  if (!std::filesystem::exists(mlir_file)) {
    LOG_FATAL << "Error: MLIR file does not exist: " << mlir_file;
    return -1;
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<vesyla::pasm::PasmDialect>();
  // Tolerate unknown dialect prefixes during parsing so that the membership
  // check below can report them cleanly instead of MLIR aborting the process.
  context.allowUnregisteredDialects(true);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_file, &context);
  if (!module) {
    LOG_FATAL << "Error: Failed to parse MLIR file: " << mlir_file;
    return -1;
  }

  if (!module_uses_only_pasm_dialect(*module)) {
    return -1;
  }

  mlir::PassManager extract_pm(&context);
  extract_pm.addPass(vesyla::pasm::createExtractCellsPass());
  if (mlir::failed(extract_pm.run(*module))) {
    LOG_FATAL << "Error: ExtractCellsPass failed.";
    return -1;
  }
  module->print(llvm::errs());

  mlir::PassManager pm(&context);
  pm.addPass(vesyla::pasm::createInterconnectPass());
  if (mlir::failed(pm.run(*module))) {
    LOG_FATAL << "Error: InterconnectPass failed.";
    return -1;
  }

  return 0;
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

  // MLIR-input mode: parse a pre-built pasm-dialect .mlir file, run only the
  // InterconnectPass on it, and exit.
  if (!mlir_file.empty()) {
    LOG_INFO << "Running mlir mode";
    return run_mlir_mode(mlir_file);
  }

  vesyla::pasm::Config cfg;
  cfg.set_arch_json(arch_file);
  cfg.set_isa_json(isa_file);

  if (!cpp_file.empty()) {
    LOG_FATAL << "Compilation from C++ model is not supported right now!";
    return -1;
  }

  if (!pasm_file.empty() && !std::filesystem::exists(pasm_file)) {
    if (!std::filesystem::exists(pasm_file)) {
      LOG_FATAL << "Error: PASM file does not exist: " << pasm_file;
      return -1;
    }
  }

  vesyla::schedule::Scheduler scheduler;
  scheduler.run(pasm_file, output_dir, allow_unsafe);

  // clean up debug intermediates unless -d/--debug was passed
  if (!keep_debug) {
    std::string mzn_dir = output_dir + "/debug/minizinc";
    if (std::filesystem::exists(mzn_dir)) {
      std::filesystem::remove_all(mzn_dir);
    }
  }

  return 0;
}
