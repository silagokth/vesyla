#include "plog/Log.h"
#include "vesyla/Parser/PasmTextParser.hpp"
#include "vesyla/Pipeline/PasmPipeline.hpp"
#include <cstdlib>
#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "conversion/affine_to_pasm/AffineToRepPass.hpp"
#include "conversion/drra_to_pasm/CreateConstraintsPass.hpp"
#include "conversion/drra_to_pasm/DrraToPasmPass.hpp"
#include "conversion/select_instructions/SelectInstructionsPass.hpp"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/Transforms/ExtractCellsPass.hpp"
#include "vesyla/Dialect/Pasm/Transforms/FlattenCellsPass.hpp"
#include "vesyla/Dialect/Pasm/Transforms/GenerateIcdepPass.hpp"
#include "vesyla/Dialect/Pasm/Transforms/InterconnectPass.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Dialect/Pasm/Transforms/RessourceBindingPass.hpp"
#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/SysPath.hpp"

namespace {

// Dump a module to a file, mirroring the scheduler's debug-mlir dumps.
void save_mlir(mlir::ModuleOp module, const std::string &filename) {
  std::error_code error_code;
  llvm::raw_fd_ostream ofs(filename, error_code);
  if (error_code) {
    LOG_FATAL << "Error: Failed to open file for writing: " << filename;
    return;
  }
  module.print(ofs);
  ofs.close();
}

mlir::OwningOpRef<mlir::ModuleOp> run_mlir_mode(const std::string &mlir_file,
                                                const std::string &output_dir,
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

  // Dump the module after each pass into a debug folder, mirroring the
  // scheduler. The <prefix>0 file is the parsed input; subsequent files follow
  // the passes. The debug folder, stage-file prefix, and extension all come
  // from the "output" section of config.json.
  vesyla::pasm::Config cfg;
  std::string module_debug_path =
      output_dir + "/" + cfg.output_path("compile_debug_dir");
  if (!std::filesystem::exists(module_debug_path)) {
    std::filesystem::create_directories(module_debug_path);
  }
  const std::string stage_prefix = cfg.output_path("compile_stage_prefix");
  const std::string stage_ext = cfg.output_path("stage_ext");
  auto stage_file = [&](int i) {
    return module_debug_path + "/" + stage_prefix + std::to_string(i) +
           stage_ext;
  };
  save_mlir(*module, stage_file(0));

  // SelectInstructionsPass runs first: it selects drra.rop instructions from the
  // upstream affine + arith ops (pattern-based, PDLL). Passes downstream all
  // consume the drra form it produces. On input that is already in drra form
  // (no arith/affine-access ops) it is a no-op.
  mlir::PassManager select_pm(&context);
  select_pm.addPass(
      vesyla::conversion::select_instructions::createSelectInstructionsPass());
  if (mlir::failed(select_pm.run(*module))) {
    LOG_FATAL << "Error: SelectInstructionsPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(1));

  // RessourceBindingPass runs after instruction selection and before icdep
  // generation: it binds each selected instruction to a concrete hardware
  // resource so downstream passes see the final resource assignment.
  mlir::PassManager ressource_binding_pm(&context);
  ressource_binding_pm.addPass(vesyla::pasm::createRessourceBindingPass());
  if (mlir::failed(ressource_binding_pm.run(*module))) {
    LOG_FATAL << "Error: RessourceBindingPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(2));

  // GenerateIcdepPass must run before DrraToPasmPass: it derives interconnect
  // dependencies from the drra.rop SSA def-use chains and their `resource`/`id`
  // attributes. DrraToPasmPass lowers each drra.rop into a region-form pasm.rop
  // that has no SSA results and splits `resource` into col/port/row/slot, so
  // the icdep pass would find nothing if it ran afterwards. The pasm.icdep ops
  // it inserts are left untouched by DrraToPasmPass.
  mlir::PassManager generate_icdep_pm(&context);
  generate_icdep_pm.addPass(vesyla::pasm::createGenerateIcdepPass());
  if (mlir::failed(generate_icdep_pm.run(*module))) {
    LOG_FATAL << "Error: GenerateIcdepPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(3));

  mlir::PassManager create_constraints_pm(&context);
  create_constraints_pm.addPass(
      vesyla::conversion::drra_to_pasm::createCreateConstraintsPass());
  if (mlir::failed(create_constraints_pm.run(*module))) {
    LOG_FATAL << "Error: CreateConstraintsPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(4));

  mlir::PassManager drra_to_pasm_pm(&context);
  drra_to_pasm_pm.addPass(
      vesyla::conversion::drra_to_pasm::createDrraToPasmPass());
  if (mlir::failed(drra_to_pasm_pm.run(*module))) {
    LOG_FATAL << "Error: DrraToPasmPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(5));

  mlir::PassManager affine_pm(&context);
  affine_pm.addPass(
      vesyla::conversion::affine_to_pasm::createAffineToRepPass());
  if (mlir::failed(affine_pm.run(*module))) {
    LOG_FATAL << "Error: AffineToRepPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(6));

  mlir::PassManager extract_pm(&context);
  extract_pm.addPass(vesyla::pasm::createExtractCellsPass());
  if (mlir::failed(extract_pm.run(*module))) {
    LOG_FATAL << "Error: ExtractCellsPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(7));

  mlir::PassManager pm(&context);
  pm.addPass(vesyla::pasm::createInterconnectPass());
  if (mlir::failed(pm.run(*module))) {
    LOG_FATAL << "Error: InterconnectPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(8));

  mlir::PassManager flatten_pm(&context);
  flatten_pm.addPass(vesyla::pasm::createFlattenCellsPass());
  if (mlir::failed(flatten_pm.run(*module))) {
    LOG_FATAL << "Error: FlattenCellsPass failed.";
    return nullptr;
  }
  save_mlir(*module, stage_file(9));

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
                "[--config FILE] [--output DIR] [--allow-unsafe] [-d|--debug]";
    LOG_INFO << "Or";
    LOG_INFO << "vesyla compile --arch FILE --isa FILE --mlir FILE "
                "[--config FILE] [--output DIR]";
    return 0;
  }

  std::string arch_file = args.get("arch", args.get("a"));
  std::string isa_file = args.get("isa", args.get("i"));
  // Top-level config; defaults to the one shipped next to the executable
  // (build/config or install/config). Currently it holds the port table.
  std::string config_file = args.get("config");
  bool config_explicit = !config_file.empty();
  if (config_file.empty()) {
    config_file = vesyla::util::SysPath::prog_dir() + "config/config.json";
  }
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
  if (!std::filesystem::exists(config_file)) {
    if (config_explicit) {
      LOG_FATAL << "Error: Config file does not exist: " << config_file;
      return -1;
    }
    LOG_WARNING << "Default config file not found at " << config_file
                << "; using built-in port defaults.";
    config_file.clear();
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
  context.getOrLoadDialect<vesyla::drra::DrraDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  // Passes run during mlir mode read the config: GenerateIcdepPass needs the
  // port table and AddDefaultValuePass needs the ISA, so both are loaded
  // before the passes. arch is only consumed later by the scheduler (and its
  // loader does unguarded traversal), so it stays after the passes.
  vesyla::pasm::Config cfg;
  if (!config_file.empty()) {
    cfg.set_config_json(config_file);
  }
  cfg.set_isa_json(isa_file);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (!mlir_file.empty()) {
    LOG_INFO << "Running mlir mode";
    module = run_mlir_mode(mlir_file, output_dir, context);
  } else {
    LOG_INFO << "Running pasm mode";
    module = run_pasm_mode(pasm_file, context);
  }
  if (!module) {
    return -1;
  }

  cfg.set_arch_json(arch_file);

  mlir::ModuleOp module_op = *module;
  vesyla::schedule::Scheduler scheduler;
  scheduler.run(module_op, output_dir, allow_unsafe, keep_debug);

  // clean up debug intermediates unless -d/--debug was passed
  if (!keep_debug) {
    std::string mzn_dir = output_dir + "/" + cfg.output_path("minizinc_dir");
    if (std::filesystem::exists(mzn_dir)) {
      std::filesystem::remove_all(mzn_dir);
    }
  }

  return 0;
}
