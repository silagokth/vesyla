#include "Scheduler.hpp"
#include "Generator.hpp"

namespace vesyla {
namespace schedule {
void Scheduler::save_mlir(mlir::ModuleOp &module, const std::string &filename) {
  std::error_code error_code;
  llvm::raw_fd_ostream ofs(filename, error_code);
  if (error_code) {
    LOG_FATAL << "Error: Failed to open file for writing: " << filename << "\n";
    std::exit(EXIT_FAILURE);
  }
  module.print(ofs);
  ofs.close();
}

void Scheduler::run(std::string pasm_file, std::string output_dir) {
  Parser parser;
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<vesyla::pasm::PasmDialect>();
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<vesyla::pasm::PasmDialect>();
  mlir::ModuleOp module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  parser.parse(pasm_file, &module);

  run(module, output_dir);
}

void Scheduler::run(mlir::ModuleOp &module, std::string output_dir) {

  // get the environment variable: VESYLA_SUITE_PATH_COMPONENTS
  std::string VESYLA_SUITE_PATH_COMPONENTS =
      std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          ? std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          : "";
  if (VESYLA_SUITE_PATH_COMPONENTS == "") {
    LOG_FATAL << "Error: VESYLA_SUITE_PATH_COMPONENTS is not set.\n";
    std::exit(EXIT_FAILURE);
  }

  // create temp directory for scheduler
  std::string module_debug_path = output_dir + "/debug/schedule";
  if (!std::filesystem::exists(module_debug_path)) {
    std::filesystem::create_directories(module_debug_path);
  }

  // Create a PassManager
  mlir::PassManager pm(module.getContext());

  save_mlir(module, module_debug_path + "/0.mlir");
  pm.addPass(vesyla::pasm::createAddSlotPortPass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/1.mlir");
  pm.addPass(vesyla::pasm::createAddDefaultValuePass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/2.mlir");

  std::string temp_dir = vesyla::util::SysPath::temp_dir();
  pm.addPass(vesyla::pasm::createScheduleEpochPass(
      {VESYLA_SUITE_PATH_COMPONENTS, temp_dir}));
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/3.mlir");

  pm.addPass(vesyla::pasm::createReplaceLoopOp());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/4.mlir");

  pm.addPass(vesyla::pasm::createMergeRawOp());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/5.mlir");

  pm.addPass(vesyla::pasm::createAddHaltPass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: Pass pipeline failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/6.mlir");

  // Save the transformed module to ASM and BIN files
  std::string codegen_path = output_dir;
  if (!std::filesystem::exists(codegen_path)) {
    std::filesystem::create_directories(codegen_path);
  }
  std::string output_filename = "instr";
  Generator g;
  g.generate(module, codegen_path, output_filename);
  LOG_INFO << "Successfully generated ASM and BIN files in directory: "
           << codegen_path;
}
} // namespace schedule
} // namespace vesyla
