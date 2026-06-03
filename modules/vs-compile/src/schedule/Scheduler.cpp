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

void Scheduler::run(mlir::ModuleOp &module, std::string output_dir,
                    bool allow_unsafe = false, bool keep_debug_files) {

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

  std::string zero_mlir =
      std::filesystem::absolute(module_debug_path + "/0.mlir").string();
  save_mlir(module, zero_mlir);

  std::string viz_script_grouped =
      vesyla::util::SysPath::prog_dir() + "scripts/script_grouped.py";
  std::string vis_dir =
      std::filesystem::absolute(output_dir + "/debug/vis").string();
  std::filesystem::create_directories(vis_dir);
  if (std::filesystem::exists(viz_script_grouped)) {
    try {
      std::string cmd = "cd " + vis_dir + " && python3 " + viz_script_grouped +
                        " " + zero_mlir;
      int rc = std::system(cmd.c_str());
      if (rc != 0) {
        LOG_WARNING << "MLIR grouped visualization failed (exit " << rc
                    << "): " << cmd;
      }
    } catch (const std::exception &e) {
      LOG_WARNING << "MLIR grouped visualization threw: " << e.what();
    } catch (...) {
      LOG_WARNING << "MLIR grouped visualization threw an unknown exception.";
    }
  } else {
    LOG_WARNING << "MLIR grouped visualization script not found: "
                << viz_script_grouped;
  }

  std::string viz_script_grouped_no_slot0 =
      vesyla::util::SysPath::prog_dir() + "scripts/script_grouped_no_slot0.py";
  if (std::filesystem::exists(viz_script_grouped_no_slot0)) {
    try {
      std::string cmd = "cd " + vis_dir + " && python3 " +
                        viz_script_grouped_no_slot0 + " " + zero_mlir;
      int rc = std::system(cmd.c_str());
      if (rc != 0) {
        LOG_WARNING << "MLIR grouped (no slot0) visualization failed (exit "
                    << rc << "): " << cmd;
      }
    } catch (const std::exception &e) {
      LOG_WARNING << "MLIR grouped (no slot0) visualization threw: "
                  << e.what();
    } catch (...) {
      LOG_WARNING << "MLIR grouped (no slot0) visualization threw an unknown "
                     "exception.";
    }
  } else {
    LOG_WARNING << "MLIR grouped (no slot0) visualization script not found: "
                << viz_script_grouped_no_slot0;
  }

  std::string viz_script =
      vesyla::util::SysPath::prog_dir() + "scripts/script.py";
  if (std::filesystem::exists(viz_script)) {
    std::string cmd =
        "cd " + vis_dir + " && python3 " + viz_script + " " + zero_mlir;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
      LOG_WARNING << "MLIR visualization failed (exit " << rc << "): " << cmd;
    }
  } else {
    LOG_WARNING << "MLIR visualization script not found: " << viz_script;
  }

  pm.addPass(vesyla::pasm::createAddSlotPortPass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createAddSlotPortPass failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/1.mlir");
  pm.addPass(vesyla::pasm::createAddDefaultValuePass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createAddDefaultValuePass failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/2.mlir");

  std::string mzn_dir =
      std::filesystem::absolute(output_dir + "/debug/minizinc").string();
  std::filesystem::create_directories(mzn_dir);
  std::string temp_dir = mzn_dir + "/";
  pm.addPass(vesyla::pasm::createScheduleEpochPass(
      {VESYLA_SUITE_PATH_COMPONENTS, temp_dir, allow_unsafe,
       keep_debug_files}));
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createScheduleEpochPass failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/3.mlir");

  pm.addPass(vesyla::pasm::createReplaceLoopOp());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createReplaceLoopOp failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/4.mlir");

  pm.addPass(vesyla::pasm::createMergeRawOp());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createMergeRawOp failed.\n";
    std::exit(EXIT_FAILURE);
  }
  pm.clear();
  save_mlir(module, module_debug_path + "/5.mlir");

  pm.addPass(vesyla::pasm::createAddHaltPass());
  if (mlir::failed(pm.run(module))) {
    LOG_FATAL << "Error: createAddHaltPass failed.\n";
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
