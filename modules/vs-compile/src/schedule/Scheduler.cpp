#include "Scheduler.hpp"

namespace vesyla {
namespace schedule {
void Scheduler::run(mlir::ModuleOp &module) {

  // get the environment variable: VESYLA_SUITE_PATH_COMPONENTS
  std::string VESYLA_SUITE_PATH_COMPONENTS =
      std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          ? std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          : "";
  if (VESYLA_SUITE_PATH_COMPONENTS == "") {
    LOG(FATAL) << "Error: VESYLA_SUITE_PATH_COMPONENTS is not set.\n";
    std::exit(-1);
  }

  // get the environment variable: VESYLA_SUITE_PATH_TMP
  std::string VESYLA_SUITE_PATH_TMP = std::getenv("VESYLA_SUITE_PATH_TMP")
                                          ? std::getenv("VESYLA_SUITE_PATH_TMP")
                                          : "";
  if (VESYLA_SUITE_PATH_TMP == "") {
    LOG(FATAL) << "Error: VESYLA_SUITE_PATH_TMP is not set.\n";
    std::exit(-1);
  }
  // create the directory if it does not exist
  if (mkdir(VESYLA_SUITE_PATH_TMP.c_str(), 0777) == -1) {
    if (errno != EEXIST) {
      LOG(FATAL) << "Error: Failed to create directory: "
                 << VESYLA_SUITE_PATH_TMP << "\n";
      std::exit(-1);
    }
  }

  // Read the architecture json file and create the json object
  std::ifstream ifs(arch_file.c_str());
  if (!ifs.is_open()) {
    LOG(FATAL) << "Error: Failed to open architecture file: " << arch_file;
    std::exit(-1);
  }
  nlohmann::json arch_json = nlohmann::json::parse(ifs);
  ifs.close();

  nlohmann::json component_map_json;
  for (const auto &cell : arch_json["cells"].items()) {
    int row = cell.value()["coordinates"]["row"];
    int col = cell.value()["coordinates"]["col"];
    std::string controller_kind = cell.value()["cell"]["controller"]["kind"];
    std::string key = std::to_string(row) + "_" + std::to_string(col);
    component_map_json[key] = controller_kind;

    for (const auto &resource :
         cell.value()["cell"]["resources_list"].items()) {
      std::string resource_kind = resource.value()["kind"];
      int slot_start = resource.value()["slot"];
      for (auto i = 0; i < resource.value()["size"]; i++) {
        int slot = slot_start + i;
        for (auto j = 0; j < 4; j++) {
          int port = j;
          std::string key = std::to_string(row) + "_" + std::to_string(col) +
                            "_" + std::to_string(slot) + "_" +
                            std::to_string(port);
          component_map_json[key] = resource_kind;
        }
      }
    }
  }

  int row = arch_json["parameters"]["ROWS"];
  int col = arch_json["parameters"]["COLS"];

  // Create a PassManager
  mlir::PassManager pm(module.getContext());

  // Add passes to the pipeline
  pm.addPass(vesyla::pasm::createAddSlotPortPass());
  pm.addPass(vesyla::pasm::createReplaceEpochOp(
      {.component_map = component_map_json.dump(),
       .component_path = VESYLA_SUITE_PATH_COMPONENTS,
       .tmp_path = VESYLA_SUITE_PATH_TMP,
       .row = row,
       .col = col}));
  pm.addPass(vesyla::pasm::createReplaceLoopOp());
  pm.addPass(vesyla::pasm::createMergeRawOp());
  pm.addPass(vesyla::pasm::createAddHaltPass());

  // Run the pass pipeline
  if (mlir::failed(pm.run(module))) {
    LOG(FATAL) << "Error: Pass pipeline failed.\n";
    std::exit(-1);
  }

  // Print the transformed module to stdout
  module->print(llvm::outs());

  // Save the transformed module to a file
  std::string output_filename = "output";
  std::string output_dir = ".";
  vesyla::pasm::CodeGen cg;
  cg.generate(module, output_dir, output_filename);
  llvm::outs() << "Generated code in " << output_dir << "/" << output_filename
               << "\n";
}
} // namespace schedule
} // namespace vesyla
