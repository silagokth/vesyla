#include "ScheduleEpochPassDetail.hpp"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace vesyla::pasm::schedule_epoch_detail {

void ScheduleEpochPassRewriter::reshape_instr(
    ::vesyla::pasm::EpochOp &op, ::mlir::PatternRewriter &rewriter) const {
  // Get the block to insert the new operations
  ::mlir::Block *block = getEpochBodyEntryBlock(op, rewriter);

  std::vector<::mlir::Operation *> ops_to_erase;
  std::vector<::mlir::Operation *> ops_to_reshape;

  for (::mlir::Operation &child_op : *block) {
    ops_to_reshape.push_back(&child_op);
  }

  for (::mlir::Operation *child_op : ops_to_reshape) {
    if (auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(child_op)) {
      // convert the RopOp to JSON
      nlohmann::json rop_json = op2json(child_op);
      std::string random_str = ::vesyla::util::Common::gen_random_string(10);
      LOG_DEBUG << "Random string for temporary file: " << random_str;
      std::string input_filename = tmp_path + "/" + random_str + ".json";
      std::string output_filename = tmp_path + "/" + random_str + "_out.json";
      LOG_DEBUG << "Input filename: " << input_filename;
      LOG_DEBUG << "Output filename: " << output_filename;
      std::string label = std::to_string(rop_json["row"].get<int>()) + "_" +
                          std::to_string(rop_json["col"].get<int>()) + "_" +
                          std::to_string(rop_json["slot"].get<int>());
      LOG_DEBUG << "Label: " << label;
      if (component_map.find(label) == component_map.end()) {
        llvm::outs() << "Error: Cannot find the component : " << label << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::string command = component_path + "/resources/" +
                            component_map[label].get<std::string>() +
                            "/compile_util reshape_instr " + input_filename +
                            " " + output_filename;

      llvm::outs() << "Executing command: " << command << "\n";

      std::ofstream file(input_filename);
      if (!file.is_open()) {
        llvm::outs() << "Error: Failed to create temporary file.\n";
        std::exit(EXIT_FAILURE);
      }
      file << rop_json.dump(4);
      file.close();
      int result = system(command.c_str());
      if (result != 0) {
        llvm::outs() << "Error: Command failed with error code: " << result
                     << "\n";
        std::exit(EXIT_FAILURE);
      }

      std::ifstream output_file(output_filename);
      if (!output_file.is_open()) {
        llvm::outs() << "Error: Failed to open output file.\n";
        std::exit(EXIT_FAILURE);
      }
      nlohmann::json output_json = nlohmann::json::parse(output_file);
      output_file.close();
      std::filesystem::remove(input_filename);
      std::filesystem::remove(output_filename);
      if (output_json["kind"].get<std::string>() != "rop") {
        llvm::outs() << "Error: Output JSON is not a RopOp.\n";
        std::exit(EXIT_FAILURE);
      }
      rewriter.setInsertionPointToEnd(block);
      json2op(output_json, rewriter);
      ops_to_erase.push_back(child_op);
    } else if (auto cop_op = llvm::dyn_cast<::vesyla::pasm::CopOp>(child_op)) {
      nlohmann::json cop_json = op2json(child_op);
      std::string random_str = ::vesyla::util::Common::gen_random_string(10);
      LOG_DEBUG << "Random string for temporary file: " << random_str;
      std::string input_filename = tmp_path + "/" + random_str + ".json";
      std::string output_filename = tmp_path + "/" + random_str + "_out.json";
      LOG_DEBUG << "Input filename: " << input_filename;
      LOG_DEBUG << "Output filename: " << output_filename;
      std::string label = std::to_string(cop_json["row"].get<int>()) + "_" +
                          std::to_string(cop_json["col"].get<int>());
      LOG_DEBUG << "Label: " << label;
      if (component_map.find(label) == component_map.end()) {
        llvm::outs() << "Error: Cannot find the component : " << label << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::string command = component_path + "/resources/" +
                            component_map[label].get<std::string>() +
                            "/compile_util reshape_instr " + input_filename +
                            " " + output_filename;

      llvm::outs() << "Executing command: " << command << "\n";

      std::ofstream file(input_filename);
      if (!file.is_open()) {
        llvm::outs() << "Error: Failed to create temporary file.\n";
        std::exit(EXIT_FAILURE);
      }
      file << cop_json.dump(4);
      file.close();
      int result = system(command.c_str());
      if (result != 0) {
        llvm::outs() << "Error: Command failed with error code: " << result
                     << "\n";
        std::exit(EXIT_FAILURE);
      }

      std::ifstream output_file(output_filename);
      if (!output_file.is_open()) {
        llvm::outs() << "Error: Failed to open output file.\n";
        std::exit(EXIT_FAILURE);
      }
      nlohmann::json output_json = nlohmann::json::parse(output_file);
      output_file.close();
      std::filesystem::remove(input_filename);
      std::filesystem::remove(output_filename);
      if (output_json["kind"].get<std::string>() != "cop") {
        llvm::outs() << "Error: Output JSON is not a CopOp.\n";
        std::exit(EXIT_FAILURE);
      }
      rewriter.setInsertionPointToEnd(block);
      json2op(output_json, rewriter);
      ops_to_erase.push_back(child_op);
    } else if (auto raw_op =
                   llvm::dyn_cast<::vesyla::pasm::RawOp>(child_op)) {
      // DO NOTHING
    } else if (auto cstr_op =
                   llvm::dyn_cast<::vesyla::pasm::CstrOp>(child_op)) {
      // DO NOTHING
    } else if (auto yield_op =
                   llvm::dyn_cast<::vesyla::pasm::YieldOp>(child_op)) {
      // DO NOTHING
    } else {
      llvm::outs() << "Illegal operation type in EpochOp: "
                   << child_op->getName() << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  for (auto op_to_erase : ops_to_erase) {
    rewriter.eraseOp(op_to_erase);
  }
}

} // namespace vesyla::pasm::schedule_epoch_detail
