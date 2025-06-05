#include "bison_util.hpp"

namespace vesyla {
namespace schedule {
mlir::ModuleOp *module;
mlir::Operation *temp_epoch_op;

/*! This function reports the error according to the current file, source line,
 * and the given message. */
void print_error(const char *message) {
  LOG_DEBUG << "Error in \"" << global_input_file_name << "\" "
            << "around line (" << global_source_line << "): " << message;
}

void print_grammar(const std::string &grammar_) {
  LOG_DEBUG << "BISON: "
            << "source line: " + std::to_string(global_source_line) + ", "
            << "grammar: " << grammar_;
}

} // namespace schedule
} // namespace vesyla
