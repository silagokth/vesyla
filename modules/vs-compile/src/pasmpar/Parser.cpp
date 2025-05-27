#include "pasmpar/Parser.hpp"

extern FILE *yyin;

namespace vesyla {
namespace pasmpar {

extern mlir::ModuleOp *module;
extern mlir::Operation *temp_epoch_op;

void Parser::parse(std::string &filename_, mlir::ModuleOp *module_) {
  module = module_;

  // Get a file handle to a file
  FILE *inputfile;
  inputfile = fopen(filename_.c_str(), "r");

  // Make sure the file is valid
  if (!inputfile) {
    cout << "File not found: " << filename_ << endl;
  }

  strncpy(global_input_file_name, filename_.c_str(), 80);

  // Set lex to read from the file
  yyin = inputfile;

  // before parsing, make a temp epoch to hold the created operations
  mlir::OpBuilder builder(module->getBodyRegion());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(&module->getBodyRegion().front());
  auto epoch_op = builder.create<vesyla::pasm::EpochOp>(
      loc, builder.getStringAttr("__temp__"));
  // add a block to the epoch operation

  mlir::Region &epochBodyRegion = epoch_op.getBody();
  mlir::Block *block;
  if (epochBodyRegion.empty()) {
    block = builder.createBlock(&epochBodyRegion);
  } else {
    block = &epochBodyRegion.front();
  }
  builder.setInsertionPointToEnd(block);
  temp_epoch_op = epoch_op.getOperation();

  // Parsing
  yyparse();

  // remove the temp epoch operation
  if (temp_epoch_op) {
    temp_epoch_op->erase();
    temp_epoch_op = nullptr;
  }

  fclose(inputfile);
}

} // namespace pasmpar
} // namespace vesyla