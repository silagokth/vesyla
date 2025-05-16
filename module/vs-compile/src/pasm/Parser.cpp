#include "pasm/Parser.hpp"

namespace vesyla {
namespace pasm {

mlir::ModuleOp *Parser::parse(std::string &input) {
  mlir::ModuleOp *module = new mlir::ModuleOp();

  // parse the input line by line
  std::istringstream iss(input);
  std::string line;
  while (std::getline(iss, line)) {
    // remove comments and whitespace
    size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    line = util::Common::remove_leading_and_trailing_white_space(line);

    // check if the line is empty
    if (line.empty()) {
      continue;
    }

    // parse the line
    parse_instr(line);
  }
}

void Parser::parse_instr(std::string &line) {

  // remove comments and whitespace
  std::string str = util::Common::remove_comments_and_whitespace(line);

  std::string kind;
  std::string id;
  std::unordered_map<std::string, std::string> params;

  // pattern to match the instruction type
  std::regex kind_pattern(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*)(.*)");

  std::smatch match;
  if (std::regex_match(str, match, kind_pattern)) {
    kind = match[1];
    str = remove_comments_and_whitespace(match[2]);
    std::regex id_pattern(R"(^<\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*))");
    if (std::regex_match(str, match, id_pattern)) {
      id = match[1];
      str = remove_comments_and_whitespace(match[2]);
    } else {
      id = gen_random_string(8);
    }
    std::regex param_pattern(R"(^\(\s*(.*)\s*)$)");
    if (std::regex_match(str, match, param_pattern)) {
      str = match[1];
      // split the parameters by comma
      std::vector<std::string> param_list;
      str.split(',', std::back_inserter(param_list));
      for (auto &param : param_list) {
        // split the parameter by '='
        std::regex param_regex(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*))");
        if (std::regex_match(param, match, param_regex)) {
          std::string key = match[1];
          std::string value = match[2];
          params[key] = value;
        } else {
          llvm::errs() << "Error: Invalid parameter format: " << param << "\n";
          return;
        }
      }
    }
  } else {
    llvm::errs() << "Error: Invalid instruction type: " << line << "\n";
    return;
  }

  // create the instruction operation
  auto instr_op =
      mlir::OpBuilder(module->getContext())
          .create<vesyla::pasm::InstrOp>(
              module->getLoc(), id, kind,
              mlir::DictionaryAttr::get(module->getContext(), params));

  module->push_back(instr_op);
}

} // namespace pasm
} // namespace vesyla