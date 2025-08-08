#include "Anchor.hpp"

namespace vesyla {
namespace tm {

AnchorExpr::AnchorExpr(string str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));

  // remove all white spaces
  str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());

  // check if the str is empty
  if (str.empty()) {
    LOG_FATAL << "Anchor string is empty!" << str;
    std::exit(EXIT_FAILURE);
  }

  // use regex to match <op_name>.<event_id>[<index_0>][<index_1>]...
  auto regex =
      std::regex("^([a-zA-Z_][a-zA-Z0-9_]*)\\.e([0-9]+)(\\[([0-9]+)\\])*$");
  std::smatch match;
  if (std::regex_match(str, match, regex)) {
    op_name = match[1];
    event_id = std::stoi(match[2]);
    std::string indices_str = match[0];
    // remove the op_name and event_id from the string
    auto regex2 = std::regex("\\[([0-9]+)\\]");
    std::smatch match2;
    std::string::const_iterator search_start(indices_str.cbegin());
    while (
        std::regex_search(search_start, indices_str.cend(), match2, regex2)) {
      search_start = match2.suffix().first;
      indices.push_back(std::stoi(match2[1]));
    }
  } else {
    LOG_FATAL << "Invalid anchor string: " << str;
    std::exit(EXIT_FAILURE);
  }
}

AnchorExpr::~AnchorExpr() {}

string AnchorExpr::to_string() {
  string str = op_name + ".e" + std::to_string(event_id);
  for (size_t i = 0; i < indices.size(); i++) {
    str += "[" + std::to_string(indices[i]) + "]";
  }
  return str;
}

Anchor::Anchor(string expr_str) : expr(expr_str) {
  int event_id = expr.event_id;
  string op_name = expr.op_name;
  vector<int> indices = expr.indices;
  name = op_name + "_e" + std::to_string(event_id);
  for (size_t i = 0; i < indices.size(); i++) {
    name += "_" + std::to_string(indices[i]);
  }
}
Anchor::Anchor(AnchorExpr expr) : expr(expr) {
  int event_id = expr.event_id;
  string op_name = expr.op_name;
  vector<int> indices = expr.indices;
  name = op_name + "_e" + std::to_string(event_id);
  for (size_t i = 0; i < indices.size(); i++) {
    name += "_" + std::to_string(indices[i]);
  }
}
Anchor::~Anchor() {}
string Anchor::to_string() { return "anchor " + name + " " + expr.to_string(); }
} // namespace tm
} // namespace vesyla
