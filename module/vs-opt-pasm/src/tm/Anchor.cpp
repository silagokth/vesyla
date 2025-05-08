#include "Anchor.hpp"

namespace vesyla {
namespace tm {

AnchorExpr::AnchorExpr(string str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));

  // check if the str is empty
  if (str.empty()) {
    LOG(FATAL) << "Anchor string is empty!" << str;
    std::exit(-1);
  }

  // use regex to match <op_name>.<event_id>[<index_0>][<index_1>]...
  auto regex =
      std::regex("([a-zA-Z_][a-zA-Z0-9_]*)\\.e([0-9]+)(\\[([0-9]+)\\])*");
  std::smatch match;
  if (std::regex_match(str, match, regex)) {
    op_name = match[1];
    event_id = std::stoi(match[2]);
    for (size_t i = 3; i < match.size(); i += 2) {
      if (match[i].length() > 0) {
        indices.push_back(std::stoi(match[i + 1]));
      }
    }
  } else {
    LOG(FATAL) << "Invalid anchor string: " << str;
    std::exit(-1);
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