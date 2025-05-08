#include "Operation.hpp"

namespace vesyla {
namespace tm {

OperationExpr::OperationExpr(string str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));

  // check if the str is empty
  if (str.empty()) {
    LOG(FATAL) << "Operator string is empty!" << str;
    std::exit(-1);
  }

  // distinguish the operator type by looking at the first character
  if (str[0] == 'e') {
    kind = EVENT;
  } else if (str[0] == 'R') {
    kind = REPEAT;
  } else if (str[0] == 'T') {
    kind = TRANSIT;
  } else {
    LOG(FATAL) << "Invalid operator string: " << str;
    std::exit(-1);
  }

  if (kind == EVENT) {
    // use regex to match e<id>
    // e.g. e1
    // e.g. e1 -> id = 1
    auto regex = std::regex("e([0-9]+)");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["id"] = match[1];
    } else {
      LOG(FATAL) << "Invalid event string: " << str;
      std::exit(-1);
    }
  } else if (kind == REPEAT) {
    // use regex to match R<iter, delay>(expr)
    // e.g. R<2, 3>(e1)
    // e.g. R<2, 3>(e1) -> iter = 2, delay = 3, expr = e1
    auto regex = std::regex("R<([0-9a-zA-Z_]+), ([0-9a-zA-Z_]+)>\\((.*)\\)");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["iter"] = match[1];
      parameters["delay"] = match[2];
      children.push_back(OperationExpr(match[3]));
    } else {
      LOG(FATAL) << "Invalid repeat operator string: " << str;
      std::exit(-1);
    }
  } else if (kind == TRANSIT) {
    // use regex to match T<delay>(expr_0, expr_1)
    // e.g. T<3>(e1, e2)
    // e.g. T<3>(e1, e2) -> delay = 3, expr_0 = e1, expr_1 = e2
    auto regex = std::regex("T<([0-9a-zA-Z_]+)>\\((.*), (.*)\\)");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["delay"] = match[1];
      children.push_back(OperationExpr(match[2]));
      children.push_back(OperationExpr(match[3]));
    } else {
      LOG(FATAL) << "Invalid transit operator string: " << str;
      std::exit(-1);
    }
  }
}

OperationExpr::~OperationExpr() {
  // destructor
  // do nothing
}

string OperationExpr::to_string() {
  string str;
  if (kind == EVENT) {
    str = "e" + parameters["id"];
  } else if (kind == REPEAT) {
    str = "R<" + parameters["iter"] + ", " + parameters["delay"] + ">(";
    for (auto &child : children) {
      str += child.to_string() + ", ";
    }
    str = str.substr(0, str.size() - 2) + ")";
  } else if (kind == TRANSIT) {
    str = "T<" + parameters["delay"] + ">(";
    for (auto &child : children) {
      str += child.to_string() + ", ";
    }
    str = str.substr(0, str.size() - 2) + ")";
  }
  return str;
}

string Operation::to_string() {
  string str = "operation " + name + " " + expr.to_string();
  return str;
}

} // namespace tm
} // namespace vesyla