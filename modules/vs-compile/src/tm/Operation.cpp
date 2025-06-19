#include "Operation.hpp"
#include <string>

namespace vesyla {
namespace tm {

Operation::Operation(string expr_str) {
  // it has to start with "operation"
  // followd by the name of the operation
  // followed by the expression
  // e.g. operation op0 T<3>(e1, e2)

  // remove the leading and trailing spaces
  const char *WhiteSpace = " \t\v\r\n";
  std::size_t start = expr_str.find_first_not_of(WhiteSpace);
  std::size_t end = expr_str.find_last_not_of(WhiteSpace);
  expr_str =
      start == end ? std::string() : expr_str.substr(start, end - start + 1);

  string pattern = "^operation\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s+(.*)$";
  std::smatch match;
  if (std::regex_match(expr_str, match, std::regex(pattern))) {
    name = match[1];
    expr = OperationExpr(match[2]);
  } else {
    LOG_FATAL << "Invalid operation string: " << expr_str;
    std::exit(-1);
  }
}

OperationExpr::OperationExpr(string str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));

  // remove all white spaces
  str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());

  // check if the str is empty
  if (str.empty()) {
    LOG_FATAL << "Operator string is empty!" << str;
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
    LOG_FATAL << "Invalid operator string: " << str;
    std::exit(-1);
  }

  if (kind == EVENT) {
    // use regex to match e<id>
    // e.g. e1
    // e.g. e1 -> id = 1
    auto regex = std::regex("^e([0-9]+)$");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["id"] = match[1];
    } else {
      LOG_FATAL << "Invalid event string: " << str;
      std::exit(-1);
    }
  } else if (kind == REPEAT) {
    // use regex to match R<iter, delay>(expr)
    // e.g. R<2, 3>(e1)
    // e.g. R<2, 3>(e1) -> iter = 2, delay = 3, expr = e1
    auto regex = std::regex("^R<([0-9a-zA-Z_]+),([0-9a-zA-Z_]+)>\\((.*)\\)$");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["iter"] = match[1];
      parameters["delay"] = match[2];
      children.push_back(OperationExpr(match[3]));
    } else {
      LOG_FATAL << "Invalid repeat operator string: " << str;
      std::exit(-1);
    }
  } else if (kind == TRANSIT) {
    // use regex to match T<delay>(expr_0, expr_1)
    // e.g. T<3>(e1, e2)
    // e.g. T<3>(e1, e2) -> delay = 3, expr_0 = e1, expr_1 = e2
    auto regex = std::regex("^T<([0-9a-zA-Z_]+)>\\((.*)\\)$");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["delay"] = match[1];
      string inner_str = match[2];
      // split the inner string by comma considering the brackets: {}, [], ()
      int bracket_curly = 0;
      int bracket_square = 0;
      int bracket_round = 0;
      string inner_str_temp = "";
      for (size_t i = 0; i < inner_str.size(); i++) {
        if (inner_str[i] == '(') {
          bracket_round++;
        } else if (inner_str[i] == ')') {
          bracket_round--;
        } else if (inner_str[i] == '{') {
          bracket_curly++;
        } else if (inner_str[i] == '}') {
          bracket_curly--;
        } else if (inner_str[i] == '[') {
          bracket_square++;
        } else if (inner_str[i] == ']') {
          bracket_square--;
        } else if (inner_str[i] == ',' && bracket_curly == 0 &&
                   bracket_square == 0 && bracket_round == 0) {
          children.push_back(OperationExpr(inner_str_temp));
          inner_str_temp = "";
          continue;
        }
        inner_str_temp += inner_str[i];
      }
      if (inner_str_temp.size() > 0) {
        children.push_back(OperationExpr(inner_str_temp));
      }
      if (bracket_curly != 0 || bracket_square != 0 || bracket_round != 0) {
        LOG_FATAL << "Invalid transit operator string: " << str;
        std::exit(-1);
      }
      if (children.size() != 2) {
        LOG_FATAL << "Invalid transit operator string: " << str;
        std::exit(-1);
      }
    } else {
      LOG_FATAL << "Invalid transit operator string: " << str;
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

std::vector<std::pair<std::string, std::vector<int>>>
OperationExpr::get_all_anchors() {
  std::vector<std::pair<std::string, std::vector<int>>> anchors;
  if (kind == TRANSIT) {
    anchors = children[0].get_all_anchors();
    anchors.insert(anchors.end(), children[1].get_all_anchors().begin(),
                   children[1].get_all_anchors().end());
  } else if (kind == REPEAT) {
    std::vector<std::pair<std::string, std::vector<int>>> child_anchors =
        children[0].get_all_anchors();
    int iter = std::stoi(parameters["iter"]);
    for (int i = 0; i < iter; i++) {
      for (int j = 0; j < child_anchors.size(); j++) {
        std::string anchor_event = child_anchors[j].first;
        std::vector<int> indices = child_anchors[j].second;
        indices.push_back(i);
        anchors.push_back({anchor_event, indices});
      }
    }
  } else if (kind == EVENT) {
    anchors.push_back({"e" + parameters["id"], {}});
  }
  return anchors;
}

std::vector<std::string> Operation::get_all_anchors() {
  std::vector<std::string> anchors;
  for (const auto &anchor : expr.get_all_anchors()) {
    std::string anchor_event = anchor.first;
    std::vector<int> indices = anchor.second;
    std::string anchor_str = anchor_event;
    for (int i = indices.size() - 1; i >= 0; --i) {
      int index = indices[i];
      anchor_str += "[" + std::to_string(index) + "]";
    }
    anchors.push_back(anchor_str);
  }
  return anchors;
}

} // namespace tm
} // namespace vesyla