#include "vesyla/Analysis/TimingModel/Operation.hpp"
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
    std::exit(EXIT_FAILURE);
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
    std::exit(EXIT_FAILURE);
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
    std::exit(EXIT_FAILURE);
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
      std::exit(EXIT_FAILURE);
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
      std::exit(EXIT_FAILURE);
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
      // split the inner string by comma considering the brackets: {}, [], (),
      // <>
      int bracket_curly = 0;
      int bracket_square = 0;
      int bracket_round = 0;
      int bracket_angle = 0;
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
        } else if (inner_str[i] == '<') {
          bracket_angle++;
        } else if (inner_str[i] == '>') {
          bracket_angle--;
        } else if (inner_str[i] == ',' && bracket_curly == 0 &&
                   bracket_square == 0 && bracket_round == 0 &&
                   bracket_angle == 0) {
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
        std::exit(EXIT_FAILURE);
      }
      if (children.size() != 2) {
        LOG_FATAL << "Invalid transit operator string: " << str;
        std::exit(EXIT_FAILURE);
      }
    } else {
      LOG_FATAL << "Invalid transit operator string: " << str;
      std::exit(EXIT_FAILURE);
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

namespace {

// While walking up from an event, a repeat is classified as outer (OR) once a
// transition has been seen below it on the path, otherwise inner (IR). A path
// with no transition puts every repeat in IR (the IR-default rule).
struct PartialAnchor {
  int event_id;
  std::vector<int> or_idx;
  std::vector<int> ir_idx;
  bool transit_seen;
};

std::vector<PartialAnchor> collect_anchors(const OperationExpr &e) {
  std::vector<PartialAnchor> result;
  if (e.kind == OperationExpr::EVENT) {
    PartialAnchor p;
    p.event_id = std::stoi(e.parameters.at("id"));
    p.transit_seen = false;
    result.push_back(std::move(p));
  } else if (e.kind == OperationExpr::TRANSIT) {
    for (const auto &child : e.children) {
      for (auto &p : collect_anchors(child)) {
        p.transit_seen = true;
        result.push_back(std::move(p));
      }
    }
  } else if (e.kind == OperationExpr::REPEAT) {
    std::vector<PartialAnchor> child_anchors = collect_anchors(e.children[0]);
    int iter = std::stoi(e.parameters.at("iter"));
    for (int i = 0; i < iter; i++) {
      for (const PartialAnchor &cp : child_anchors) {
        PartialAnchor p = cp;
        // Insert at the front so each field ends up outer-index-first.
        if (p.transit_seen) {
          p.or_idx.insert(p.or_idx.begin(), i);
        } else {
          p.ir_idx.insert(p.ir_idx.begin(), i);
        }
        result.push_back(std::move(p));
      }
    }
  }
  return result;
}

} // namespace

std::vector<::vesyla::Anchor> OperationExpr::get_all_anchors() {
  std::vector<::vesyla::Anchor> anchors;
  for (const PartialAnchor &p : collect_anchors(*this)) {
    ::vesyla::Anchor a;
    a.mt_idx = p.event_id;
    a.or_idx = p.or_idx;
    a.ir_idx = p.ir_idx;
    anchors.push_back(std::move(a));
  }
  return anchors;
}

std::vector<std::string> Operation::get_all_anchors() {
  std::vector<std::string> anchors;
  for (::vesyla::Anchor anchor : expr.get_all_anchors()) {
    anchor.name = name;
    anchors.push_back(anchor.to_string());
  }
  return anchors;
}

} // namespace tm
} // namespace vesyla
