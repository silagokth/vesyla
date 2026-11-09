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
    // use regex to match R<iter, delay>(expr) or, with slopes (AGU Slope
    // Levels), R<iter, delay, slope, follows[, dslope, dfollows]>(expr): the
    // trip count is iter + slope * (index of the repeat `follows` repeats up),
    // the delay after pass c is delay + dslope * (index of the repeat
    // `dfollows` repeats up, 0 being c itself). dslope may be a variable.
    // e.g. R<2, 3>(e1) -> iter = 2, delay = 3, expr = e1
    auto regex = std::regex(
        "^R<([0-9a-zA-Z_]+),([0-9a-zA-Z_]+)(?:,(-?[0-9]+),([0-9]+)"
        "(?:,(-?[0-9a-zA-Z_]+),([0-9]+))?)?>\\((.*)\\)$");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      parameters["iter"] = match[1];
      parameters["delay"] = match[2];
      if (match[3].matched) {
        parameters["slope"] = match[3];
        parameters["follows"] = match[4];
      }
      if (match[5].matched) {
        parameters["dslope"] = match[5];
        parameters["dfollows"] = match[6];
      }
      children.push_back(OperationExpr(match[7]));
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
    str = "R<" + parameters["iter"] + ", " + parameters["delay"];
    if (parameters.count("slope")) {
      str += ", " + parameters["slope"] + ", " + parameters["follows"];
    }
    if (parameters.count("dslope")) {
      str += ", " + parameters["dslope"] + ", " + parameters["dfollows"];
    }
    str += ">(";
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

// A repeat is classified as outer (OR) when a transition lies between it and
// the event, otherwise inner (IR). A path with no transition puts every
// repeat in IR (the IR-default rule). A repeat with a slope repeats its child
// iter + slope * (index of the repeat `follows` repeats up) times.
struct PartialAnchor {
  int event_id;
  std::vector<int> or_idx;
  std::vector<int> ir_idx;
};

struct Ancestor {
  const OperationExpr *repeat;
  int index;
};

int64_t trips_of(const OperationExpr &e, const std::vector<Ancestor> &anc) {
  int64_t iter = std::stoll(e.parameters.at("iter"));
  auto slope_it = e.parameters.find("slope");
  if (slope_it == e.parameters.end()) {
    return iter;
  }
  int64_t slope = std::stoll(slope_it->second);
  size_t follows = std::stoull(e.parameters.at("follows"));
  if (follows == 0 || follows > anc.size()) {
    LOG_FATAL << "A repeat follows a repeat it does not have: " << follows
              << " up of " << anc.size();
    std::exit(EXIT_FAILURE);
  }
  return iter + slope * anc[anc.size() - follows].index;
}

// `or_boundary`: the ancestors before it are above a transition (OR).
void collect_anchors(const OperationExpr &e, std::vector<Ancestor> &anc,
                     size_t or_boundary, std::vector<PartialAnchor> &out) {
  if (e.kind == OperationExpr::EVENT) {
    PartialAnchor p;
    p.event_id = std::stoi(e.parameters.at("id"));
    for (size_t i = 0; i < anc.size(); ++i) {
      (i < or_boundary ? p.or_idx : p.ir_idx).push_back(anc[i].index);
    }
    out.push_back(std::move(p));
  } else if (e.kind == OperationExpr::TRANSIT) {
    for (const auto &child : e.children) {
      collect_anchors(child, anc, anc.size(), out);
    }
  } else if (e.kind == OperationExpr::REPEAT) {
    int64_t trips = trips_of(e, anc);
    for (int64_t i = 0; i < trips; i++) {
      anc.push_back({&e, static_cast<int>(i)});
      collect_anchors(e.children[0], anc, or_boundary, out);
      anc.pop_back();
    }
  }
}

} // namespace

std::vector<::vesyla::Anchor> OperationExpr::get_all_anchors() {
  std::vector<::vesyla::Anchor> anchors;
  std::vector<PartialAnchor> partial;
  std::vector<Ancestor> anc;
  collect_anchors(*this, anc, 0, partial);
  for (const PartialAnchor &p : partial) {
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
