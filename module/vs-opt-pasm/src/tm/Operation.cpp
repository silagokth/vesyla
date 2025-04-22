#include "Operation.hpp"

namespace vesyla {
namespace tm {

RepeatOperator::RepeatOperator(string str) : Expression(str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));
  // use regex to match R<iter, delay>(expr)
  // e.g. R<2, 3>(e1)
  // e.g. R<2, 3>(e1) -> iter = 2, delay = 3, expr = e1
  auto regex = std::regex("R<([0-9a-zA-Z_]+), ([0-9a-zA-Z_]+)>\\((.*)\\)");
  std::smatch match;
  if (std::regex_match(str, match, regex)) {
    iter = std::stoi(match[1]);
    delay = match[2];
    if (match[3].str().find("e") == 0) {
      expr = new Event(match[3]);
    } else if (match[3].str().find("R") == 0) {
      expr = new RepeatOperator(match[3]);
    } else if (match[3].str().find("T") == 0) {
      expr = new TransitOperator(match[3]);
    } else {
      BOOST_LOG_TRIVIAL(fatal) << "Invalid repeat operator string: " << str;
      std::exit(-1);
    }
  } else {
    BOOST_LOG_TRIVIAL(fatal) << "Invalid repeat operator string: " << str;
    std::exit(-1);
  }
}

TransitOperator::TransitOperator(string str) : Expression(str) {
  // trim the string
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char ch) { return !std::isspace(ch); }));
  // use regex to match T<delay>(expr_0, expr_1)
  // e.g. T<3>(e1, e2)
  // e.g. T<3>(e1, e2) -> delay = 3, expr_0 = e1, expr_1 = e2
  auto regex = std::regex("T<([0-9a-zA-Z_]+)>\\((.*), (.*)\\)");
  std::smatch match;
  if (std::regex_match(str, match, regex)) {
    delay = match[1];
    if (match[2].str().find("e") == 0) {
      expr_0 = new Event(match[2]);
    } else if (match[2].str().find("R") == 0) {
      expr_0 = new RepeatOperator(match[2]);
    } else if (match[2].str().find("T") == 0) {
      expr_0 = new TransitOperator(match[2]);
    } else {
      BOOST_LOG_TRIVIAL(fatal) << "Invalid transit operator string: " << str;
      std::exit(-1);
    }
    if (match[3].str().find("e") == 0) {
      expr_1 = new Event(match[3]);
    } else if (match[3].str().find("R") == 0) {
      expr_1 = new RepeatOperator(match[3]);
    } else if (match[3].str().find("T") == 0) {
      expr_1 = new TransitOperator(match[3]);
    } else {
      BOOST_LOG_TRIVIAL(fatal) << "Invalid transit operator string: " << str;
      std::exit(-1);
    }
  } else {
    BOOST_LOG_TRIVIAL(fatal) << "Invalid transit operator string: " << str;
    std::exit(-1);
  }
}

} // namespace tm
} // namespace vesyla