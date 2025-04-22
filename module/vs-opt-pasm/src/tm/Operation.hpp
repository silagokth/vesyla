#ifndef __VESYLA_TM_OPERATION_HPP__
#define __VESYLA_TM_OPERATION_HPP__

#include "util/Common.hpp"
#include <regex>
#include <string>

using namespace std;

namespace vesyla {
namespace tm {

struct TransitOperator;
struct RepeatOperator;

struct Expression {
  Expression(string str) {
    // Never use this constructor
    BOOST_LOG_TRIVIAL(fatal) << "Expression constructor should never be used";
    std::exit(-1);
  };
  virtual ~Expression() {}
  virtual string to_string() = 0;
};
struct Event : public Expression {
  int id;
  Event(string str) : Expression(str) {
    // trim the string
    str.erase(str.begin(),
              std::find_if(str.begin(), str.end(),
                           [](unsigned char ch) { return !std::isspace(ch); }));
    // use regex to match eid
    // e.g. e1
    // e.g. e1 -> 1
    auto regex = std::regex("e([0-9]+)");
    std::smatch match;
    if (std::regex_match(str, match, regex)) {
      id = std::stoi(match[1]);
    } else {
      BOOST_LOG_TRIVIAL(fatal) << "Invalid event string: " << str;
      std::exit(-1);
    }
  }
  ~Event() {
    // do nothing
  };
  string to_string() override { return "e" + std::to_string(id); };
};
struct RepeatOperator : public Expression {
  string delay;
  int iter;
  Expression *expr;
  RepeatOperator(string str);
  ~RepeatOperator() { delete expr; };
  string to_string() override {
    return "R<" + std::to_string(iter) + ", " + delay + ">(" +
           expr->to_string() + ")";
  };
};
struct TransitOperator : public Expression {
  string delay;
  Expression *expr_0;
  Expression *expr_1;
  TransitOperator(string str);
  ~TransitOperator() {
    delete expr_0;
    delete expr_1;
  };
  string to_string() override {
    return "T<" + delay + ">(" + expr_0->to_string() + ", " +
           expr_1->to_string() + ")";
  };
};

} // namespace tm
} // namespace vesyla

#endif // __VESYLA_TM_OPERATION_HPP__