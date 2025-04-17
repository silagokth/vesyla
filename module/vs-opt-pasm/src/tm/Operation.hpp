#ifndef __VESYLA_TM_OPERATION_HPP__
#define __VESYLA_TM_OPERATION_HPP__

#include "util/Common.hpp"
#include <regex>
#include <string>

using namespace std;

struct Expression {
  Expression(string str) {
    // Never use this constructor
    LOG(FATAL) << "Expression constructor should never be used";
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
      LOG(FATAL) << "Invalid event string: " << str;
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
  RepeatOperator(string str) : Expression(str) {
    // trim the string
    str.erase(str.begin(),
              std::find_if(str.begin(), str.end(),
                           [](unsigned char ch) { return !std::isspace(ch); }));
    // use regex to match R<iter, delay>(expr)
    // e.g. R<2, 3>(e1)
    // e.g. R<2, 3>(e1) -> iter = 2, delay = 3, expr = e1
    // the delay can be either an identifier or a number
    auto regex = std::regex("R<([0-9]+), ([0-9a-zA-Z_]+)>(.*)");
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
        LOG(FATAL) << "Invalid repeat operator string: " << str;
      }
    } else {
      LOG(FATAL) << "Invalid repeat operator string: " << str;
    }
  }
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
  TransitOperator(string str) : Expression(str) {
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
        LOG(FATAL) << "Invalid transit operator string: " << str;
      }
      if (match[3].str().find("e") == 0) {
        expr_1 = new Event(match[3]);
      } else if (match[3].str().find("R") == 0) {
        expr_1 = new RepeatOperator(match[3]);
      } else if (match[3].str().find("T") == 0) {
        expr_1 = new TransitOperator(match[3]);
      } else {
        LOG(FATAL) << "Invalid transit operator string: " << str;
      }
    } else {
      LOG(FATAL) << "Invalid transit operator string: " << str;
    }
  }
  ~TransitOperator() {
    delete expr_0;
    delete expr_1;
  };
  string to_string() override {
    return "T<" + delay + ">(" + expr_0->to_string() + ", " +
           expr_1->to_string() + ")";
  };
};

#endif // __VESYLA_TM_OPERATION_HPP__