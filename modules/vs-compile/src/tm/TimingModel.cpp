#include "tm/TimingModel.hpp"

namespace vesyla {
namespace tm {

TimingModel::TimingModel() {
  // Constructor implementation
}

TimingModel::~TimingModel() {
  // Destructor implementation
}

string TimingModel::to_string() {
  string str = "";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    str += "  " + it->second.to_string() + "\n";
  }
  for (auto it = anchors.begin(); it != anchors.end(); ++it) {
    str += "  " + it->second.to_string() + "\n";
  }
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    str += "  " + it->to_string() + "\n";
  }
  return str;
}

string TimingModel::to_mzn() {
  if (state != COMPILED) {
    LOG(WARNING) << "Timing model is not compiled. Compiling...";
    compile();
  }

  int max_latency = 1000;

  // build translation table for operations
  unordered_map<string, int> op2idx;
  unordered_map<int, string> idx2op;
  int idx = 0;
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    op2idx[it->second.name] = idx;
    idx2op[idx] = it->second.name;
    idx++;
  }
  int num_ops = operations.size();

  // generate minizinc model
  string str = "include \"globals.mzn\";\n";
  str += "int: MAX_LATENCY = " + std::to_string(max_latency) + ";\n";
  str += "int: NUM_OPS = " + std::to_string(num_ops) + ";\n";

  // add total latency variable
  str += "var 0..MAX_LATENCY: total_latency;\n";

  // add operations
  str += "array [0..NUM_OPS-1] of var 0..MAX_LATENCY: op_start_vec;\n";
  str += "array [0..NUM_OPS-1] of var 0..MAX_LATENCY: op_end_vec;\n";
  str += "constraint min(op_start_vec) == 0;\n";
  str += "constraint max(op_end_vec) == total_latency;\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    str += "var 0..MAX_LATENCY: " + it->second.name + ";\n";
    str += "constraint op_start_vec[" +
           std::to_string(op2idx[it->second.name]) + "] == " + it->second.name +
           ";\n";
    str += "constraint op_end_vec[" + std::to_string(op2idx[it->second.name]) +
           "] == " + it->second.name + " + " + it->second.duration_expr + ";\n";
  }

  // add variables
  for (auto it = variables.begin(); it != variables.end(); ++it) {
    str += "var 0..MAX_LATENCY: " + *it + ";\n";
  }

  // add anchors
  for (auto it = anchors.begin(); it != anchors.end(); ++it) {
    str += "var 0..MAX_LATENCY: " + it->second.name + ";\n";
    str += "constraint " + it->second.name + " == " + it->second.timing_expr +
           ";\n";
  }

  // add constraints
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    str += "constraint " + it->expr + ";\n";
  }

  // add objective
  str += "solve minimize total_latency;\n";

  return str;
}

bool is_number(string str) {
  // Check if the string is a valid number
  if (str.empty())
    return false;
  for (size_t i = 0; i < str.size(); ++i) {
    if (!isdigit(str[i]) && str[i] != '.' && str[i] != '-')
      return false;
  }
  return true;
}
bool is_identifier(string str) {
  // Check if the string is a valid identifier
  if (str.empty())
    return false;
  if (!isalpha(str[0]) && str[0] != '_')
    return false;
  for (size_t i = 1; i < str.size(); ++i) {
    if (!isalnum(str[i]) && str[i] != '_')
      return false;
  }
  return true;
}

void TimingModel::compile() {
  anchors.clear();
  variables.clear();

  // Recursively extract all variables from operation delay
  std::function<void(OperationExpr &)> extract_variables =
      [&](OperationExpr &expr) {
        if (expr.kind == OperationExpr::TRANSIT ||
            expr.kind == OperationExpr::REPEAT) {
          // Extract all variables from transit
          string delay_expr = expr.parameters["delay"];
          LOG(DEBUG) << "Delay expression: " << delay_expr;
          // Is it a number or a variable?
          if (is_number(delay_expr)) {
            // Is it a number, do nothing
          } else if (is_identifier(delay_expr)) {
            // Is it a variable
            variables.insert(delay_expr);
          } else {
            LOG(ERROR) << "Invalid delay expression: " << delay_expr;
          }
        }
        for (auto &child : expr.children) {
          extract_variables(child);
        }
      };
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    OperationExpr expr = it->second.expr;
    extract_variables(expr);
  }

  // Extract all anchors from constraints:
  // op_name.e<event_id>[<index_0>][<index_1>]...
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    if (it->kind != "linear") {
      LOG(FATAL) << "Invalid constraint kind: " << it->kind;
      exit(-1);
    }

    // Extract all anchors from constraint
    // e.g. op_name.e<event_id>
    // e.g. op_name.e<event_id>[<index_0>]
    // e.g. op_name.e<event_id>[<index_0>][<index_1>]...

    string pattern = "([a-zA-Z_][a-zA-Z0-9_]*\\.e[0-9]+)(\\s*\\[([0-9]+)\\])*";
    std::regex regex(pattern);
    std::smatch match;
    while (std::regex_search(it->expr, match, regex)) {
      Anchor anchor(match[0]);
      anchors[anchor.name] = anchor;

      // replace the event identifier in the expression with the anchor name
      string anchor_string_pattern = match[0];
      anchor_string_pattern =
          std::regex_replace(anchor_string_pattern, std::regex("\\."), "\\.");
      anchor_string_pattern =
          std::regex_replace(anchor_string_pattern, std::regex("\\["), "\\[");
      anchor_string_pattern =
          std::regex_replace(anchor_string_pattern, std::regex("\\]"), "\\]");
      it->expr = std::regex_replace(it->expr, std::regex(anchor_string_pattern),
                                    anchor.name);
    }
  }

  for (auto it = operations.begin(); it != operations.end(); ++it) {
    OperationExpr expr = it->second.expr;
    std::vector<string> anchors_in_op;
    for (auto it2 = anchors.begin(); it2 != anchors.end(); ++it2) {
      if (it2->second.expr.op_name == it->first) {
        anchors_in_op.push_back(it2->first);
      }
    }

    // Build binary tree for the operation expression
    BinaryTree *tree = build_binary_tree(expr);

    // calculate the duration of the operation
    tree->traverse_LRC([](BinaryTree *node) {
      BinaryTreeData *tree_data = (BinaryTreeData *)node->data;
      BinaryTree *left = node->left;
      BinaryTree *right = node->right;
      if (tree_data->expr.kind == OperationExpr::TRANSIT) {
        string left_duration =
            static_cast<BinaryTreeData *>(left->data)->duration;
        string right_duration =
            static_cast<BinaryTreeData *>(right->data)->duration;
        tree_data->duration = "(" + left_duration + "+" + right_duration +
                              "+(" + tree_data->expr.parameters["delay"] + "))";
      } else if (tree_data->expr.kind == OperationExpr::REPEAT) {
        string left_duration =
            static_cast<BinaryTreeData *>(left->data)->duration;
        int iter = std::stoi(tree_data->expr.parameters["iter"]);
        tree_data->duration = "(" + left_duration + "*" + std::to_string(iter) +
                              "+(" + tree_data->expr.parameters["delay"] +
                              ")*" + std::to_string(iter - 1) + ")";
      } else if (tree_data->expr.kind == OperationExpr::EVENT) {
        tree_data->duration = "(1)";
      }
    });
    // calculate the start time of the operation
    static_cast<BinaryTreeData *>(tree->data)->start = "(0)";
    tree->traverse_CLR([](BinaryTree *node) {
      BinaryTree *left = node->left;
      BinaryTree *right = node->right;
      BinaryTreeData *tree_data = (BinaryTreeData *)(node->data);
      if (left) {
        static_cast<BinaryTreeData *>(left->data)->start = tree_data->start;
      }
      if (right) {
        static_cast<BinaryTreeData *>(right->data)->start =
            "(" + tree_data->start + "+" +
            static_cast<BinaryTreeData *>(left->data)->duration + "+(" +
            tree_data->expr.parameters["delay"] + "))";
      }
    });

    unordered_map<BinaryTree *, BinaryTree *> node_parent_map;
    tree->traverse_CLR([&node_parent_map](BinaryTree *node) {
      BinaryTree *left = node->left;
      BinaryTree *right = node->right;
      if (left) {
        node_parent_map[static_cast<BinaryTree *>(left)] = node;
      }
      if (right) {
        node_parent_map[static_cast<BinaryTree *>(right)] = node;
      }
    });

    for (auto it2 = anchors_in_op.begin(); it2 != anchors_in_op.end(); ++it2) {
      string anchor_name = *it2;
      string op_name = it->first;
      Anchor &anchor = anchors[anchor_name];
      string event_id = std::to_string(anchor.expr.event_id);
      std::vector<BinaryTree *> r_op_stack;
      for (auto it3 = node_parent_map.begin(); it3 != node_parent_map.end();
           ++it3) {
        if (static_cast<BinaryTreeData *>(it3->first->data)->expr.kind ==
            OperationExpr::EVENT) {
          string event_id_1 = static_cast<BinaryTreeData *>(it3->first->data)
                                  ->expr.parameters["id"];
          if (event_id_1 == event_id) {
            // go through all its parents and find the repeat operation
            BinaryTree *parent = it3->second;

            while (parent) {
              if (static_cast<BinaryTreeData *>(parent->data)->expr.kind ==
                  OperationExpr::REPEAT) {
                r_op_stack.push_back(parent);
              }

              if (node_parent_map.find(parent) != node_parent_map.end()) {
                parent = node_parent_map[parent];
              } else {
                parent = nullptr;
              }
            }
            // reverse the stack
            std::reverse(r_op_stack.begin(), r_op_stack.end());
            vector<int> indices = anchor.expr.indices;

            if (r_op_stack.size() < indices.size()) {
              LOG(ERROR) << "r_op_stack size: " << r_op_stack.size();
              for (auto i = 0; i < r_op_stack.size(); ++i) {
                LOG(ERROR) << "r_op_stack[" << i << "]: "
                           << static_cast<BinaryTreeData *>(r_op_stack[i]->data)
                                  ->expr.to_string();
              }
              LOG(ERROR) << "indices size: " << indices.size();
              for (auto i = 0; i < indices.size(); ++i) {
                LOG(ERROR) << "indices[" << i << "]: " << indices[i];
              }
              LOG(FATAL) << "Too many indices!";
              std::exit(-1);
            }

            string expr_str =
                "(" + op_name + "+" +
                static_cast<BinaryTreeData *>(it3->first->data)->start;
            for (auto i = 0; i < indices.size(); ++i) {
              int index = indices[i];
              int iter =
                  std::stoi(static_cast<BinaryTreeData *>(r_op_stack[i]->data)
                                ->expr.parameters["iter"]);
              if (index >= iter) {
                LOG(FATAL) << "Index out of range: index(" << index
                           << ") >= iter(" << iter << ")";
                std::exit(-1);
              }
              expr_str =
                  expr_str + "+(" +
                  static_cast<BinaryTreeData *>(r_op_stack[i]->left->data)
                      ->duration +
                  "+(" +
                  static_cast<BinaryTreeData *>(r_op_stack[i]->data)
                      ->expr.parameters["delay"] +
                  "))*" + std::to_string(index);
            }
            expr_str += ")";
            anchor.timing_expr = expr_str;
          }
        }
      }
    }

    it->second.duration_expr =
        static_cast<BinaryTreeData *>(tree->data)->duration;

    // delete the tree
    delete tree;
  }

  // Compilation logic
  state = COMPILED;
}

BinaryTree *TimingModel::build_binary_tree(OperationExpr &expr) {
  BinaryTreeData *data = new BinaryTreeData();
  data->expr = expr;
  data->start = "";
  data->duration = "";
  BinaryTree *left = nullptr;
  BinaryTree *right = nullptr;
  if (expr.kind == OperationExpr::TRANSIT) {
    left = build_binary_tree(expr.children[0]);
    right = build_binary_tree(expr.children[1]);

  } else if (expr.kind == OperationExpr::REPEAT) {
    left = build_binary_tree(expr.children[0]);
  } else if (expr.kind == OperationExpr::EVENT) {
    // do nothing
  } else {
    LOG(FATAL) << "Invalid operation expression kind: " << expr.kind;
    std::exit(-1);
  }
  BinaryTree *tree = new BinaryTree(data, left, right);
  return tree;
}

void TimingModel::from_string(string str) {
  operations.clear();
  constraints.clear();

  // split the string into lines
  std::istringstream iss(str);
  std::string line;
  std::vector<string> lines;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }
  // parse the lines
  for (size_t i = 0; i < lines.size(); i++) {
    string line = lines[i];
    // remove all contents after the # symbol, since it is a comment
    size_t pos = line.find("#");
    if (pos != string::npos) {
      line = line.substr(0, pos);
    }
    // remove leading and trailing spaces
    const char *WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    line = start == end ? std::string() : line.substr(start, end - start + 1);
    // check if the line is empty
    if (line.empty() || line == "" || line == "\n") {
      continue;
    }

    if (line.find("operation") != string::npos) {
      // parse the operation
      Operation op(line);
      operations[op.name] = op;
    } else if (line.find("constraint") != string::npos) {
      // parse the constraint
      Constraint constraint(line);
      constraints.push_back(constraint);
    } else {
      LOG(FATAL) << "Invalid line: " << line;
      std::exit(-1);
    }
  }
}

} // namespace tm
} // namespace vesyla