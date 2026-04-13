// TODO this should come from a config file
// or global variables at some point
#define MAX_SLOTS 16
#define NUM_PORTS_PER_RESOURCE 4
#define MAX_LATENCY 10000000

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

int TimingModel::to_mzn(std::ostream &mzn_file, std::ostream &dzn_file,
                        bool allow_act_mode_2) {
  if (state != COMPILED) {
    LOG_WARNING << "Timing model is not compiled. Compiling...";
    compile();
    LOG_DEBUG << "Timing model compiled.";
  }

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

  // generate dzn file
  LOG_DEBUG << "Generating DZN file...";
  dzn_file << "op_cell_row = array1d(0.." + std::to_string(num_ops - 1) +
                  ", [\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    dzn_file << it->second.row;
    dzn_file << ", % " << it->second.name << "\n";
  }
  dzn_file << "]);\n\n";
  LOG_DEBUG << "cell_row generated with " << num_ops << " operations.\n";
  dzn_file << "op_cell_col = array1d(0.." + std::to_string(num_ops - 1) +
                  ", [\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    dzn_file << it->second.col;
    dzn_file << ", % " << it->second.name << "\n";
  }
  dzn_file << "]);\n\n";
  LOG_DEBUG << "cell_col generated with " << num_ops << " operations.\n";
  dzn_file << "op_slot = array1d(0.." + std::to_string(num_ops - 1) + ", [\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    dzn_file << it->second.slot;
    dzn_file << ", % " << it->second.name << "\n";
  }
  dzn_file << "]);\n\n";
  LOG_DEBUG << "slot generated with " << num_ops << " operations.\n";
  dzn_file << "op_port = array1d(0.." + std::to_string(num_ops - 1) + ", [\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    dzn_file << it->second.port;
    dzn_file << ", % " << it->second.name << "\n";
  }
  dzn_file << "]);\n\n";
  LOG_DEBUG << "DZN file generated with " << num_ops << " operations.\n";

  // generate minizinc model
  mzn_file << "include \"globals.mzn\";\n";
  mzn_file << "int: MAX_LATENCY = " + std::to_string(MAX_LATENCY) + ";\n";
  mzn_file << "int: NUM_OPS = " + std::to_string(num_ops) + ";\n";
  mzn_file << "int: MAX_SLOTS = " + std::to_string(MAX_SLOTS) + ";\n";
  mzn_file << "int: NUM_PORTS_PER_RESOURCE = " +
                  std::to_string(NUM_PORTS_PER_RESOURCE) + ";\n";

  // add total latency variable
  mzn_file << "var 0..MAX_LATENCY: total_latency;\n";

  // add operations
  mzn_file << "array [0..NUM_OPS-1] of var 0..MAX_LATENCY: op_start_vec;\n";
  mzn_file << "array [0..NUM_OPS-1] of var 0..MAX_LATENCY: op_end_vec;\n";

  mzn_file << "array [0..NUM_OPS-1] of int: op_cell_row;\n";
  mzn_file << "array [0..NUM_OPS-1] of int: op_cell_col;\n";

  mzn_file << "array [0..NUM_OPS-1] of var 0..MAX_SLOTS-1: op_slot;\n";
  mzn_file
      << "array [0..NUM_OPS-1] of var 0..NUM_PORTS_PER_RESOURCE-1: op_port;\n";

  mzn_file << "constraint min(op_start_vec) == 0;\n";
  mzn_file << "constraint max(op_end_vec) == total_latency;\n";
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    mzn_file << "var 0..MAX_LATENCY: " + it->second.name + ";\n";
    mzn_file << "constraint op_start_vec[" +
                    std::to_string(op2idx[it->second.name]) +
                    "] == " + it->second.name + ";\n";
    mzn_file << "constraint op_end_vec[" +
                    std::to_string(op2idx[it->second.name]) +
                    "] == " + it->second.name + " + " +
                    it->second.duration_expr + ";\n";
  }

  // add variables
  for (auto it = variables.begin(); it != variables.end(); ++it) {
    mzn_file << "var 0..MAX_LATENCY: " + *it + ";\n";
  }

  // add anchors
  for (auto it = anchors.begin(); it != anchors.end(); ++it) {
    mzn_file << "var 0..MAX_LATENCY: " + it->second.name + ";\n";
    mzn_file << "constraint " + it->second.name +
                    " == " + it->second.timing_expr + ";\n";
  }

  // add constraints
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    mzn_file << "constraint " + it->expr + ";\n";
  }

  // add act modes
  mzn_file << "var bool: use_act_mode_0;\n";
  mzn_file << "var bool: use_act_mode_1;\n";
  if (allow_act_mode_2) {
    mzn_file << "constraint (use_act_mode_0 + use_act_mode_1) <= 1;\n";
  } else {
    mzn_file << "constraint (use_act_mode_0 + use_act_mode_1) = 1;\n";
  }
  mzn_file << R"(
% act mode 0: if resources are within 4 slots of each other
% then their start times can be the same, else they must not have the same start time
constraint if use_act_mode_0 then
   forall(i, j in 0..NUM_OPS-1 where i < j) (
        (op_slot[i] != op_slot[j] /\ op_cell_row[i] == op_cell_row[j] /\ op_cell_col[i] == op_cell_col[j]) -> (
          (abs(op_slot[i] - op_slot[j]) >= 4) -> (op_start_vec[i] != op_start_vec[j])
        )
     )
endif;
  )";
  mzn_file << R"(
constraint if use_act_mode_1 then
  % for all operations
  forall(i, j in 0..NUM_OPS-1 where i < j) (
    % if two operations start at the same time, are in the same cell,
    % and are in different slots
    (op_start_vec[i] == op_start_vec[j] /\
     op_cell_row[i] == op_cell_row[j] /\
     op_cell_col[i] == op_cell_col[j] /\
     op_slot[i] != op_slot[j]) -> (
      % for each port P
      forall(P in 0..NUM_PORTS_PER_RESOURCE-1) (
        % the operations must use the same port P
        (exists(k in 0..NUM_OPS-1) (
          op_start_vec[k] == op_start_vec[i] /\
          op_cell_row[k] == op_cell_row[i] /\
          op_cell_col[k] == op_cell_col[i] /\
          op_slot[k] == op_slot[i] /\
          op_port[k] == P
        )) <-> (exists(m in 0..NUM_OPS-1) (
          op_start_vec[m] == op_start_vec[j] /\
          op_cell_row[m] == op_cell_row[j] /\
          op_cell_col[m] == op_cell_col[j] /\
          op_slot[m] == op_slot[j] /\
          op_port[m] == P
        ))
      )
    )
  )
endif;
  )";

  // add objective
  mzn_file << "solve minimize total_latency;\n";

  return 0;
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
          LOG_DEBUG << "Delay expression: " << delay_expr;
          // Is it a number or a variable?
          if (is_number(delay_expr)) {
            // Is it a number, do nothing
          } else if (is_identifier(delay_expr)) {
            // Is it a variable
            variables.insert(delay_expr);
          } else {
            LOG_ERROR << "Invalid delay expression: " << delay_expr;
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

  // Pre-process wildcard constraints: expand anchor patterns containing '*'
  // e.g. "A.e0[*] != B.e0[*]" with A having 3 events and B having 4 events
  // expands to 12 specific constraints (cartesian product).
  {
    std::string wc_pattern_str =
        "([a-zA-Z_][a-zA-Z0-9_]*\\.e[0-9]+)(\\s*\\[(-?[0-9]+|\\*)\\])*";
    std::regex wc_pattern_regex(wc_pattern_str);

    std::vector<Constraint> expanded_constraints;
    std::vector<bool> to_remove_flags(constraints.size(), false);

    for (size_t ci = 0; ci < constraints.size(); ++ci) {
      Constraint &cstr = constraints[ci];
      std::string expr = cstr.expr;

      // Collect all wildcard anchor occurrences with their positions
      struct WildcardMatch {
        size_t start;
        size_t length;
        std::vector<std::string> expansions;
      };
      std::vector<WildcardMatch> wc_matches;

      std::sregex_iterator it(expr.begin(), expr.end(), wc_pattern_regex);
      std::sregex_iterator end_it;

      for (; it != end_it; ++it) {
        std::string matched = (*it)[0];
        if (matched.find('*') == std::string::npos)
          continue;

        // Parse the wildcard anchor expression
        AnchorExpr anchor_expr(matched);
        std::string op_name = anchor_expr.op_name;

        if (operations.find(op_name) == operations.end()) {
          LOG_FATAL << "Operation not found for wildcard anchor: " << op_name;
          exit(EXIT_FAILURE);
        }

        Operation &op = operations[op_name];
        std::string target_event = "e" + std::to_string(anchor_expr.event_id);

        // Find all specific anchor strings that match the wildcard pattern
        std::vector<std::string> specific_anchors;
        std::regex idx_re("\\[([0-9]+)\\]");
        std::vector<std::string> event_anchors;
        for (const auto &anchor_str : op.get_all_anchors()) {
          if (anchor_str.substr(0, target_event.size()) != target_event)
            continue;
          if (anchor_str.size() > target_event.size() &&
              anchor_str[target_event.size()] != '[')
            continue;
          event_anchors.push_back(anchor_str);
        }

        // Determine the number of iterations per dimension by scanning all
        // event anchors, so negative indices can be resolved.
        std::vector<int> dim_iters;
        for (const auto &anchor_str : event_anchors) {
          std::vector<int> ai;
          std::smatch m;
          std::string remaining = anchor_str.substr(target_event.size());
          std::string::const_iterator ss(remaining.cbegin());
          while (std::regex_search(ss, remaining.cend(), m, idx_re)) {
            ss = m.suffix().first;
            ai.push_back(std::stoi(m[1]));
          }
          while (dim_iters.size() < ai.size())
            dim_iters.push_back(0);
          for (size_t d = 0; d < ai.size(); d++)
            dim_iters[d] = std::max(dim_iters[d], ai[d] + 1);
        }

        for (const auto &anchor_str : event_anchors) {
          // Parse the anchor's indices
          std::vector<int> anchor_indices;
          std::smatch m;
          std::string remaining = anchor_str.substr(target_event.size());
          std::string::const_iterator ss(remaining.cbegin());
          while (std::regex_search(ss, remaining.cend(), m, idx_re)) {
            ss = m.suffix().first;
            anchor_indices.push_back(std::stoi(m[1]));
          }

          // Dimensions must match
          if (anchor_indices.size() != anchor_expr.indices.size())
            continue;

          // Each non-wildcard index must equal the anchor's index.
          // Negative indices are resolved Python-style using the iteration count.
          bool matches = true;
          for (size_t i = 0; i < anchor_expr.indices.size(); i++) {
            if (!anchor_expr.wildcard_mask[i]) {
              int expected = anchor_expr.indices[i];
              if (expected < 0) {
                if (i < dim_iters.size())
                  expected = dim_iters[i] + expected;
                else {
                  matches = false;
                  break;
                }
              }
              if (expected != anchor_indices[i]) {
                matches = false;
                break;
              }
            }
          }

          if (matches) {
            specific_anchors.push_back(op_name + "." + anchor_str);
          }
        }

        if (specific_anchors.empty()) {
          LOG_WARNING << "No anchors matched wildcard pattern: " << matched;
        }

        wc_matches.push_back(
            {(size_t)it->position(), (size_t)matched.size(), specific_anchors});
      }

      if (wc_matches.empty())
        continue;

      // Mark original wildcard constraint for removal
      to_remove_flags[ci] = true;

      // Generate cartesian product of all wildcard expansions
      std::vector<std::vector<size_t>> combos = {{}};
      for (const auto &wm : wc_matches) {
        std::vector<std::vector<size_t>> new_combos;
        for (const auto &combo : combos) {
          for (size_t j = 0; j < wm.expansions.size(); j++) {
            std::vector<size_t> new_combo = combo;
            new_combo.push_back(j);
            new_combos.push_back(new_combo);
          }
        }
        combos = new_combos;
      }

      // Create one new constraint per combination
      for (const auto &combo : combos) {
        std::string new_expr = expr;
        // Replace in reverse order to preserve earlier positions
        for (int i = (int)wc_matches.size() - 1; i >= 0; i--) {
          new_expr.replace(wc_matches[i].start, wc_matches[i].length,
                           wc_matches[i].expansions[combo[i]]);
        }
        expanded_constraints.push_back(Constraint(cstr.kind, new_expr));
      }
    }

    // Rebuild constraints: keep non-wildcard ones, append expanded ones
    std::vector<Constraint> updated_constraints;
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
      if (!to_remove_flags[ci]) {
        updated_constraints.push_back(constraints[ci]);
      }
    }
    for (const auto &c : expanded_constraints) {
      updated_constraints.push_back(c);
    }
    constraints = updated_constraints;
  }

  // Extract all anchors from constraints:
  // op_name.e<event_id>[<index_0>][<index_1>]...
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    if (it->kind != "linear") {
      LOG_FATAL << "Invalid constraint kind: " << it->kind;
      exit(EXIT_FAILURE);
    }

    // Extract all anchors from constraint
    // e.g. op_name.e<event_id>
    // e.g. op_name.e<event_id>[<index_0>]
    // e.g. op_name.e<event_id>[<index_0>][<index_1>]...
    string pattern = "([a-zA-Z_][a-zA-Z0-9_]*\\.e[0-9]+)(\\s*\\[(-?[0-9]+)\\])*";
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
    BinaryTree<BinaryTreeData> *tree = build_binary_tree(expr);

    // calculate the duration of the operation
    tree->traverse_LRC([](BinaryTree<BinaryTreeData> *node) {
      BinaryTreeData *tree_data = node->data;
      BinaryTree<BinaryTreeData> *left = node->left;
      BinaryTree<BinaryTreeData> *right = node->right;
      if (tree_data->expr.kind == OperationExpr::TRANSIT) {
        string left_duration = left->data->duration;
        string right_duration = right->data->duration;
        tree_data->duration = "(" + left_duration + "+" + right_duration +
                              "+(" + tree_data->expr.parameters["delay"] + "))";
      } else if (tree_data->expr.kind == OperationExpr::REPEAT) {
        string left_duration = left->data->duration;
        int iter = std::stoi(tree_data->expr.parameters["iter"]);
        tree_data->duration = "(" + left_duration + "*" + std::to_string(iter) +
                              "+(" + tree_data->expr.parameters["delay"] +
                              ")*" + std::to_string(iter - 1) + ")";
      } else if (tree_data->expr.kind == OperationExpr::EVENT) {
        tree_data->duration = "(1)";
      }
    });
    // calculate the start time of the operation
    tree->data->start = "(0)";
    tree->traverse_CLR([](BinaryTree<BinaryTreeData> *node) {
      BinaryTree<BinaryTreeData> *left = node->left;
      BinaryTree<BinaryTreeData> *right = node->right;
      BinaryTreeData *tree_data = node->data;
      if (left) {
        left->data->start = tree_data->start;
      }
      if (right) {
        right->data->start = "(" + tree_data->start + "+" +
                             left->data->duration + "+(" +
                             tree_data->expr.parameters["delay"] + "))";
      }
    });

    unordered_map<BinaryTree<BinaryTreeData> *, BinaryTree<BinaryTreeData> *>
        node_parent_map;
    tree->traverse_CLR([&node_parent_map](BinaryTree<BinaryTreeData> *node) {
      BinaryTree<BinaryTreeData> *left = node->left;
      BinaryTree<BinaryTreeData> *right = node->right;
      if (left) {
        node_parent_map[left] = node;
      }
      if (right) {
        node_parent_map[right] = node;
      }
    });

    for (auto it2 = anchors_in_op.begin(); it2 != anchors_in_op.end(); ++it2) {
      string anchor_name = *it2;
      string op_name = it->first;
      Anchor &anchor = anchors[anchor_name];
      string event_id = std::to_string(anchor.expr.event_id);
      std::vector<BinaryTree<BinaryTreeData> *> r_op_stack;
      for (auto it3 = node_parent_map.begin(); it3 != node_parent_map.end();
           ++it3) {
        if (it3->first->data->expr.kind == OperationExpr::EVENT) {
          string event_id_1 = it3->first->data->expr.parameters["id"];
          if (event_id_1 == event_id) {
            // go through all its parents and find the repeat operation
            BinaryTree<BinaryTreeData> *parent = it3->second;

            while (parent) {
              if (parent->data->expr.kind == OperationExpr::REPEAT) {
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
              LOG_ERROR << "r_op_stack size: " << r_op_stack.size();
              for (auto i = 0; i < r_op_stack.size(); ++i) {
                LOG_ERROR << "r_op_stack[" << i
                          << "]: " << r_op_stack[i]->data->expr.to_string();
              }
              LOG_ERROR << "indices size: " << indices.size();
              for (auto i = 0; i < indices.size(); ++i) {
                LOG_ERROR << "indices[" << i << "]: " << indices[i];
              }
              LOG_FATAL << "Too many indices!";
              std::exit(EXIT_FAILURE);
            }

            string expr_str = "(" + op_name + "+" + it3->first->data->start;
            for (auto i = 0; i < indices.size(); ++i) {
              int index = indices[i];
              int iter =
                  std::stoi(r_op_stack[i]->data->expr.parameters["iter"]);
              // resolve negative indices (Python-style: -1 means last)
              int resolved = (!anchor.expr.wildcard_mask[i] && index < 0)
                                 ? iter + index
                                 : index;
              if (resolved < 0 || resolved >= iter) {
                LOG_FATAL << "Index out of range: resolved(" << resolved
                          << ") not in [0, " << iter << ")";
                std::exit(EXIT_FAILURE);
              }
              expr_str = expr_str + "+(" + r_op_stack[i]->left->data->duration +
                         "+(" + r_op_stack[i]->data->expr.parameters["delay"] +
                         "))*" + std::to_string(resolved);
            }
            expr_str += ")";
            anchor.timing_expr = expr_str;
          }
        }
      }
    }

    it->second.duration_expr = tree->data->duration;

    // delete the tree
    delete tree;
  }

  // Compilation logic
  state = COMPILED;
}

BinaryTree<BinaryTreeData> *
TimingModel::build_binary_tree(OperationExpr &expr) {
  BinaryTreeData *data = new BinaryTreeData();
  data->expr = expr;
  data->start = "";
  data->duration = "";
  BinaryTree<BinaryTreeData> *left = nullptr;
  BinaryTree<BinaryTreeData> *right = nullptr;
  if (expr.kind == OperationExpr::TRANSIT) {
    left = build_binary_tree(expr.children[0]);
    right = build_binary_tree(expr.children[1]);

  } else if (expr.kind == OperationExpr::REPEAT) {
    left = build_binary_tree(expr.children[0]);
  } else if (expr.kind == OperationExpr::EVENT) {
    // do nothing
  } else {
    LOG_FATAL << "Invalid operation expression kind: " << expr.kind;
    std::exit(EXIT_FAILURE);
  }
  BinaryTree<BinaryTreeData> *tree =
      new BinaryTree<BinaryTreeData>(data, left, right);
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
      LOG_FATAL << "Invalid line: " << line;
      std::exit(EXIT_FAILURE);
    }
  }
}

Operation TimingModel::get_operation(string name) {
  if (operations.find(name) == operations.end()) {
    LOG_FATAL << "Operation not found: " << name;
    std::exit(EXIT_FAILURE);
  }
  return operations[name];
}

} // namespace tm
} // namespace vesyla
