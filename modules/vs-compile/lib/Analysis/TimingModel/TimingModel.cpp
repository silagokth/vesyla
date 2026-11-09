// TODO this should come from a config file
// or global variables at some point
#define MAX_SLOTS 16
#define NUM_PORTS_PER_RESOURCE 4
#define MAX_LATENCY 10000000

#include "vesyla/Analysis/TimingModel/TimingModel.hpp"

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

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
    str += "  constraint " + it->kind + " " + it->to_string() + ";\n";
  }
  return str;
}

int TimingModel::to_mzn(std::ostream &mzn_file, std::ostream &dzn_file,
                        bool allow_act_mode_2, bool feasibility_only) {
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
    // +1 accounts for the act-issue controller cycle preceding background
    // iteration; without it total_latency under-counts and halt fires mid-iter.
    // It stays 1 for any AGU_LATENCY: ScheduleEpochPass issues each ACT
    // AGU_LATENCY - 1 cycles early, so an access always lands one cycle after
    // its model time.
    mzn_file << "constraint op_end_vec[" +
                    std::to_string(op2idx[it->second.name]) +
                    "] == " + it->second.name + " + " +
                    it->second.duration_expr + " + 1;\n";
  }

  // add variables
  for (auto it = variables.begin(); it != variables.end(); ++it) {
    mzn_file << "var 0..MAX_LATENCY: " + *it + ";\n";
  }

  // add anchors
  for (auto it = anchors.begin(); it != anchors.end(); ++it) {
    mzn_file << "var 0..MAX_LATENCY: " + it->second.name + ";\n";
    if (it->second.timing_expr.empty()) {
      LOG_ERROR << "Anchor " << it->second.name
                << " does not have a timing expression!";
      // print all anchors
      for (auto it2 = anchors.begin(); it2 != anchors.end(); ++it2) {
        LOG_ERROR << "Anchor " << it2->second.name
                  << " with expr: " << it2->second.anchor.to_string()
                  << " and timing_expr: " << it2->second.timing_expr;
      }
      std::exit(EXIT_FAILURE);
    }
    mzn_file << "constraint " + it->second.name +
                    " == " + it->second.timing_expr + ";\n";
  }

  // add constraints
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    mzn_file << "constraint " + it->to_string() + ";\n";
  }

  // A feasibility-only model stops at the timing constraints: the act modes
  // only ever forbid schedules, so if this relaxation has none, no act mode
  // can have one either. Asking for any schedule rather than the shortest
  // lets the solver stop at the first one it finds.
  if (feasibility_only) {
    mzn_file << "solve satisfy;\n";
    return 0;
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

  extractVariables();
  extractAnchors();
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    resolveOperationTiming(it->second);
  }

  state = COMPILED;
}

void TimingModel::extractVariables() {
  std::function<void(OperationExpr &)> extract_variables =
      [&](OperationExpr &expr) {
        if (expr.kind == OperationExpr::TRANSIT ||
            expr.kind == OperationExpr::REPEAT) {
          string delay_expr = expr.parameters["delay"];
          LOG_DEBUG << "Delay expression: " << delay_expr;
          if (is_number(delay_expr)) {
            // Is it a number, do nothing
          } else if (is_identifier(delay_expr)) {
            variables.insert(delay_expr);
          } else {
            LOG_ERROR << "Invalid delay expression: " << delay_expr;
          }
          // The delay slope of a repeat may be a variable too.
          auto ds = expr.parameters.find("dslope");
          if (ds != expr.parameters.end() && !is_number(ds->second) &&
              is_identifier(ds->second)) {
            variables.insert(ds->second);
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
}

void TimingModel::extractAnchors() {
  // Register a tm::Anchor for every anchor referenced by a Constraint's
  // src_anchor / dst_anchor. The owning operation id becomes the anchor name,
  // and tm::Anchor derives the flat MZN identifier from it.
  auto register_anchor = [&](const std::string &id,
                             const std::optional<Constraint::Anchor> &a) {
    if (!a)
      return;
    ::vesyla::Anchor core = *a;
    core.name = id;
    Anchor anchor(core);
    anchors[anchor.name] = anchor;
  };

  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    if (it->kind != "linear") {
      LOG_FATAL << "Invalid constraint kind: " << it->kind;
      exit(EXIT_FAILURE);
    }
    register_anchor(it->src_id, it->src_anchor);
    register_anchor(it->dst_id, it->dst_anchor);
  }
}

namespace {

// A linear expression in the delay variables with integer coefficients, the
// form every start time and duration takes.
struct LinExpr {
  int64_t c = 0;
  std::map<std::string, int64_t> terms;
  LinExpr() {}
  explicit LinExpr(int64_t k) : c(k) {}
  static LinExpr of(const std::string &delay) {
    LinExpr e;
    if (is_number(delay)) {
      e.c = std::stoll(delay);
    } else {
      e.terms[delay] = 1;
    }
    return e;
  }
  LinExpr &operator+=(const LinExpr &o) {
    c += o.c;
    for (const auto &[k, v] : o.terms) {
      terms[k] += v;
    }
    return *this;
  }
  LinExpr operator*(int64_t n) const {
    LinExpr e;
    e.c = c * n;
    for (const auto &[k, v] : terms) {
      e.terms[k] = v * n;
    }
    return e;
  }
  std::string str() const {
    std::string s = "(" + std::to_string(c);
    for (const auto &[k, v] : terms) {
      if (v != 0) {
        s += "+(" + std::to_string(v) + ")*" + k;
      }
    }
    return s + ")";
  }
};

// Start times and durations of an operation's expression tree. A repeat with
// a slope (AGU Slope Levels) runs iter + slope * (index of the repeat
// `follows` repeats up) times, so a duration is summed pass by pass where a
// subtree's trip counts depend on the pass, and stays the closed form
// child * iter + delay * (iter - 1) where they do not.
class Timer {
public:
  using Node = BinaryTree<BinaryTreeData> *;

  explicit Timer(Node root) { collect_events(root); }

  LinExpr duration(Node n) {
    std::vector<std::pair<Node, int64_t>> ctx;
    return dur(n, ctx);
  }

  // The start of the `indices`-th instance (outermost repeat first; missing
  // indices read 0) of event `event_id`, relative to the operation's start.
  LinExpr start(Node n, int event_id, const std::vector<int> &indices) {
    std::vector<std::pair<Node, int64_t>> ctx;
    return time(n, ctx, event_id, indices);
  }

private:
  std::unordered_map<Node, std::set<int>> events_;

  void collect_events(Node n) {
    if (!n) {
      return;
    }
    collect_events(n->left);
    collect_events(n->right);
    std::set<int> &e = events_[n];
    if (n->data->expr.kind == OperationExpr::EVENT) {
      e.insert(std::stoi(n->data->expr.parameters.at("id")));
    }
    if (n->left) {
      e.insert(events_[n->left].begin(), events_[n->left].end());
    }
    if (n->right) {
      e.insert(events_[n->right].begin(), events_[n->right].end());
    }
  }

  static int64_t param(Node n, const char *key, int64_t fallback) {
    auto it = n->data->expr.parameters.find(key);
    return it == n->data->expr.parameters.end() ? fallback
                                                 : std::stoll(it->second);
  }

  // The delay after pass c of repeat n: delay + dslope * (c, or the index of
  // the repeat dfollows repeats up).
  static LinExpr pass_delay(Node n, const std::vector<std::pair<Node, int64_t>> &ctx,
                            int64_t c) {
    LinExpr d = LinExpr::of(n->data->expr.parameters.at("delay"));
    auto ds = n->data->expr.parameters.find("dslope");
    if (ds == n->data->expr.parameters.end() || ds->second == "0") {
      return d;
    }
    int64_t dfollows = param(n, "dfollows", 0);
    int64_t x = c;
    if (dfollows != 0) {
      if (dfollows > static_cast<int64_t>(ctx.size())) {
        LOG_FATAL << "A delay slope follows a repeat it does not have: "
                  << OperationExpr(n->data->expr).to_string();
        std::exit(EXIT_FAILURE);
      }
      x = ctx[ctx.size() - dfollows].second;
    }
    d += LinExpr::of(ds->second) * x;
    return d;
  }

  static bool has_dslope(Node n) {
    auto ds = n->data->expr.parameters.find("dslope");
    return ds != n->data->expr.parameters.end() && ds->second != "0";
  }

  static int64_t trips(Node n, const std::vector<std::pair<Node, int64_t>> &ctx) {
    int64_t iter = param(n, "iter", 1);
    int64_t slope = param(n, "slope", 0);
    int64_t follows = param(n, "follows", 0);
    if (follows == 0 || slope == 0) {
      return iter;
    }
    if (follows > static_cast<int64_t>(ctx.size())) {
      LOG_FATAL << "A repeat follows a repeat it does not have: "
                << OperationExpr(n->data->expr).to_string();
      std::exit(EXIT_FAILURE);
    }
    return iter + slope * ctx[ctx.size() - follows].second;
  }

  // Whether a repeat in `sub` follows the repeat `k` repeats above `sub`'s
  // root (1: the repeat directly around `sub`).
  static bool depends(Node sub, int64_t k) {
    if (!sub) {
      return false;
    }
    if (sub->data->expr.kind == OperationExpr::REPEAT) {
      if (param(sub, "slope", 0) != 0 && param(sub, "follows", 0) == k) {
        return true;
      }
      return depends(sub->left, k + 1);
    }
    return depends(sub->left, k) || depends(sub->right, k);
  }

  LinExpr dur(Node n, std::vector<std::pair<Node, int64_t>> &ctx) {
    const OperationExpr &e = n->data->expr;
    if (e.kind == OperationExpr::EVENT) {
      return LinExpr(1);
    }
    if (e.kind == OperationExpr::TRANSIT) {
      LinExpr d = dur(n->left, ctx);
      d += dur(n->right, ctx);
      d += LinExpr::of(e.parameters.at("delay"));
      return d;
    }
    int64_t t = trips(n, ctx);
    if (t < 1) {
      LOG_FATAL << "A repeat with no iterations: " << OperationExpr(e).to_string();
      std::exit(EXIT_FAILURE);
    }
    LinExpr d;
    if (!depends(n->left, 1)) {
      ctx.push_back({n, 0});
      LinExpr child = dur(n->left, ctx);
      ctx.pop_back();
      d = child * t;
    } else {
      for (int64_t c = 0; c < t; ++c) {
        ctx.push_back({n, c});
        d += dur(n->left, ctx);
        ctx.pop_back();
      }
    }
    if (!has_dslope(n)) {
      d += LinExpr::of(e.parameters.at("delay")) * (t - 1);
    } else {
      for (int64_t c = 0; c + 1 < t; ++c) {
        d += pass_delay(n, ctx, c);
      }
    }
    return d;
  }

  LinExpr time(Node n, std::vector<std::pair<Node, int64_t>> &ctx,
               int event_id, const std::vector<int> &indices) {
    const OperationExpr &e = n->data->expr;
    if (e.kind == OperationExpr::EVENT) {
      return LinExpr(0);
    }
    if (e.kind == OperationExpr::TRANSIT) {
      if (events_[n->left].count(event_id)) {
        return time(n->left, ctx, event_id, indices);
      }
      LinExpr t = dur(n->left, ctx);
      t += LinExpr::of(e.parameters.at("delay"));
      t += time(n->right, ctx, event_id, indices);
      return t;
    }
    int64_t idx = ctx.size() < indices.size() ? indices[ctx.size()] : 0;
    int64_t t = trips(n, ctx);
    if (idx >= t) {
      LOG_FATAL << "Index out of range: index(" << idx << ") >= iter(" << t
                << ") in " << OperationExpr(e).to_string();
      std::exit(EXIT_FAILURE);
    }
    LinExpr s;
    if (!depends(n->left, 1)) {
      ctx.push_back({n, 0});
      LinExpr child = dur(n->left, ctx);
      ctx.pop_back();
      s = child * idx;
    } else {
      for (int64_t c = 0; c < idx; ++c) {
        ctx.push_back({n, c});
        s += dur(n->left, ctx);
        ctx.pop_back();
      }
    }
    if (!has_dslope(n)) {
      s += LinExpr::of(e.parameters.at("delay")) * idx;
    } else {
      for (int64_t c = 0; c < idx; ++c) {
        s += pass_delay(n, ctx, c);
      }
    }
    ctx.push_back({n, idx});
    s += time(n->left, ctx, event_id, indices);
    ctx.pop_back();
    return s;
  }
};

} // namespace

void TimingModel::resolveOperationTiming(Operation &op) {
  OperationExpr expr = op.expr;
  BinaryTree<BinaryTreeData> *tree = build_binary_tree(expr);
  Timer timer(tree);

  op.duration_expr = timer.duration(tree).str();

  for (auto it = anchors.begin(); it != anchors.end(); ++it) {
    Anchor &anchor = it->second;
    if (anchor.anchor.name != op.name) {
      continue;
    }
    // OR indices (outer repeats) then IR indices (inner repeats), both
    // outer-first, the order of the repeats around the event.
    std::vector<int> indices = anchor.anchor.or_idx;
    indices.insert(indices.end(), anchor.anchor.ir_idx.begin(),
                   anchor.anchor.ir_idx.end());
    anchor.timing_expr =
        "(" + op.name + "+" +
        timer.start(tree, anchor.anchor.mt_idx, indices).str() + ")";
  }

  delete tree;
}

unordered_map<BinaryTree<BinaryTreeData> *, BinaryTree<BinaryTreeData> *>
TimingModel::buildParentMap(BinaryTree<BinaryTreeData> *tree) {
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
  return node_parent_map;
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
