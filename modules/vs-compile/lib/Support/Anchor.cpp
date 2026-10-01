#include "vesyla/Support/Anchor.hpp"

#include <algorithm>
#include <regex>

namespace vesyla {

namespace {

// Parse a field, e.g. "[0][12]", into its point indices. Returns nullopt if
// the string is not a non-empty run of "[<non-negative-int>]".
std::optional<std::vector<int>> parse_field(const std::string &group) {
  if (group.empty()) {
    return std::nullopt;
  }
  static const std::regex bracket_re("\\[([0-9]+)\\]");
  std::vector<int> indices;
  auto it = std::sregex_iterator(group.begin(), group.end(), bracket_re);
  auto end = std::sregex_iterator();
  std::size_t consumed = 0;
  for (; it != end; ++it) {
    // Every character must belong to a bracket; reject stray content.
    if (static_cast<std::size_t>(it->position()) != consumed) {
      return std::nullopt;
    }
    indices.push_back(std::stoi((*it)[1].str()));
    consumed += it->length();
  }
  if (consumed != group.size()) {
    return std::nullopt;
  }
  return indices;
}

// Split on '.' keeping empty tokens.
std::vector<std::string> split_dot(const std::string &s) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) {
    if (c == '.') {
      parts.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  parts.push_back(cur);
  return parts;
}

std::string render_field(const std::vector<int> &f) {
  std::string s;
  for (int x : f) {
    s += "[" + std::to_string(x) + "]";
  }
  return s;
}

} // namespace

bool Anchor::operator==(const Anchor &o) const {
  return name == o.name && or_idx == o.or_idx && mt_idx == o.mt_idx &&
         ir_idx == o.ir_idx;
}

std::optional<Anchor> Anchor::parse(const std::string &str_) {
  std::string str = str_;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());

  static const std::regex name_re("^([a-zA-Z_][a-zA-Z0-9_]*)(.*)$");
  std::smatch m;
  if (!std::regex_match(str, m, name_re)) {
    return std::nullopt;
  }

  Anchor a;
  a.name = m[1];
  std::string tail = m[2];

  // Bare name: MT defaults to 0, no OR/IR.
  if (tail.empty()) {
    return a;
  }

  std::vector<std::string> groups = split_dot(tail);
  bool leading_dot = groups.front().empty();
  if (leading_dot) {
    groups.erase(groups.begin());
  }
  // No group may be empty once the (single) leading dot is removed. This
  // rejects "..", trailing dots, and a bare "." tail.
  for (const auto &g : groups) {
    if (g.empty()) {
      return std::nullopt;
    }
  }

  const std::size_t n = groups.size();
  std::string or_str;
  std::string mt_str;
  std::string ir_str;

  if (leading_dot) {
    // OR absent: fields start at MT.
    if (n == 1) {
      mt_str = groups[0];
    } else if (n == 2) {
      mt_str = groups[0];
      ir_str = groups[1];
    } else {
      return std::nullopt;
    }
  } else {
    // OR present: a bare single group (OR alone, no MT) is illegal.
    if (n == 2) {
      or_str = groups[0];
      mt_str = groups[1];
    } else if (n == 3) {
      or_str = groups[0];
      mt_str = groups[1];
      ir_str = groups[2];
    } else {
      return std::nullopt;
    }
  }

  if (!or_str.empty()) {
    auto parsed = parse_field(or_str);
    if (!parsed) {
      return std::nullopt;
    }
    a.or_idx = std::move(*parsed);
  }
  // MT is a single index.
  auto mt_parsed = parse_field(mt_str);
  if (!mt_parsed || mt_parsed->size() != 1) {
    return std::nullopt;
  }
  a.mt_idx = (*mt_parsed)[0];
  if (!ir_str.empty()) {
    auto parsed = parse_field(ir_str);
    if (!parsed) {
      return std::nullopt;
    }
    a.ir_idx = std::move(*parsed);
  }

  return a;
}

std::string Anchor::to_string() const {
  // Bare form: MT is the default 0 and there is no OR/IR.
  if (mt_idx == 0 && !has_or() && !has_ir()) {
    return name;
  }
  std::string out = name;
  // A leading dot marks OR absent when a later field is present.
  if (!has_or()) {
    out += ".";
  }
  std::vector<std::string> parts;
  if (has_or()) {
    parts.push_back(render_field(or_idx));
  }
  parts.push_back("[" + std::to_string(mt_idx) + "]");
  if (has_ir()) {
    parts.push_back(render_field(ir_idx));
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) {
      out += ".";
    }
    out += parts[i];
  }
  return out;
}

std::string Anchor::flat_name(const std::string &owner) const {
  std::string s = owner;
  for (int v : or_idx) {
    s += "_o" + std::to_string(v);
  }
  s += "_e" + std::to_string(mt_idx);
  for (int v : ir_idx) {
    s += "_i" + std::to_string(v);
  }
  return s;
}

bool AnchorRange::valid() const {
  // A range spans only IR indices; the name, OR indices, and MT are shared.
  if (lo.name != hi.name || lo.or_idx != hi.or_idx || lo.mt_idx != hi.mt_idx) {
    return false;
  }
  if (lo.ir_idx.size() != hi.ir_idx.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lo.ir_idx.size(); ++i) {
    if (lo.ir_idx[i] > hi.ir_idx[i]) {
      return false;
    }
  }
  return true;
}

std::optional<AnchorRange> AnchorRange::parse(const std::string &str_) {
  std::string str = str_;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());

  // Expand every "[a:b]" bracket into "[a]" for the lo text and "[b]" for the
  // hi text; point brackets "[n]" go to both. Any number of IR dimensions may
  // be a range. The two point-anchor texts are then parsed by Anchor::parse,
  // and valid() rejects ranges that fall outside IR (OR/MT must match).
  std::string lo_str;
  std::string hi_str;
  std::size_t i = 0;
  while (i < str.size()) {
    if (str[i] != '[') {
      lo_str += str[i];
      hi_str += str[i];
      ++i;
      continue;
    }
    const std::size_t rb = str.find(']', i);
    if (rb == std::string::npos) {
      return std::nullopt;
    }
    std::string inner = str.substr(i + 1, rb - i - 1);
    const std::size_t colon = inner.find(':');
    if (colon == std::string::npos) {
      lo_str += "[" + inner + "]";
      hi_str += "[" + inner + "]";
    } else {
      if (inner.find(':', colon + 1) != std::string::npos) {
        return std::nullopt; // at most one bound per bracket
      }
      lo_str += "[" + inner.substr(0, colon) + "]";
      hi_str += "[" + inner.substr(colon + 1) + "]";
    }
    i = rb + 1;
  }

  auto lo = Anchor::parse(lo_str);
  auto hi = Anchor::parse(hi_str);
  if (!lo || !hi) {
    return std::nullopt;
  }
  AnchorRange r{*lo, *hi};
  if (!r.valid()) {
    return std::nullopt;
  }
  return r;
}

std::string AnchorRange::to_string() const {
  if (lo == hi) {
    return lo.to_string();
  }
  // lo/hi share name/OR/MT and differ only in IR (valid()); render the IR
  // brackets as ranges "[lo:hi]" where the bounds differ, "[n]" where equal.
  std::string out = lo.name;
  if (lo.or_idx.empty()) {
    out += "."; // leading dot marks OR absent
  }
  std::vector<std::string> parts;
  if (!lo.or_idx.empty()) {
    parts.push_back(render_field(lo.or_idx));
  }
  parts.push_back("[" + std::to_string(lo.mt_idx) + "]");
  if (!lo.ir_idx.empty()) {
    std::string ir;
    for (std::size_t i = 0; i < lo.ir_idx.size(); ++i) {
      ir += "[" + std::to_string(lo.ir_idx[i]);
      if (lo.ir_idx[i] != hi.ir_idx[i]) {
        ir += ":" + std::to_string(hi.ir_idx[i]);
      }
      ir += "]";
    }
    parts.push_back(ir);
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) {
      out += ".";
    }
    out += parts[i];
  }
  return out;
}

} // namespace vesyla
