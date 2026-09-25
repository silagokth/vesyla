#ifndef __VESYLA_SUPPORT_ANCHOR_HPP__
#define __VESYLA_SUPPORT_ANCHOR_HPP__

#include <optional>
#include <string>
#include <vector>

namespace vesyla {

// A point-valued reference to an operation event.
//
// Text form: name[OR].[MT].[IR]
//   - OR (outer repetition) and IR (inner repetition) are each a run of zero or
//     more point indices "[n][m]...".
//   - MT (middle transition) is a single, required index. It always has a
//     value; when left out of the text it defaults to 0.
//   - OR or IR may appear in the text only when MT is written too. A leading
//     dot (immediately after the name) means OR is absent.
//   - An absent OR/IR reads as 0 (see *_or_zero()).
//
// Legal shapes: name | name.[MT] | name[OR].[MT] | name.[MT].[IR] |
//               name[OR].[MT].[IR]. A bare name means MT = 0 with no OR/IR.
struct Anchor {
  std::string name;
  std::vector<int> or_idx;
  int mt_idx = 0;
  std::vector<int> ir_idx;

  // Whether an optional field was written in the text.
  bool has_or() const { return !or_idx.empty(); }
  bool has_ir() const { return !ir_idx.empty(); }

  // Effective indices: an absent OR/IR field defaults to a single 0.
  std::vector<int> or_or_zero() const {
    return has_or() ? or_idx : std::vector<int>{0};
  }
  std::vector<int> ir_or_zero() const {
    return has_ir() ? ir_idx : std::vector<int>{0};
  }

  bool operator==(const Anchor &o) const;
  bool operator!=(const Anchor &o) const { return !(*this == o); }

  // Parse an anchor from text. Returns nullopt on any syntax or rule
  // violation (e.g. OR without MT, an MT field with more than one bracket, a
  // range ':' in a single anchor).
  static std::optional<Anchor> parse(const std::string &str);

  // Serialize back to canonical text.
  std::string to_string() const;

  // A flat, MZN-safe identifier combining an owner id with the indices,
  // e.g. owner "op0" with mt=1, or={2}, ir={3} -> "op0_e1_o2_i3".
  std::string flat_name(const std::string &owner) const;
};

// A range between two anchors that are identical except (optionally) in the
// last IR index, where lo <= hi. Text uses the compact sugar
// name[..].[..].[..:..], with the range on the innermost IR bracket. A range
// with no ':' (or lo == hi) is a degenerate single point.
struct AnchorRange {
  Anchor lo;
  Anchor hi;

  bool valid() const;

  static std::optional<AnchorRange> parse(const std::string &str);

  std::string to_string() const;
};

} // namespace vesyla

#endif // __VESYLA_SUPPORT_ANCHOR_HPP__
