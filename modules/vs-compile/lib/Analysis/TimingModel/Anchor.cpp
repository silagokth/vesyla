#include "vesyla/Analysis/TimingModel/Anchor.hpp"

namespace vesyla {
namespace tm {

Anchor::Anchor(string expr_str) {
  auto parsed = ::vesyla::Anchor::parse(expr_str);
  if (!parsed) {
    LOG_FATAL << "Invalid anchor string: " << expr_str;
    std::exit(EXIT_FAILURE);
  }
  anchor = *parsed;
  name = anchor.flat_name(anchor.name);
}

Anchor::Anchor(::vesyla::Anchor anchor_) : anchor(anchor_) {
  name = anchor.flat_name(anchor.name);
}

Anchor::~Anchor() {}

string Anchor::to_string() {
  return "anchor " + name + " " + anchor.to_string();
}

} // namespace tm
} // namespace vesyla
