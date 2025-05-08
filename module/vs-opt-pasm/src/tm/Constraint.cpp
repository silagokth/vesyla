#include "Constraint.hpp"

namespace vesyla {
namespace tm {
string Constraint::to_string() { return "constraint " + kind + " " + expr; }

} // namespace tm
} // namespace vesyla