#ifndef __VESYLA_PASM_OPTION_LIMIT_HPP__
#define __VESYLA_PASM_OPTION_LIMIT_HPP__

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace vesyla::pasm {

// The configuration options a resource instance of `kind` switches between
// within an epoch: its NUM_OPTIONS (`num_options`), capped in reconfigurable
// mode by what its ISA option fields can name (conf `option`, evt `option` /
// `init_addr`). Without reconfiguration the options are hardcoded in the
// fabric and never encoded, so the table size alone counts. 0 stays 0: the
// kind takes no option.
int64_t option_limit(llvm::StringRef kind, int64_t num_options);

// The same for the resource instance at (row, col, slot) of the loaded
// architecture, from its kind and NUM_OPTIONS (4 when not given).
int64_t option_limit_at(int row, int col, int slot);

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_OPTION_LIMIT_HPP__
