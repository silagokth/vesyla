#include "Array.hpp"

#ifndef ARCH_IO_DEPTH
#define ARCH_IO_DEPTH 65536
#endif
#ifndef ARCH_IO_WIDTH
#define ARCH_IO_WIDTH 256
#endif

typedef Array<ARCH_IO_DEPTH, ARCH_IO_WIDTH> IO;