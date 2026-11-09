#ifndef __VESYLA_PASM_ANCHOR_DOMAIN_HPP__
#define __VESYLA_PASM_ANCHOR_DOMAIN_HPP__

#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

// The iteration domain an AnchorRangeAttr spans (AGU Slope Levels). Without
// slopes it is the box [ir_lo, ir_hi]; a dimension with a slope has bounds
// that move with the index of the dimension `follows` dimensions out, so the
// domain is enumerated in program order, outermost dimension first.
namespace vesyla::pasm {

struct RangeDim {
  uint32_t lo = 0;
  uint32_t hi = 0;
  int32_t lo_slope = 0;
  int32_t hi_slope = 0;
  uint32_t follows = 0;
};

inline llvm::SmallVector<RangeDim> range_dims(AnchorRangeAttr a) {
  llvm::SmallVector<RangeDim> dims;
  for (size_t d = 0; d < a.getIrLo().size(); ++d) {
    RangeDim dim;
    dim.lo = a.getIrLo()[d];
    dim.hi = a.getIrHi()[d];
    if (d < a.getIrLoSlope().size()) {
      dim.lo_slope = a.getIrLoSlope()[d];
      dim.hi_slope = a.getIrHiSlope()[d];
      dim.follows = a.getIrFollows()[d];
    }
    dims.push_back(dim);
  }
  return dims;
}

inline bool has_slopes(llvm::ArrayRef<RangeDim> dims) {
  for (const RangeDim &d : dims) {
    if (d.follows != 0 && (d.lo_slope != 0 || d.hi_slope != 0)) {
      return true;
    }
  }
  return false;
}

inline bool has_slopes(AnchorRangeAttr a) { return has_slopes(range_dims(a)); }

// The bounds of dimension `d` at a point whose outer dimensions are `point`.
// A sloped dimension's stored bound is the value at index 0 of the followed
// dimension, which may be negative (an empty first pass): it is kept as a
// signed 32-bit value in the unsigned field.
inline int64_t dim_lo(llvm::ArrayRef<RangeDim> dims, size_t d,
                      llvm::ArrayRef<uint32_t> point) {
  const RangeDim &dim = dims[d];
  if (dim.follows == 0 || dim.follows > d) {
    return dim.lo;
  }
  return int64_t(static_cast<int32_t>(dim.lo)) +
         int64_t(dim.lo_slope) * point[d - dim.follows];
}

inline int64_t dim_hi(llvm::ArrayRef<RangeDim> dims, size_t d,
                      llvm::ArrayRef<uint32_t> point) {
  const RangeDim &dim = dims[d];
  if (dim.follows == 0 || dim.follows > d) {
    return dim.hi;
  }
  return int64_t(static_cast<int32_t>(dim.hi)) +
         int64_t(dim.hi_slope) * point[d - dim.follows];
}

// Every point of the domain, in program order. A dimension whose bounds cross
// at some outer point contributes nothing there.
inline std::vector<std::vector<uint32_t>>
range_points(llvm::ArrayRef<RangeDim> dims) {
  std::vector<std::vector<uint32_t>> points;
  std::vector<uint32_t> cur(dims.size(), 0);
  std::function<void(size_t)> walk = [&](size_t d) {
    if (d == dims.size()) {
      points.push_back(cur);
      return;
    }
    int64_t lo = dim_lo(dims, d, cur);
    int64_t hi = dim_hi(dims, d, cur);
    for (int64_t i = lo; i <= hi; ++i) {
      cur[d] = static_cast<uint32_t>(i);
      walk(d + 1);
    }
  };
  walk(0);
  return points;
}

// The box around the domain: per dimension the least lo and the greatest hi
// over the followed dimension's own box. A dimension empty everywhere gets
// hi = lo.
inline std::pair<llvm::SmallVector<uint32_t>, llvm::SmallVector<uint32_t>>
range_box(llvm::ArrayRef<RangeDim> dims) {
  llvm::SmallVector<uint32_t> lo, hi;
  for (size_t d = 0; d < dims.size(); ++d) {
    const RangeDim &dim = dims[d];
    if (dim.follows == 0 || dim.follows > d) {
      lo.push_back(dim.lo);
      hi.push_back(dim.hi);
      continue;
    }
    int64_t flo = lo[d - dim.follows];
    int64_t fhi = hi[d - dim.follows];
    int64_t l0 = int64_t(static_cast<int32_t>(dim.lo));
    int64_t h0 = int64_t(static_cast<int32_t>(dim.hi));
    int64_t l = std::min(l0 + int64_t(dim.lo_slope) * flo,
                         l0 + int64_t(dim.lo_slope) * fhi);
    int64_t h = std::max(h0 + int64_t(dim.hi_slope) * flo,
                         h0 + int64_t(dim.hi_slope) * fhi);
    l = std::max<int64_t>(l, 0);
    h = std::max<int64_t>(h, l);
    lo.push_back(static_cast<uint32_t>(l));
    hi.push_back(static_cast<uint32_t>(h));
  }
  return {lo, hi};
}

// How many points the domain holds.
inline uint64_t range_size(AnchorRangeAttr a) {
  llvm::SmallVector<RangeDim> dims = range_dims(a);
  if (!has_slopes(dims)) {
    uint64_t n = 1;
    for (const RangeDim &d : dims) {
      n *= static_cast<uint64_t>(d.hi) - d.lo + 1;
    }
    return n;
  }
  return range_points(dims).size();
}

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_ANCHOR_DOMAIN_HPP__
