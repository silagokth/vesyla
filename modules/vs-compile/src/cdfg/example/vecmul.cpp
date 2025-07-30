// dot_product.cpp
#include <cstdint>
#include <vector>

#define N 128 // Size of the input vectors

[[clang::annotate("drra")]]
void vecmul() {
  [[clang::annotate("input_buffer")]]
  std::vector<int16_t> a(N);
  [[clang::annotate("input_buffer")]]
  std::vector<int16_t> b(N);

  [[clang::annotate("output_buffer")]]
  int16_t result = 0;

  [[clang::annotate("loop")]]
  for (int16_t i = 0; i < N; ++i) {
    [[clang::annotate("thisisdpu")]] result = a[i] * b[i];
  }
}
