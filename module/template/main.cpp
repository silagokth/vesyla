#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 1024
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate 32 random numbers in range [0,100) for both vector A and B
  vector<int16_t> v(2 * N);
  for (auto i = 0; i < 2 * N; i++) {
    v[i] = rand() % 100;
  }

  // Write the random numbers to the input buffer at starting address 0, and the
  // number of row to write is 2.
  __input_buffer__.write<int16_t>(0, (2 * N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 1024
  // Read the input buffer to A. The starting address is 0, and the number of
  // row to read is 1.
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, N / 16);
  // Read the input buffer to B. The starting address is 1, and the number of
  // row to read is 1.
  vector<int16_t> b = __input_buffer__.read<int16_t>(N / 16, N / 16);
  // Add A and B
  vector<int16_t> c(N);
  for (auto i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
  // Write the result C to the output buffer at starting address 0, and the
  // number of row to write is 1.
  __output_buffer__.write<int16_t>(0, N / 16, c);
}
