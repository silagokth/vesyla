#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 512
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> v(2 * N);
  for (auto i = 0; i < 2 * N; i++) {
    v[i] = rand() % 100;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (2 * N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 512
  // Read the input buffer to A and B.
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, N / 16);
  vector<int16_t> b = __input_buffer__.read<int16_t>(N / 16, N / 16);
  vector<int16_t> c(N);
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, N / 16, c);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
