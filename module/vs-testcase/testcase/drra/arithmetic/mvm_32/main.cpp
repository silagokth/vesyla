#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 32
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> v(N * (N + 1));
  for (auto i = 0; i < N * (N +1); i++) {
    v[i] = rand() % 3;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (N * (N +1)) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 32
  // Read the input buffer to A and B.
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, N*N / 16);
  vector<int16_t> x = __input_buffer__.read<int16_t>(N*N / 16, N/16);

  vector<int16_t> y(N);
  for (int i = 0; i < N; i++) {
    y[i]=0;
    for (int j = 0; j < N; j++) {
      y[i] += a[i * N + j] * x[j];
    }
  }
  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, N / 16, y);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
