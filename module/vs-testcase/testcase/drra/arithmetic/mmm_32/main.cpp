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
  vector<int16_t> v(2 * N * N);
  for (auto i = 0; i < 2 * N * N; i++) {
    v[i] = rand() % 3;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (2 * N * N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 32
  // Read the input buffer to A and B.
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, N * N / 16);
  vector<int16_t> b = __input_buffer__.read<int16_t>(N * N / 16, N * N / 16);

  vector<int16_t> c(1 * N);
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < N; j++) {
      c[i * N + j]=0;
      for (int k = 0; k < N; k++) {
        c[i * N + j] += a[i * N + k] * b[j * N + k];
      }
    }
  }
  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, 1 * N / 16, c);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
