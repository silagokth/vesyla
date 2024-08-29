#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 1408
#define M 9
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = (rand() % (2*M+1)) - M;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 1408
  // Read the input buffer to A. The starting address is 0, and the number of
  // row to read is 1.
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, 88);
  // Read the input buffer to B. The starting address is 1, and the number of
  // row to read is 1.
  // vector<int16_t> b = __input_buffer__.read<int16_t>(0, 2);
  // Add A and B
  vector<int16_t> c(1408);
  for(int i = 0; i < 1408; i++) {
      c[i] = (a[i] >= 0) ? a[i] : 0;
    }
  
  // Write the result C to the output buffer at starting address 0, and the
  // number of row to write is 1.
  __output_buffer__.write<int16_t>(0, 88, c);
  //exit(0);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
