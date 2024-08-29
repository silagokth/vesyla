#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 22*65*16
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = rand() % 10;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 22*65*16
  // Read the input buffer to A. The starting address is 0, and the number of
  // row to read is 1.
  vector<int16_t> c(64);
  for(int s = 0; s < 64; s++)
  {
  vector<int16_t> a = __input_buffer__.read<int16_t>(0, 22);
  // Read the input buffer to B. The starting address is 1, and the number of
  // row to read is 1.
  vector<int16_t> b = __input_buffer__.read<int16_t>(22+s*22, 22);
  // Add A and B
  
  int temp = 0;
  for(int i = 0; i < 344; i++) {
        temp += a[i] * b[i];
}
        c[s] = temp;
  
        //cout << i <<": " <<c[i] << endl;
  }
  // Write the result C to the output buffer at starting address 0, and the
  // number of row to write is 1.
  __output_buffer__.write<int16_t>(0, 4, c);
  //exit(0);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
