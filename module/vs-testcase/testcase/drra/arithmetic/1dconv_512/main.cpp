#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 33*16
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
#define N 33*16
  // Read the input buffer to A. The starting address is 0, and the number of
  // row to read is 1.
  vector<int16_t> a = __input_buffer__.read<int16_t>(1, 32);
  // Read the input buffer to B. The starting address is 1, and the number of
  // row to read is 1.
  vector<int16_t> b = __input_buffer__.read<int16_t>(0, 1);
  // Add A and B
  vector<int16_t> c(512);
  for(int i = 0; i < 510; i++) {
        int sum = 0;
        for(int j = 0; j < 3; j++) {
            
            sum += a[i + j] * b[j];
            
        }
        c[i] = sum;
        //cout << i <<": " <<c[i] << endl;
    }

  c[510] = c[446];
  c[511] = c[447];
  
  
  // Write the result C to the output buffer at starting address 0, and the
  // number of row to write is 1.
  __output_buffer__.write<int16_t>(0, 32, c);
  //exit(0);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
