#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>

/*
 * Uncomment the following macro to enable debug mode.
 */
// #define DEBUG

/*
 * Uncomment the following macro to generate human readable output file for
 * debugging. It is mandatory to enable define DATA_TYPE if DEBUG is enabled.
 */
// #define DATA_TYPE int16_t

int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init()
{
  // Set the seed for random number generator
  // srand((unsigned)time(NULL));
  srand(42);

  // Generate N random numbers in range [0,10)
  vector<int16_t> v((2 + 32) * 16);
  for (auto i = 0; i < (2 + 32)*16; i++)
  {
        v[i] = rand() % 10;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, (2 + 32), v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0()
{
  // Read the input buffer to A and B.
  vector<int16_t> input = __input_buffer__.read<int16_t>(0, 2);
  vector<int16_t> weights = __input_buffer__.read<int16_t>(2, 32);
  vector<int16_t> output(16);
  for (int i = 0; i < 16; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      output[i] += input[j] * weights[i * 32 + j];
    }
  }
  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, 1, output);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
