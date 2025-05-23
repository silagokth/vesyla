/*
 * Uncomment the following macro to enable debug mode.
 */
#define DEBUG

/*
 * Uncomment the following macro to generate human readable output file for
 * debugging. It is mandatory to enable define DATA_TYPE if DEBUG is enabled.
 */
#define DATA_TYPE int16_t

#include "Drra.hpp"
#include "IO.hpp"
#include "Util.hpp"
#include <cstdlib>

int main() {
  IO __input_buffer__;
  IO __output_buffer__;

  // Initialize the input buffer
  init(__input_buffer__);
  store_data("mem/sram_image_in.bin", __input_buffer__);

  // if DEBUG is enabled, generate human readable (almost) output file
#ifdef DEBUG
  bin_to_hex_file("mem/sram_image_in.bin", "mem/sram_image_in.hex");
  bin_to_num_file<DATA_TYPE>("mem/sram_image_in.bin", "mem/sram_image_in.txt");
#endif

  // Run the model 0
  reset(__output_buffer__);
  model_l0(__input_buffer__, __output_buffer__);
  store_data("mem/sram_image_m0.bin", __output_buffer__);

  return 0;
}

/*
 * Generate the input SRAM image.
 */
void init(IO &input_buffer) {
#define N 32
  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> v(2 * N);
  for (auto i = 0; i < 2 * N; i++) {
    v[i] = rand() % 100;
  }

  // Write the random numbers to the input buffer.
  input_buffer.write<int16_t>(0, (2 * N) / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0(IO &input_buffer, IO &output_buffer) {
#define N 32
  // Read the input buffer to A and B.
  vector<int16_t> a = input_buffer.read<int16_t>(0, N / 16);
  vector<int16_t> b = input_buffer.read<int16_t>(N / 16, N / 16);
  vector<int16_t> c(N);

  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }

  // Write A to the output buffer
  output_buffer.write<int16_t>(0, N / 16, c);
}