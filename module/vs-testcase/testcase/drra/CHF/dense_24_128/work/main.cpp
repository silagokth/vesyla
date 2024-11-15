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
void init() {
#define DIM_INPUT 24
#define DIM_INPUT_FILL 32
#define DIM_OUTPUT 128
#define DIM_OUTPUT_FILL 128

  // Set the seed for random number generator
  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,100)
  vector<int16_t> input(DIM_INPUT_FILL, 0);
  vector<int16_t> weight(DIM_INPUT_FILL * DIM_OUTPUT, 0);
  vector<int16_t> bias(DIM_OUTPUT_FILL, 0);

  for (auto i = 0; i < DIM_INPUT_FILL; i++) {
    if (i < DIM_INPUT) {
      input[i] = rand() % 10;
    }
  }
  for (auto i = 0; i < DIM_OUTPUT; i++) {
    for (auto j = 0; j < DIM_INPUT_FILL; j++) {
      if (j < DIM_INPUT) {
        weight[i * DIM_INPUT_FILL + j] = rand() % 10;
      }
    }
  }
  for (auto i = 0; i < DIM_OUTPUT_FILL; i++) {
    if (i < DIM_OUTPUT) {
      bias[i] = rand() % 10;
    }
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, input.size() / 16, input);
  __input_buffer__.write<int16_t>(input.size() / 16, weight.size() / 16,
                                  weight);
  __input_buffer__.write<int16_t>(input.size() / 16 + weight.size() / 16,
                                  bias.size() / 16, bias);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define DIM_INPUT 24
#define DIM_INPUT_FILL 32
#define DIM_OUTPUT 128
#define DIM_OUTPUT_FILL 128

  // Read input, weight, and bias
  vector<int16_t> input =
      __input_buffer__.read<int16_t>(0, DIM_INPUT_FILL / 16);
  vector<int16_t> weight = __input_buffer__.read<int16_t>(
      DIM_INPUT_FILL / 16, DIM_OUTPUT * DIM_INPUT_FILL / 16);
  vector<int16_t> bias = __input_buffer__.read<int16_t>(
      DIM_INPUT_FILL / 16 + DIM_OUTPUT * DIM_INPUT_FILL / 16,
      DIM_OUTPUT_FILL / 16);
  vector<int16_t> output(DIM_OUTPUT_FILL, 0);
  for (int i = 0; i < DIM_OUTPUT; i++) {
    for (int j = 0; j < DIM_INPUT; j++) {
      output[i] += input[j] * weight[i * DIM_INPUT_FILL + j];
      output[i] += bias[i];
    }
  }
  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, DIM_OUTPUT_FILL / 16, output);

  // print the input, weight, bias, and output
  cout << "Input: " << endl;
  for (int i = 0; i < DIM_INPUT_FILL; i++) {
    cout << input[i] << " ";
  }
  cout << endl;
  cout << "Weight: " << endl;
  for (int i = 0; i < DIM_OUTPUT; i++) {
    for (int j = 0; j < DIM_INPUT_FILL; j++) {
      cout << weight[i * DIM_INPUT_FILL + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
  cout << "Bias: " << endl;
  for (int i = 0; i < DIM_OUTPUT_FILL; i++) {
    cout << bias[i] << " ";
  }
  cout << endl;
  cout << "Output: " << endl;
  for (int i = 0; i < DIM_OUTPUT_FILL; i++) {
    cout << output[i] << " ";
  }
  cout << endl;
  exit(0);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
