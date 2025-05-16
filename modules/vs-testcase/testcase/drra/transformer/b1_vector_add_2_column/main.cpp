#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>


int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 32 * 16
  // (1, 256)
  // = 16 x 16
  // Set the seed for random number generator

  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,10)
  vector<int16_t> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = rand() % 21 - 10 ;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, N / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 32 * 16
  // Read the input buffer to A of size 64 x 1.
  


  vector<int16_t> x_vector = __input_buffer__.read<int16_t>(0, 16);

  vector<int16_t> b_vector = __input_buffer__.read<int16_t>(16, 16);

  vector<int16_t> output;


cout << "x_vector is as below" << endl;
  for (int i = 0; i < 256; i++){
    cout << x_vector[i] << " ";
  }
cout << endl;

cout << "b_vector is as below" << endl;
  for (int i = 0; i < 256; i++){
    cout << b_vector[i] << " ";
  }
cout << endl;


  int16_t sum = 0;
  //int16_t payload = 0;
  //int16_t zero = 0;

    for (int j = 0; j < 256; j++){
         //cout << "current soft: "<< x_vector[i + k] << endl;
         //cout << "current v: "<< w_matrix[k + j * 64] << endl;
         //cout << "-----next------" << endl;
        sum = x_vector[j] + b_vector[j];
        //payload = max(zero, sum);
        output.push_back(sum);

      }
        //cout << "output: "<< acc << endl;
    
  


cout << "result vector is as below" << endl;
for(int i = 0; i < 256; i++){
  cout << output[i] << " ";
}
cout << endl;


  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, 16, output);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
