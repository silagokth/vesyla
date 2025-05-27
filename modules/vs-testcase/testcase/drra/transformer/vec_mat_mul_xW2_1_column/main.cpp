#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>


int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 1040 * 16
  // (1, 256) x (256, 64)
  // = 16x16 + 1024 x 16 = 1040 x 16
  // Set the seed for random number generator

  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,10)
  vector<int16_t> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = rand() % 10;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, N / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 1040 * 16
  // Read the input buffer to A of size 64 x 1.
  
  //vector<int16_t> soft_matrix = __input_buffer__.read<int16_t>(0, 1);

  vector<int16_t> x_vector = __input_buffer__.read<int16_t>(0, 16);

  //vector<int16_t> v_matrix = __input_buffer__.read<int16_t>(1, 16);

  vector<int16_t> w_matrix = __input_buffer__.read<int16_t>(16, 1024);

  vector<int16_t> output;


cout << "x_vector is as below" << endl;
  
 
  for (int j = 0; j < 256; j++){
    cout << x_vector[j] << " ";
  }

cout << endl;


cout << "w_matrix is as below" << endl;
  for (int i = 0; i < 64; i++){
    for (int j = 0; j < 256; j++){
      cout << w_matrix[i * 256 + j] << " ";
    }
    cout << endl;
  }
cout << endl;


  for (int i = 0; i < 1; i++){
    for (int j = 0; j < 64; j++){
      int16_t acc = 0;
      for (int k = 0; k < 256; k++){
         //cout << "current soft: "<< x_vector[i + k] << endl;
         //cout << "current v: "<< w_matrix[k + j * 64] << endl;
         //cout << "-----next------" << endl;

        acc = acc + x_vector[i + k] * w_matrix[k + j * 256];
      }
        //cout << "output: "<< acc << endl;
        output.push_back(acc);
    }
  }


cout << "result vector is as below" << endl;
for(int i = 0; i < 64; i++){
  cout << output[i] << " ";
}
cout << endl;


  // Write A to the output buffer
  __output_buffer__.write<int16_t>(0, 4, output);
}

/*
 * Define the DRR algorithm model. It will generate the DRR output SRAM image by
 * simulate the instruction execution using python simulator.
 */
void model_l1() { simulate_code_segment(0); }
