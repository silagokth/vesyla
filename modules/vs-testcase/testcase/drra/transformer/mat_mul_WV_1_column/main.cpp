#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>


int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 17 * 16
  // Set the seed for random number generator
  // (4,4) x (4, 64) = 1 x 16 + 4 x 4 x 16 = 17 x 16

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
#define N 17 * 16
  // Read the input buffer to A of size 64 x 1.
  vector<int16_t> soft_matrix = __input_buffer__.read<int16_t>(0, 1);

  vector<int16_t> v_matrix = __input_buffer__.read<int16_t>(1, 16);

  vector<int16_t> output;


cout << "soft_matrix is as below" << endl;
  for (int i = 0; i < 16; i++){
    cout << soft_matrix[i] << " ";
  }
cout << endl;


cout << "v_matrix is as below" << endl;
  for (int i = 0; i < 16 * 16; i++){
    cout << v_matrix[i] << " ";
  }
cout << endl;


  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 64; j++){
      int16_t acc = 0;
      for (int k = 0; k < 4; k++){
         //cout << "current soft: "<< soft_matrix[i * 4 + k] << endl;
         //cout << "current v: "<< v_matrix[k * 64 + j] << endl;
         //cout << "-----next------" << endl;

        acc = acc + soft_matrix[i * 4 + k] * v_matrix[k + j * 4];
      }
        //cout << "output: "<< acc << endl;
        output.push_back(acc);
    }
  }


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
