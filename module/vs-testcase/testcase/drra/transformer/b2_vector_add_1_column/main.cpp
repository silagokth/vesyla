#include "Drra.hpp"
#include "Util.hpp"
#include <cstdlib>


int main() { return run_simulation(); }

/*
 * Generate the input SRAM image.
 */
void init() {
#define N 8 * 16


  srand((unsigned)time(NULL));

  // Generate N random numbers in range [0,10)
  vector<int16_t> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = rand() % 10 ;
  }

  // Write the random numbers to the input buffer.
  __input_buffer__.write<int16_t>(0, N / 16, v);
}

/*
 * Define the reference algorithm model. It will generate the reference output
 * SRAM image. You can use free C++ programs.
 */
void model_l0() {
#define N 8 * 16
  

  vector<int16_t> x_vector = __input_buffer__.read<int16_t>(0, 4);

  vector<int16_t> b_vector = __input_buffer__.read<int16_t>(4, 4);

  vector<int16_t> output;


cout << "x_vector is as below" << endl;
  for (int i = 0; i < 64; i++){
    cout << x_vector[i] << " ";
  }
cout << endl;

cout << "b_vector is as below" << endl;
  for (int i = 0; i < 64; i++){
    cout << b_vector[i] << " ";
  }
cout << endl;


  int16_t sum = 0;


    for (int j = 0; j < 64; j++){
        sum = x_vector[j] + b_vector[j];
        output.push_back(sum);
      }
        //cout << "output: "<< acc << endl;
    
  


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
