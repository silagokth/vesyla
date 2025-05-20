#include "Util.hpp"

void bin_to_hex_file(string bin_file_, string hex_file_) {
  ifstream ifs(bin_file_);
  if (!ifs.is_open()) {
    cout << "Error: Can't open file: " << bin_file_ << endl;
    abort();
  }
  ofstream ofs(hex_file_);
  if (!ofs.is_open()) {
    cout << "Error: Can't open file: " << hex_file_ << endl;
    abort();
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::smatch sm;
    std::regex e1("\\s*(\\d+)\\s+([01]+)\\s*");
    if (std::regex_match(line.cbegin(), line.cend(), sm, e1)) {
      int addr = stoi(sm[1]);
      string raw_data = sm[2];
      string hex_data;
      if (raw_data.size() % 4 != 0) {
        // pad 0s
        raw_data = string(4 - raw_data.size() % 4, '0') + raw_data;
      }
      for (size_t i = 0; i < raw_data.size(); i += 4) {
        unsigned long number = std::bitset<4>(raw_data.substr(i, 4)).to_ulong();
        // convert to hex string
        std::stringstream ss;
        ss << std::hex << number;
        hex_data += ss.str();
      }
      ofs << addr << " " << hex_data << endl;
    }
  }
  ofs.close();
}

IO file_to_io(string file_) {
  IO io;
  ifstream ifs(file_);
  if (!ifs.is_open()) {
    cout << "Error: Can't open file: " << file_ << endl;
    abort();
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::smatch sm;
    std::regex e1("\\s*(\\d+)\\s+([01]+)\\s*");
    if (std::regex_match(line.cbegin(), line.cend(), sm, e1)) {
      int addr = stoi(sm[1]);
      string raw_data = sm[2];
      io.write(addr, 1, raw_data);
    }
  }
  return io;
}
void io_to_file(IO io_, string file_) {
  ofstream ofs(file_);
  if (!ofs.is_open()) {
    cout << "Error: Can't open file: " << file_ << endl;
    abort();
  }
  vector<size_t> active_rows = io_.get_active_rows();
  for (size_t idx : active_rows) {
    ofs << idx << " " << io_.read(idx, 1) << endl;
  }
  ofs.close();
}

// void assemble() {
//   string cmd;
//   cmd = "bash ../scripts/assemble.sh";
//   assert(system(cmd.c_str()) == 0);
// }

// void compile() {
//   string cmd;
//   cmd = "bash ../script/compile.sh ../pasm";
//   assert(system(cmd.c_str()) == 0);
// }

// void simulate_code_segment(int id) {
//   if (current_model == 2) {
//     string cmd;
//     cmd = "bash ../script/instr_sim.sh " + to_string(id);
//     assert(system(cmd.c_str()) == 0);
//   } else if (current_model == 3) {
//     string cmd;
//     cmd = "bash ../script/rtl_sim.sh " + to_string(id);
//     assert(system(cmd.c_str()) == 0);
//   } else {
//     cout << "Error: current_model is not allowed: " << current_model << endl;
//     abort();
//   }
// }

// int run_simulation(void) {
// cout << "Initialization ..." << endl;
// init();
// store_input_data("mem/sram_image_in.bin");
// #ifdef DEBUG
//   bin_to_hex_file("mem/sram_image_in.bin", "mem/sram_image_in.hex");
//   bin_to_num_file<DATA_TYPE>("mem/sram_image_in.bin",
//   "mem/sram_image_in.txt");
// #endif
//   cout << "Run model 0 ..." << endl;
//   reset_all();
//   load_input_data("mem/sram_image_in.bin");
//   current_model = 0;
//   model_l0();
//   store_output_data("mem/sram_image_m0.bin");
// #ifdef DEBUG
//   bin_to_hex_file("mem/sram_image_m0.bin", "mem/sram_image_m0.hex");
//   bin_to_num_file<DATA_TYPE>("mem/sram_image_m0.bin",
//   "mem/sram_image_m0.txt");
// #endif
// cout << "Run model 1 ..." << endl;
// assemble();
// compile();
// reset_all();
// load_input_data("mem/sram_image_in.bin");
// current_model = 2;
// model_l1();
// #ifdef DEBUG
//   bin_to_hex_file("mem/sram_image_m2.bin", "mem/sram_image_m2.hex");
//   bin_to_num_file<DATA_TYPE>("mem/sram_image_m2.bin",
//   "mem/sram_image_m2.txt");
// #endif
//   cout << "Verify model 2 against model 0";
//   assert(verify("mem/sram_image_m0.bin", "mem/sram_image_m2.bin"));
//   cout << " - Success!" << endl;
//   reset_all();
//   load_input_data("mem/sram_image_in.bin");
//   current_model = 3;
//   model_l1();
// #ifdef DEBUG
//   bin_to_hex_file("mem/sram_image_m3.bin", "mem/sram_image_m3.hex");
//   bin_to_num_file<DATA_TYPE>("mem/sram_image_m3.bin",
//   "mem/sram_image_m3.txt");
// #endif
//   cout << "Verify model 3 against model 0";
//   assert(verify("mem/sram_image_m0.bin", "mem/sram_image_m3.bin"));
//   cout << " - Success!" << endl;

const char *great_succes = R"V0G0N(
                                %@%%%@
                           @@@@@@@@@@@@
                         @@@@@@@@@@@@@@@@@@
                        @@@@@@%@@@@@@@@@@@@@
                        @@@@%*++====++#@@@@@
                          @@%*+==-===+*@@@@@
                          @*#@%+=--=*#*%@@@
                          %+#+*++=*=#%+#+@
                           *#+==+=+===+*%
                            #++##=*+++*
                            %+%%##%@%=*
                             +++++*+=+*
               +=            *++==++++#           ==
               *=          +###+#+*+**#**         =
               ++----=   **+#*##*+=##**+++++     =::=
              **=----=++++++##*##+##**#=+++++++-:=-:-=
              *+=-=--#*=+++=+**#=******===+====%----:-#
            #++====-+@=+=====+++=+**++*========*@=-:--*
            *=====@@@========+++==+*++===========@#%++*+
           @=---@#========*===+====*++===+========#%==--%
          *#%%@@+=============+====*++------=======++==-*+
          **=--=*=====--=-=========*+=-----========++++==+*
         ***+==+====-==--==----====*+-----=-====+==*=*++==*
        ++*+*+++==--=--==------====*+----===-=========*+=+*=
        =*++*++===---==------------+=---------=====+=-=+=++=
        =++===+===-----=--------=--+---------==+=-====-++=+=
        ==*==+==+=-----=---------=-=---------====-=++-==+=+=+
        ==++=====+-----==--------==---------======+======+=+
        =++====++==-----==-----=---------=--===+==++====+++=
        +==+++=++==------==-------==----=--=======  +=+=+++
         ==++=*+====--=----=------+---==--=======+    +=++
          %++**  ===-==-----==------===--========+
                 ====-=-------=====+===-========++
                 ======------===-=++=--======+==++
                 +======--======================++
                 +==============================++
   _____                _      _____                             _
  / ____|              | |    / ____|                           | |
 | |  __ _ __ ___  __ _| |_  | (___  _   _  ___ ___ ___  ___ ___| |
 | | |_ | '__/ _ \/ _` | __|  \___ \| | | |/ __/ __/ _ \/ __/ __| |
 | |__| | | |  __/ (_| | |_   ____) | |_| | (_| (_|  __/\__ \__ \_|
  \_____|_|  \___|\__,_|\__| |_____/ \__,_|\___\___\___||___/___(_)
)V0G0N";
//   cout << great_succes << endl;

//   return 0;
// }
