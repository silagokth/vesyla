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
  static const std::regex e1("\\s*(\\d+)\\s+([01]+)\\s*");
  while (std::getline(ifs, line)) {
    std::smatch sm;
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