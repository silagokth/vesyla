#ifndef _CUS_BACKING_H
#define _CUS_BACKING_H

#include "Array.hpp"
#include "sst/elements/memHierarchy/membackend/backing.h"
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_WIDTH 1024
#define MAX_DEPTH 65536

namespace SST {
namespace MemHierarchy {
namespace Backend {

class BackingIO : public Backing {
private:
  std::string memory_file_;
  size_t width_, depth_;
  bool read_only_ = true;
  Array<MAX_DEPTH, MAX_WIDTH> *io_;

public:
  BackingIO(std::string memory_file, size_t io_width, size_t io_depth,
            bool read_only)
      : Backing(), memory_file_(memory_file), width_(io_width),
        depth_(io_depth), read_only_(read_only) {
    // Check if exceeding maximum width or depth
    if ((io_width > MAX_WIDTH) || (io_depth > MAX_DEPTH)) {
      throw std::runtime_error("Exceeding maximum width or depth");
    }
    io_ = new Array<MAX_DEPTH, MAX_WIDTH>();

    // Load memory file
    ifstream ifs(memory_file);
    if (!ifs.is_open()) {
      cout << "Error: Can't open file: " << memory_file << endl;
      abort();
    }

    std::string current_line;
    while (std::getline(ifs, current_line)) {
      std::smatch sm;
      std::regex e1("\\s*(\\d+)\\s+([01]+)\\s*");
      if (std::regex_match(current_line.cbegin(), current_line.cend(), sm,
                           e1)) {
        int addr = stoi(sm[1]);
        string raw_data = sm[2];
        raw_data.resize(MAX_WIDTH, '0');
        io_->write(addr, 1, raw_data);
      }
    }
  }

  ~BackingIO() {
    // Dump memory to file
    if (!read_only_)
      dump(nullptr);
    delete io_;
  }

  void set(Addr addr, uint8_t value) {
    string raw_data = std::bitset<8>(value).to_string();
    raw_data.resize(MAX_WIDTH, '0');
    io_->write(addr, 1, raw_data);
  }

  void set(Addr addr, size_t size, std::vector<uint8_t> &data) {
    string raw_data;
    for (size_t i = 0; i < size; i++) {
      raw_data += std::bitset<8>(data[i]).to_string();
    }
    // Pad 0s if size is not multiple of width
    raw_data.resize(MAX_WIDTH, '0');
    io_->write(addr, 1, raw_data);
  }

  uint8_t get(Addr addr) {
    string raw_data = io_->read(addr, 1);
    return std::bitset<8>(raw_data).to_ulong();
  }

  void get(Addr addr, size_t size, std::vector<uint8_t> &data) {
    string raw_data = io_->read(addr, size);
    // printf("raw_data: %s\n", raw_data.c_str());
    for (size_t i = 0; i < width_ / 8; i++) {
      data.push_back(
          std::bitset<8>(raw_data.substr(width_ - 8 * (i + 1), 8)).to_ulong());
    }
  }

  void dump(FILE *fp) {
    ofstream ofs(memory_file_);
    if (!ofs.is_open()) {
      cout << "Error: Can't open file: " << memory_file_ << endl;
      abort();
    }
    vector<size_t> active_rows = io_->get_active_rows();
    std::string current_row;
    for (size_t idx : active_rows) {
      current_row = io_->read(idx, 1);
      current_row.resize(width_, '0');
      ofs << idx << " " << current_row << endl;
    }
    ofs.close();
  }

  void setReadOnly(bool ro) { read_only_ = ro; }
};
} // namespace Backend
} // namespace MemHierarchy
} // namespace SST

#endif // _CUS_BACKING_H