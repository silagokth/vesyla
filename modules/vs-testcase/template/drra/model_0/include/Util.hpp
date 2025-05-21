/**
 * This file contains the utility functions for debugging.
 */

#ifndef __UTIL_HPP__
#define __UTIL_HPP__

// #include "Drra.hpp"
#include "IO.hpp"
#include "Stream.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>

/**
 * Print the contents of a vector.
 * @param vec The array to print.
 */
template <typename T> void print_vector(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << ",";
  }
  std::cout << std::endl;
}

template <typename T> T decode_bitset_token(bitset<sizeof(T) * 8> raw_data_) {
  T token = 0;
  if (sizeof(T) == sizeof(uint64_t)) {
    uint64_t x = (*(uint64_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint32_t)) {
    uint32_t x = (*(uint32_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint16_t)) {
    uint16_t x = (*(uint16_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint8_t)) {
    uint8_t x = (*(uint8_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else {
    abort();
  }
  return token;
}

template <size_t chunk_size>
vector<std::bitset<chunk_size>> chop_bitset(std::bitset<chunk_size> raw_data_,
                                            size_t size_) {
  assert(size_ <= chunk_size);
  assert(chunk_size % size_ == 0);
  vector<std::bitset<chunk_size>> vec;
  for (size_t i = 0; i < chunk_size / size_; i++) {
    std::bitset<chunk_size> section_i = 0;
    for (size_t j = 0; j < size_; j++) {
      section_i[j] = raw_data_[i * size_ + j];
    }
    vec.push_back(section_i);
  }
  return vec;
}

template <size_t chunk_size, typename T>
vector<T> decode_bitset(std::bitset<chunk_size> raw_data_) {
  vector<std::bitset<chunk_size>> vec =
      chop_bitset<chunk_size>(raw_data_, sizeof(T) * 8);
  vector<T> dec_vec(vec.size(), 0);
  for (size_t i = 0; i < vec.size(); i++) {
    std::bitset<sizeof(T) * 8> tt = vec[i].to_ulong();
    dec_vec[i] = decode_bitset_token<T>(tt);
  }
  return dec_vec;
}

/**
 * Print the contents of a stream of bitset.
 * @param stream The stream to print.
 */
template <size_t chunk_size, typename T>
void print_bitset_stream(Stream<bitset<chunk_size>> stream) {
  for (size_t i = 0; i < stream.size(); i++) {
    auto raw_data = stream[i];
    auto vec = decode_bitset<chunk_size, T>(raw_data);
    print_vector<T>(vec);
  }
}

template <typename T> void bin_to_num_file(string bin_file_, string num_file_) {
  ifstream ifs(bin_file_);
  if (!ifs.is_open()) {
    cout << "Error: Can't open file: " << bin_file_ << endl;
    abort();
  }
  ofstream ofs(num_file_);
  if (!ofs.is_open()) {
    cout << "Error: Can't open file: " << num_file_ << endl;
    abort();
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::smatch sm;
    std::regex e1("\\s*(\\d+)\\s+([01]+)\\s*");
    if (std::regex_match(line.cbegin(), line.cend(), sm, e1)) {
      int addr = stoi(sm[1]);
      string raw_data = sm[2];
      // check raw_data size must be multiple of the size of T
      if (raw_data.size() % (sizeof(T) * 8) != 0) {
        cout << "Error: raw_data size is not multiple of " << sizeof(T) * 8
             << endl;
        abort();
      }
      // chop raw_data into chunks
      ofs << addr << " [";
      vector<std::bitset<sizeof(T) * 8>> vec;
      for (size_t i = 0; i < raw_data.size(); i += sizeof(T) * 8) {
        vec.push_back(
            std::bitset<sizeof(T) * 8>(raw_data.substr(i, sizeof(T) * 8)));
      }
      for (size_t i = 0; i < vec.size(); i++) {
        unsigned long num = vec[i].to_ulong();
        T value;
        // use memcpy to fill the value
        memcpy(&value, &num, sizeof(T));
        ofs << value;
        if (i < vec.size() - 1) {
          ofs << ", ";
        }
      }
      ofs << "]" << endl;
    }
  }
  ofs.close();
}

void bin_to_hex_file(string bin_file_, string hex_file_);

IO file_to_io(string file_);
void io_to_file(IO io_, string file_);

#endif // __UTIL_HPP__